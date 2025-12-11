#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace std;

// Размер блока
#define BLOCK_DIM 16
// Тип элементов матрицы
#define BASE_TYPE float

// Ядро: транспонирование матрицы на GPU
__global__ void matrixTranspose(const BASE_TYPE* A, BASE_TYPE* AT, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) {
        return;
    }

    // Индекс в исходной матрице A (по строкам)
    int iA  = row * cols + col;
    // Индекс в транспонированной матрице AT (там размеры cols x rows)
    int iAT = col * rows + row;

    AT[iAT] = A[iA];
}

// Функция вычисления числа, которое >= a и кратно b
int toMultiple(int a, int b) {
    int mod = a % b;
    if (mod != 0) {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

// Инициализация матрицы случайными значениями [0;1)
void initMatrixRandom(BASE_TYPE* A, int rows, int cols) {
    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        A[i] = static_cast<BASE_TYPE>(rand()) / static_cast<BASE_TYPE>(RAND_MAX);
    }
}

// Транспонирование матрицы на CPU
void transposeCPU(const BASE_TYPE* A, BASE_TYPE* AT, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            AT[j * rows + i] = A[i * cols + j];
        }
    }
}

// Проверка совпадения транспонированных матриц (CPU и GPU)
bool compareTransposed(const BASE_TYPE* AT_cpu, const BASE_TYPE* AT_gpu, int rows, int cols) {
    bool ok = true;
    // AT: cols x rows
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            int idx = j * rows + i;
            BASE_TYPE a = AT_cpu[idx];
            BASE_TYPE b = AT_gpu[idx];
            if (a != b) {
                cout << "Mismatch at index " << idx << " (row=" << i << ", col=" << j << "): " << "CPU=" << a << ", GPU=" << b << endl;
                ok = false;
                return ok;
            }
        }
    }
    return ok;
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    cudaError_t cuda_error;

    cuda_error = cudaSetDevice(device);
    if (cuda_error != cudaSuccess) {
        cout << "Error selecting device: " << cudaGetErrorString(cuda_error) << endl;
        return 1;
    }

    cuda_error = cudaGetDeviceProperties(&prop, device);
    if (cuda_error != cudaSuccess) {
        cout << "Error getting device properties: " << cudaGetErrorString(cuda_error) << endl;
        return 1;
    }

    cout << "===== DEVICE PROPERTIES =====" << endl;
    cout << "Device name: " << prop.name << endl;
    cout << "Number of multiprocessors: " << prop.multiProcessorCount << endl;
    cout << "Global memory size: " << prop.totalGlobalMem << " bytes" << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Max grid size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
    cout << "Max block dimensions: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
    cout << endl;

    // ===== ПАРАМЕТРЫ ЗАДАЧИ =====
    int rows = 1000;
    int cols = 2000;

    // Делаем размеры кратными BLOCK_DIM
    rows = toMultiple(rows, BLOCK_DIM);
    cols = toMultiple(cols, BLOCK_DIM);

    const size_t sizeA = rows * cols * sizeof(BASE_TYPE);
    const size_t sizeAT = cols * rows * sizeof(BASE_TYPE);

    cout << "===== TASK PARAMETERS =====" << endl;
    cout << "Original matrix size (rows x cols): " << rows << " x " << cols << endl;
    cout << "Transposed matrix size (cols x rows): " << cols << " x " << rows << endl;
    cout << fixed << setprecision(2);
    cout << "Data size A : " << sizeA / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size AT: " << sizeAT / (1024.0 * 1024.0) << " MB" << endl << endl;

    // ===== Счётчики времени для CPU =====
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    cout << "===== CPU PHASE =====" << endl;

    // ===== Память на CPU =====
    BASE_TYPE* h_A = static_cast<BASE_TYPE*>(malloc(sizeA));
    BASE_TYPE* h_AT_cpu = static_cast<BASE_TYPE*>(malloc(sizeAT));
    BASE_TYPE* h_AT_gpu = static_cast<BASE_TYPE*>(malloc(sizeAT));

    if (!h_A || !h_AT_cpu || !h_AT_gpu) {
        cerr << "Host memory allocation failed" << endl;
        if (h_A) free(h_A);
        if (h_AT_cpu) free(h_AT_cpu);
        if (h_AT_gpu) free(h_AT_gpu);
        return 1;
    }

    // Инициализируем матрицу случайными числами
    srand(0);
    initMatrixRandom(h_A, rows, cols);

    // Замер времени транспонирования на CPU
    cudaEventRecord(start_cpu, 0);
    transposeCPU(h_A, h_AT_cpu, rows, cols);
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);

    float cpu_time = 0.0f;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
    cout << setprecision(6);
    cout << "CPU transpose time: " << cpu_time << " ms" << endl << endl;

    cout << "===== GPU =====" << endl;

    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_malloc, stop_malloc;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;

    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_malloc);
    cudaEventCreate(&stop_malloc);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    float total_gpu_time = 0.0f;
    float malloc_time = 0.0f;
    float h2d_time = 0.0f;
    float kernel_time = 0.0f;
    float d2h_time = 0.0f;

    cudaEventRecord(start_total, 0);

    // Память на GPU
    BASE_TYPE* d_A  = nullptr;
    BASE_TYPE* d_AT = nullptr;

    // Выделение памяти на GPU
    cudaEventRecord(start_malloc, 0);
    cuda_error = cudaMalloc((void**)&d_A, sizeA);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_A) failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A);
        free(h_AT_cpu);
        free(h_AT_gpu);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_AT, sizeAT);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_AT) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        free(h_A);
        free(h_AT_cpu);
        free(h_AT_gpu);
        return 1;
    }

    cudaEventRecord(stop_malloc, 0);
    cudaEventSynchronize(stop_malloc);
    cudaEventElapsedTime(&malloc_time, start_malloc, stop_malloc);
    cout << "GPU allocation time: " << malloc_time << " ms" << endl;

    // Копируем A с CPU на GPU (H2D)
    cudaEventRecord(start_h2d, 0);
    cuda_error = cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);

    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy host->device failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_AT);
        free(h_A);
        free(h_AT_cpu);
        free(h_AT_gpu);
        return 1;
    }
    cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d);
    cout << "Host->device copy time: " << h2d_time << " ms" << endl;

    // ===== Настройка grid / block =====
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 blocksPerGrid(
        (cols + BLOCK_DIM - 1) / BLOCK_DIM,
        (rows + BLOCK_DIM - 1) / BLOCK_DIM
    );

    cout << "Grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << endl;
    cout << "Block: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << endl;

    // Запуск ядра
    cudaEventRecord(start_kernel, 0);
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_AT, rows, cols);
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        cerr << "Kernel launch failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_AT);
        free(h_A);
        free(h_AT_cpu);
        free(h_AT_gpu);
        return 1;
    }

    // Дожидаемся завершения всех запущенных CUDA-ядер
    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_AT);
        free(h_A);
        free(h_AT_cpu);
        free(h_AT_gpu);
        return 1;
    }

    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    cout << "Kernel time: " << kernel_time << " ms" << endl;

    // Копируем матрицу AT с GPU на CPU
    cudaEventRecord(start_d2h, 0);
    cuda_error = cudaMemcpy(h_AT_gpu, d_AT, sizeAT, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);

    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy device->host failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_AT);
        free(h_A);
        free(h_AT_cpu);
        free(h_AT_gpu);
        return 1;
    }

    // Считаем время копирования данных с GPU на CPU
    cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h);
    cout << "Device->host copy time: " << d2h_time << " ms" << endl;

    // Общее время работы GPU
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_gpu_time, start_total, stop_total);

    cout << "Total GPU time: " << total_gpu_time << " ms" << endl << endl;

    // ===== ПРОВЕРКА РЕЗУЛЬТАТА =====
    cout << "===== RESULT CHECK =====" << endl;

    // Полностью сравниваем матрицу, созданную на CPU, с матрицей, полученной с GPU
    bool equal = compareTransposed(h_AT_cpu, h_AT_gpu, rows, cols);
    if (equal) {
        cout << "CPU and GPU transposed matrices are identical." << endl;
    } else {
        cout << "CPU and GPU transposed matrices differ!" << endl;
    }

    // ===== КРАТКАЯ СВОДКА ПО ПРОИЗВОДИТЕЛЬНОСТИ =====
    cout << endl;
    cout << "===== PERFORMANCE SUMMARY =====" << endl;
    cout << "CPU transpose time: " << cpu_time << " ms" << endl;
    cout << "GPU total time: " << total_gpu_time << " ms" << endl;
    cout << "GPU kernel time: " << kernel_time << " ms" << endl;

    // Считаем ускорение GPU относительно CPU
    if (cpu_time > 0.0f && kernel_time > 0.0f) {
        double speedup_kernel = cpu_time / kernel_time;
        double speedup_total = cpu_time / total_gpu_time;

        cout << "Speedup (CPU / GPU kernel only): " << fixed << setprecision(3) << speedup_kernel << "x" << endl;
        cout << "Speedup (CPU / GPU total): " << speedup_total << "x" << endl;
    }

    // ===== СВОДКА ПО ПЕРЕДАЧЕ ДАННЫХ =====
    cout << endl;
    cout << "===== DATA TRANSFER =====" << endl;
    cout << "Data size A : " << fixed << setprecision(2) << sizeA / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size AT: " << sizeAT / (1024.0 * 1024.0) << " MB" << endl;
    cout << setprecision(6);
    cout << "Allocation time: " << malloc_time << " ms" << endl;
    cout << "Host->device time: " << h2d_time << " ms" << endl;
    cout << "Device->host time: " << d2h_time << " ms" << endl;

    cudaFree(d_A);
    cudaFree(d_AT);
    free(h_A);
    free(h_AT_cpu);
    free(h_AT_gpu);

    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_malloc);
    cudaEventDestroy(stop_malloc);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    return 0;
}
