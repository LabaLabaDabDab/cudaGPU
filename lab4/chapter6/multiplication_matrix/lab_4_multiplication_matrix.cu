#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// Размер блока
#define BLOCK_DIM 16
// Тип элементов матрицы
#define BASE_TYPE double

// Ядро: C = A * B
__global__ void matrixMult(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Arows, int Acols, int Bcols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= Arows || col >= Bcols) {
        return;
    }

    BASE_TYPE sum = 0.0;
    for (int k = 0; k < Acols; ++k) {
        sum += A[row * Acols + k] * B[k * Bcols + col];
    }

    C[row * Bcols + col] = sum;
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

// Перемножение матриц на CPU: C = A * B
void matrixMultCPU(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Arows, int Acols, int Bcols) {
    for (int i = 0; i < Arows; ++i) {
        for (int j = 0; j < Bcols; ++j) {
            BASE_TYPE sum = 0.0;
            for (int k = 0; k < Acols; ++k) {
                sum += A[i * Acols + k] * B[k * Bcols + j];
            }
            C[i * Bcols + j] = sum;
        }
    }
}

// Проверка совпадения C_cpu и C_gpu
bool compareMatrices(const BASE_TYPE* C_cpu, const BASE_TYPE* C_gpu, int Arows, int Bcols, double eps = 1e-6) {
    for (int i = 0; i < Arows; ++i) {
        for (int j = 0; j < Bcols; ++j) {
            int idx = i * Bcols + j;
            BASE_TYPE a = C_cpu[idx];
            BASE_TYPE b = C_gpu[idx];
            if (fabs(a - b) > eps) {
                cout << "Mismatch at element (" << i << ", " << j << "): "
                     << "CPU=" << a << ", GPU=" << b
                     << ", diff=" << fabs(a - b) << endl;
                return false;
            }
        }
    }
    return true;
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
    int Arows = 100;
    int Acols = 200;
    int Brows = Acols;
    int Bcols = 150;

    // Делаем размеры кратными BLOCK_DIM
    Arows = toMultiple(Arows, BLOCK_DIM);
    Acols = toMultiple(Acols, BLOCK_DIM);
    Brows = Acols; // B должно иметь столько строк, сколько столбцов у A
    Bcols = toMultiple(Bcols, BLOCK_DIM);

    const size_t Asize = static_cast<size_t>(Arows) * Acols * sizeof(BASE_TYPE); // Arows x Acols
    const size_t Bsize = static_cast<size_t>(Brows) * Bcols * sizeof(BASE_TYPE); // Brows x Bcols
    const size_t Csize = static_cast<size_t>(Arows) * Bcols * sizeof(BASE_TYPE); // Arows x Bcols

    cout << "===== TASK PARAMETERS =====" << endl;
    cout << "A: " << Arows << " x " << Acols << endl;
    cout << "B: " << Brows << " x " << Bcols << endl;
    cout << "C: " << Arows << " x " << Bcols << endl;
    cout << fixed << setprecision(2);
    cout << "Data size A: " << Asize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size B: " << Bsize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size C: " << Csize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Total data (A+B+C): " << (Asize + Bsize + Csize) / (1024.0 * 1024.0) << " MB" << endl << endl;

    // ===== Счётчики времени для CPU =====
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    cout << "===== CPU PHASE =====" << endl;

    // Память на CPU
    BASE_TYPE* h_A = static_cast<BASE_TYPE*>(malloc(Asize));
    BASE_TYPE* h_B = static_cast<BASE_TYPE*>(malloc(Bsize));
    BASE_TYPE* h_C_cpu = static_cast<BASE_TYPE*>(malloc(Csize));
    BASE_TYPE* h_C_gpu = static_cast<BASE_TYPE*>(malloc(Csize));

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        cerr << "Host memory allocation failed" << endl;
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if (h_C_cpu) free(h_C_cpu);
        if (h_C_gpu) free(h_C_gpu);
        return 1;
    }

    // Инициализируем матрицы случайными числами
    srand(0);
    initMatrixRandom(h_A, Arows, Acols);
    initMatrixRandom(h_B, Brows, Bcols);

    // Замер времени умножения матриц на CPU
    cudaEventRecord(start_cpu, 0);
    matrixMultCPU(h_A, h_B, h_C_cpu, Arows, Acols, Bcols);
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);

    float cpu_time = 0.0f;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
    cout << setprecision(6);
    cout << "CPU matrix multiplication time: " << cpu_time << " ms" << endl << endl;

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
    BASE_TYPE* d_A = nullptr;
    BASE_TYPE* d_B = nullptr;
    BASE_TYPE* d_C = nullptr;

    // Выделение памяти на GPU
    cudaEventRecord(start_malloc, 0);

    cuda_error = cudaMalloc((void**)&d_A, Asize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_A) failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_B, Bsize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_B) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_C, Csize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_C) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 1;
    }

    cudaEventRecord(stop_malloc, 0);
    cudaEventSynchronize(stop_malloc);
    cudaEventElapsedTime(&malloc_time, start_malloc, stop_malloc);
    cout << "GPU allocation time: " << malloc_time << " ms" << endl;

    // Копируем A и B с CPU на GPU (H2D)
    cudaEventRecord(start_h2d, 0);

    cuda_error = cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy h_A -> d_A failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 1;
    }

    cuda_error = cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy h_B -> d_B failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 1;
    }

    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d);
    cout << "Host->device copy time (A+B): " << h2d_time << " ms" << endl;

    // ===== Настройка grid / block =====
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 blocksPerGrid(
        (Bcols + BLOCK_DIM - 1) / BLOCK_DIM,
        (Arows + BLOCK_DIM - 1) / BLOCK_DIM
    );

    cout << "Grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << endl;
    cout << "Block: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << endl;

    // Запуск ядра (одна итерация)
    cudaEventRecord(start_kernel, 0);

    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Arows, Acols, Bcols);

    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        cerr << "Kernel launch failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 1;
    }

    // Дожидаемся завершения всех запущенных CUDA-ядер
    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 1;
    }

    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    cout << "Kernel time: " << kernel_time << " ms" << endl;

    // Копируем матрицу C с GPU на CPU
    cudaEventRecord(start_d2h, 0);

    cuda_error = cudaMemcpy(h_C_gpu, d_C, Csize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);

    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy d_C -> h_C failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 1;
    }

    cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h);
    cout << "Device->host copy time (C): " << d2h_time << " ms" << endl;

    // Общее время работы GPU
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_gpu_time, start_total, stop_total);

    cout << "Total GPU time: " << total_gpu_time << " ms" << endl << endl;

    // ===== ПРОВЕРКА РЕЗУЛЬТАТА =====
    cout << "===== RESULT CHECK =====" << endl;

    bool equal = compareMatrices(h_C_cpu, h_C_gpu, Arows, Bcols);
    if (equal) {
        cout << "CPU and GPU results are identical (within tolerance)." << endl;
    } else {
        cout << "CPU and GPU results differ!" << endl;
    }

    // ===== КРАТКАЯ СВОДКА ПО ПРОИЗВОДИТЕЛЬНОСТИ =====
    cout << endl;
    cout << "===== PERFORMANCE SUMMARY =====" << endl;
    cout << "CPU matrix multiplication: " << cpu_time << " ms" << endl;
    cout << "GPU total time: " << total_gpu_time << " ms" << endl;
    cout << "GPU kernel time: " << kernel_time << " ms" << endl;

    if (cpu_time > 0.0f && kernel_time > 0.0f) {
        double speedup_kernel = cpu_time / kernel_time;
        double speedup_total = cpu_time / total_gpu_time;

        cout << "Speedup (CPU / GPU kernel only): " << fixed << setprecision(3) << speedup_kernel << "x" << endl;
        cout << "Speedup (CPU / GPU total): " << speedup_total << "x" << endl;
    }
 
    // ===== СВОДКА ПО ПЕРЕДАЧЕ ДАННЫХ =====
    cout << endl;
    cout << "===== DATA TRANSFER =====" << endl;
    cout << "Data size A: " << fixed << setprecision(2) << Asize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size B: " << Bsize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size C: " << Csize / (1024.0 * 1024.0) << " MB" << endl;
    cout << setprecision(6);
    cout << "Allocation time: " << malloc_time << " ms" << endl;
    cout << "Host->device time: " << h2d_time << " ms" << endl;
    cout << "Device->host time: " << d2h_time << " ms" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

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
