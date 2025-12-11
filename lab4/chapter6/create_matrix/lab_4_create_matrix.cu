#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace std;

#define BLOCK_DIM 16

// Ядро: создаёт элементы матрицы на GPU.
__global__ void createMatrix(int* A, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        int idx = row * n + col;
        A[idx] = 10 * row + col;
    }
}

// Заполнение матрицы A на CPU
void fillMatrixHost(int* A, int n) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int idx = j * n + i;
            A[idx] = 10 * j + i;
        }
    }
}

// Проверка совпадения двух матриц A и B на CPU
bool compareMatrices(const int* A, const int* B, int n) {
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            if (A[idx] != B[idx]) {
                printf("Mismatch at index %d (row=%d, col=%d): A=%d, B=%d\n", idx, i, j, A[idx], B[idx]);
                ok = false;
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
    const int n = 10000;
    const size_t size = n * n * sizeof(int);

    cout << "===== TASK PARAMETERS =====" << endl;
    cout << "Matrix size: " << n << " x " << n << endl;
    cout << "Total elements: " << n * n << endl;
    cout << fixed << setprecision(2);
    cout << "Total data size: " << size / (1024.0 * 1024.0) << " MB" << endl << endl;

    // ===== Счётчики времени для CPU =====
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    // ===== Память на CPU =====
    cout << "===== CPU DATA PREPARATION =====" << endl;

    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);

    if (!h_A || !h_B) {
        cerr << "Host memory allocation failed" << endl;
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        return 1;
    }

    // Замер времени создания матрицы на CPU
    cudaEventRecord(start_cpu, 0);
    fillMatrixHost(h_A, n);
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);

    float cpu_time = 0.0f;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
    cout << setprecision(6);
    cout << "CPU matrix creation time: " << cpu_time << " ms" << endl << endl;

    cout << "===== GPU =====" << endl;

    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_malloc, stop_malloc;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_memcpy, stop_memcpy;

    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_malloc);
    cudaEventCreate(&stop_malloc);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_memcpy);
    cudaEventCreate(&stop_memcpy);

    float total_gpu_time = 0.0f;
    float malloc_time = 0.0f;
    float kernel_time = 0.0f;
    float memcpy_time = 0.0f;

    cudaEventRecord(start_total, 0);

    int* d_B = nullptr;

    // ===== Память на GPU =====
    cudaEventRecord(start_malloc, 0);
    cuda_error = cudaMalloc((void**)&d_B, size);
    cudaEventRecord(stop_malloc, 0);
    cudaEventSynchronize(stop_malloc);
    if (cuda_error != cudaSuccess) {
        cout << "cudaMalloc(d_B) failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A);
        free(h_B);
        return 1;
    }

    cudaEventElapsedTime(&malloc_time, start_malloc, stop_malloc);
    cout << "GPU memory allocation time: " << malloc_time << " ms" << endl;

    // ===== Настройка grid / block =====
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 blocksPerGrid(
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    cout << "Grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << endl;
    cout << "Block: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << endl;

    const int iterations = 10;
    cout << "Kernel iterations: " << iterations << endl;

    // Запуск ядра
    cudaEventRecord(start_kernel, 0);

    // Запускаем ядро несколько раз подряд, чтобы померить среднее время работы
    for (int iter = 0; iter < iterations; ++iter) {
        createMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_B, n);
        cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            cerr << "Kernel launch failed on iteration " << iter << ": " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(d_B);
            free(h_A);
            free(h_B);
            return 1;
        }
    }

    // Дожидаемся завершения всех запущенных CUDA-ядер
    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_B);
        free(h_A);
        free(h_B);
        return 1;
    }

    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

    // Среднее время выполнения одного запуска ядра
    float avg_kernel_time = kernel_time / iterations;
    cout << "Kernel time (total): " << kernel_time << " ms" << endl;
    cout << "Kernel time (per iter): " << avg_kernel_time << " ms" << endl;

    // Копируем матрицу B с GPU на CPU
    cudaEventRecord(start_memcpy, 0);
    cuda_error = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_memcpy, 0);
    cudaEventSynchronize(stop_memcpy);

    // Проверяем, что копирование с устройства на хост прошло успешно
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy device->host failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_B);
        free(h_A);
        free(h_B);
        return 1;
    }

    // Считаем время копирования данных с GPU на CPU
    cudaEventElapsedTime(&memcpy_time, start_memcpy, stop_memcpy);
    cout << "Device->host copy time: " << memcpy_time << " ms" << endl;

    // Считаем общее время работы GPU
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_gpu_time, start_total, stop_total);

    cout << "Total GPU time: " << total_gpu_time << " ms" << endl << endl;

    // ===== ПРОВЕРКА РЕЗУЛЬТАТА =====
    cout << "===== RESULT CHECK =====" << endl;

    // Полностью сравниваем матрицу, созданную на CPU, с матрицей, полученной с GPU
    bool equal = compareMatrices(h_A, h_B, n);
    if (equal) {
        cout << "Matrices A and B are identical." << endl;
    } else {
        cout << "Matrices A and B are NOT identical." << endl;
    }

    // ===== КРАТКАЯ СВОДКА ПО ПРОИЗВОДИТЕЛЬНОСТИ =====
    cout << endl;
    cout << "===== PERFORMANCE SUMMARY =====" << endl;
    cout << "CPU time (1 matrix): " << cpu_time << " ms" << endl;
    cout << "GPU time (total): " << total_gpu_time << " ms" << endl;
    cout << "GPU kernel (" << iterations << " matrices): " << kernel_time << " ms" << endl;
    cout << "GPU kernel per matrix: " << avg_kernel_time << " ms" << endl;

    // Считаем ускорение GPU относительно CPU
    if (cpu_time > 0.0f && avg_kernel_time > 0.0f) {
        double speedup_total = cpu_time / total_gpu_time;
        double speedup_kernel = cpu_time / avg_kernel_time;

        cout << "Speedup (CPU / GPU total): " << fixed << setprecision(3) << speedup_total << "x" << endl;
        cout << "Speedup (CPU / GPU kernel only): " << speedup_kernel << "x" << endl;
    }

    // ===== СВОДКА ПО ПЕРЕДАЧЕ ДАННЫХ =====
    cout << endl;
    cout << "===== DATA TRANSFER =====" << endl;
    cout << "Data size: " << fixed << setprecision(2) << size / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Allocation time: " << setprecision(6) << malloc_time << " ms" << endl;
    cout << "D2H copy time: " << memcpy_time << " ms" << endl;

    cudaFree(d_B);
    free(h_A);
    free(h_B);

    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_malloc);
    cudaEventDestroy(stop_malloc);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_memcpy);
    cudaEventDestroy(stop_memcpy);

    return 0;
}
