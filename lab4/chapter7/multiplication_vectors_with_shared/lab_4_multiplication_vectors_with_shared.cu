#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// Размер блока
#define BLOCK_DIM 256
// Тип элементов векторов
#define BASE_TYPE float

// Пустое warm-up ядро для инициализации CUDA контекста
__global__ void warmupKernel() {}

// Ядро для скалярного произведения с использованием глобальной памяти
// Каждая нить умножает пару элементов и атомарно добавляет вклад к общему результату
__global__ void dotProductGlobal(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int numElem) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    BASE_TYPE partial_sum = 0.0f;

    if (globalIdx < numElem) {
        partial_sum = A[globalIdx] * B[globalIdx];
    }

    atomicAdd(C, partial_sum);
}

// Ядро для скалярного произведения с использованием разделяемой памяти
// Внутри блока выполняется редукция, затем одна атомарная операция на блок
__global__ void dotProductShared(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int numElem) {
    __shared__ BASE_TYPE partial_sums[BLOCK_DIM];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    BASE_TYPE thread_sum = 0.0f;
    if (globalIdx < numElem) {
        thread_sum = A[globalIdx] * B[globalIdx];
    }

    partial_sums[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(C, partial_sums[0]);
    }
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

// Инициализация вектора случайными значениями [0;1)
// Все выровненные "хвостовые" элементы заполняем нулями
void initVectorRandom(BASE_TYPE* A, int numElem, int alignedNumElem) {
    for (int i = 0; i < alignedNumElem; ++i) {
        if (i < numElem) {
            A[i] = static_cast<BASE_TYPE>(rand()) / static_cast<BASE_TYPE>(RAND_MAX);
        } else {
            A[i] = 0.0f;
        }
    }
}

// Скалярное произведение на CPU
BASE_TYPE dotProductCPU(const BASE_TYPE* A, const BASE_TYPE* B, int numElem) {
    BASE_TYPE sum = 0.0f;
    for (int i = 0; i < numElem; ++i) {
        sum += A[i] * B[i];
    }
    return sum;
}

// Проверка корректности результата
bool compareScalars(BASE_TYPE cpu, BASE_TYPE gpu, double eps = 1e-2) {
    double diff = fabs(static_cast<double>(cpu) - static_cast<double>(gpu));
    if (diff > eps) {
        cout << "Mismatch: CPU=" << cpu << ", GPU=" << gpu << ", diff=" << diff << ", eps=" << eps << endl;
        return false;
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
    int numElem = 1000000;
    int alignedNumElem = toMultiple(numElem, BLOCK_DIM);

    const size_t Vsize = static_cast<size_t>(alignedNumElem) * sizeof(BASE_TYPE);
    const size_t Rsize = sizeof(BASE_TYPE);

    const int iterations = 100;

    cout << "===== TASK PARAMETERS =====" << endl;
    cout << "Vector length (original): " << numElem << endl;
    cout << "Vector length (aligned): " << alignedNumElem << endl;
    cout << "Block size: " << BLOCK_DIM << endl;
    cout << "Iterations: " << iterations << endl;
    cout << fixed << setprecision(2);
    cout << "Data size A: " << Vsize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size B: " << Vsize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Result size: " << Rsize << " bytes" << endl;
    cout << "Total data (A+B+Result): " << (2.0 * Vsize + Rsize) / (1024.0 * 1024.0) << " MB" << endl << endl;

    // ===== CPU =====
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    cout << "===== CPU PHASE =====" << endl;

    // Память на CPU
    BASE_TYPE* h_A = static_cast<BASE_TYPE*>(malloc(Vsize));
    BASE_TYPE* h_B = static_cast<BASE_TYPE*>(malloc(Vsize));

    if (!h_A || !h_B) {
        cerr << "Host memory allocation failed" << endl;
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        cudaEventDestroy(start_cpu);
        cudaEventDestroy(stop_cpu);
        return 1;
    }

    srand(0);
    initVectorRandom(h_A, numElem, alignedNumElem);
    initVectorRandom(h_B, numElem, alignedNumElem);

    cudaEventRecord(start_cpu, 0);
    BASE_TYPE h_C_cpu = dotProductCPU(h_A, h_B, numElem);
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);

    float cpu_time = 0.0f;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    cout << setprecision(6);
    cout << "CPU dot product time: " << cpu_time << " ms" << endl;
    cout << "CPU result: " << h_C_cpu << endl << endl;

    cout << "===== GPU =====" << endl;

    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_malloc, stop_malloc;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel_global, stop_kernel_global;
    cudaEvent_t start_kernel_shared, stop_kernel_shared;
    cudaEvent_t start_d2h, stop_d2h;

    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_malloc);
    cudaEventCreate(&stop_malloc);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel_global);
    cudaEventCreate(&stop_kernel_global);
    cudaEventCreate(&start_kernel_shared);
    cudaEventCreate(&stop_kernel_shared);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    float total_gpu_time = 0.0f;
    float malloc_time = 0.0f;
    float h2d_time = 0.0f;
    float kernel_time_global = 0.0f;
    float kernel_time_shared = 0.0f;
    float d2h_time = 0.0f;

    BASE_TYPE* d_A = nullptr;
    BASE_TYPE* d_B = nullptr;
    BASE_TYPE* d_C_global = nullptr;
    BASE_TYPE* d_C_shared = nullptr;

    cudaEventRecord(start_total, 0);

    warmupKernel<<<1, 1>>>();
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        cerr << "Warm-up kernel launch failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A);
        free(h_B);
        cudaEventDestroy(start_cpu);
        cudaEventDestroy(stop_cpu);
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_malloc);
        cudaEventDestroy(stop_malloc);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel_global);
        cudaEventDestroy(stop_kernel_global);
        cudaEventDestroy(start_kernel_shared);
        cudaEventDestroy(stop_kernel_shared);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
        return 1;
    }

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize after warm-up failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A);
        free(h_B);
        cudaEventDestroy(start_cpu);
        cudaEventDestroy(stop_cpu);
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_malloc);
        cudaEventDestroy(stop_malloc);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel_global);
        cudaEventDestroy(stop_kernel_global);
        cudaEventDestroy(start_kernel_shared);
        cudaEventDestroy(stop_kernel_shared);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
        return 1;
    }

    // Выделение памяти на GPU
    cudaEventRecord(start_malloc, 0);

    cuda_error = cudaMalloc((void**)&d_A, Vsize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_A) failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A);
        free(h_B);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_B, Vsize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_B) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        free(h_A);
        free(h_B);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_C_global, Rsize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_C_global) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        free(h_A);
        free(h_B);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_C_shared, Rsize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_C_shared) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_global);
        free(h_A);
        free(h_B);
        return 1;
    }

    cudaEventRecord(stop_malloc, 0);
    cudaEventSynchronize(stop_malloc);
    cudaEventElapsedTime(&malloc_time, start_malloc, stop_malloc);
    cout << "GPU allocation time: " << malloc_time << " ms" << endl;

    // --- H2D ---
    cudaEventRecord(start_h2d, 0);

    cuda_error = cudaMemcpy(d_A, h_A, Vsize, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy h_A -> d_A failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C_global); 
        cudaFree(d_C_shared);
        free(h_A); 
        free(h_B);
        return 1;
    }

    cuda_error = cudaMemcpy(d_B, h_B, Vsize, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy h_B -> d_B failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C_global); 
        cudaFree(d_C_shared);
        free(h_A); 
        free(h_B);
        return 1;
    }

    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d);
    cout << "Host->device copy time (A+B): " << h2d_time << " ms" << endl;

    // ===== Настройка grid / block =====
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (alignedNumElem + BLOCK_DIM - 1) / BLOCK_DIM;

    cout << "Grid: (" << blocksPerGrid << ")" << endl;
    cout << "Block: (" << threadsPerBlock << ")" << endl << endl;

    // --- kernel: global ---
    cudaEventRecord(start_kernel_global, 0);

    for (int iter = 0; iter < iterations; ++iter) {
        cuda_error = cudaMemset(d_C_global, 0, Rsize);
        if (cuda_error != cudaSuccess) {
            cerr << "cudaMemset(d_C_global) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(d_A); 
            cudaFree(d_B); 
            cudaFree(d_C_global); 
            cudaFree(d_C_shared);
            free(h_A); 
            free(h_B);
            return 1;
        }

        dotProductGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_global, numElem);

        cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            cerr << "dotProductGlobal launch failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(d_A); 
            cudaFree(d_B); 
            cudaFree(d_C_global); 
            cudaFree(d_C_shared);
            free(h_A); 
            free(h_B);
            return 1;
        }
    }

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize after dotProductGlobal failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C_global); 
        cudaFree(d_C_shared);
        free(h_A); 
        free(h_B);
        return 1;
    }

    cudaEventRecord(stop_kernel_global, 0);
    cudaEventSynchronize(stop_kernel_global);
    cudaEventElapsedTime(&kernel_time_global, start_kernel_global, stop_kernel_global);

    float avg_kernel_time_global = kernel_time_global / iterations;

    cout << "Global kernel total time (" << iterations << " iters): " << kernel_time_global << " ms" << endl;
    cout << "Global kernel avg time: " << avg_kernel_time_global << " ms" << endl << endl;

    // --- kernel: shared ---
    cudaEventRecord(start_kernel_shared, 0);

    for (int iter = 0; iter < iterations; ++iter) {
        cuda_error = cudaMemset(d_C_shared, 0, Rsize);
        if (cuda_error != cudaSuccess) {
            cerr << "cudaMemset(d_C_shared) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(d_A); 
            cudaFree(d_B); 
            cudaFree(d_C_global); 
            cudaFree(d_C_shared);
            free(h_A); 
            free(h_B);
            return 1;
        }

        dotProductShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_shared, numElem);

        cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            cerr << "dotProductShared launch failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(d_A); 
            cudaFree(d_B); 
            cudaFree(d_C_global); 
            cudaFree(d_C_shared);
            free(h_A); 
            free(h_B);
            return 1;
        }
    }

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize after dotProductShared failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C_global); 
        cudaFree(d_C_shared);
        free(h_A); 
        free(h_B);
        return 1;
    }

    cudaEventRecord(stop_kernel_shared, 0);
    cudaEventSynchronize(stop_kernel_shared);
    cudaEventElapsedTime(&kernel_time_shared, start_kernel_shared, stop_kernel_shared);

    float avg_kernel_time_shared = kernel_time_shared / iterations;

    cout << "Shared kernel total time (" << iterations << " iters): " << kernel_time_shared << " ms" << endl;
    cout << "Shared kernel avg time: " << avg_kernel_time_shared << " ms" << endl << endl;

    // --- D2H ---
    cudaEventRecord(start_d2h, 0);

    BASE_TYPE h_C_global = 0.0f;
    BASE_TYPE h_C_shared = 0.0f;

    cuda_error = cudaMemcpy(&h_C_global, d_C_global, Rsize, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy d_C_global -> h_C_global failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C_global); 
        cudaFree(d_C_shared);
        free(h_A); 
        free(h_B);
        return 1;
    }

    cuda_error = cudaMemcpy(&h_C_shared, d_C_shared, Rsize, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy d_C -> h_C failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C_global); 
        cudaFree(d_C_shared);
        free(h_A); 
        free(h_B);
        return 1;
    }

    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h);
    cout << "Device->host copy time (results): " << d2h_time << " ms" << endl;

    // Общее время работы GPU
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_gpu_time, start_total, stop_total);

    cout << "Total GPU time: " << total_gpu_time << " ms" << endl << endl;

    // ===== ПРОВЕРКА РЕЗУЛЬТАТА =====
    cout << "===== RESULT CHECK =====" << endl;

    bool ok_global = compareScalars(h_C_cpu, h_C_global);
    bool ok_shared = compareScalars(h_C_cpu, h_C_shared);

    cout << "CPU result: " << h_C_cpu << endl;
    cout << "Global result: " << h_C_global << endl;
    cout << "Shared result: " << h_C_shared << endl;

    if (ok_global) {
        cout << "Global kernel result matches CPU (within tolerance)." << endl;
    } else {
        cout << "Global kernel result differs from CPU." << endl;
    }

    if (ok_shared) {
        cout << "Shared kernel result matches CPU (within tolerance)." << endl;
    } else {
        cout << "Shared kernel result differs from CPU." << endl;
    }

    // ===== КРАТКАЯ СВОДКА ПО ПРОИЗВОДИТЕЛЬНОСТИ =====
    cout << endl;
    cout << "===== PERFORMANCE SUMMARY =====" << endl;
    cout << "CPU dot product: " << cpu_time << " ms" << endl;
    cout << "GPU total time: " << total_gpu_time << " ms" << endl;
    cout << "GPU global kernel avg: " << avg_kernel_time_global << " ms" << endl;
    cout << "GPU shared kernel avg: " << avg_kernel_time_shared << " ms" << endl;

    if (cpu_time > 0.0f && avg_kernel_time_global > 0.0f && avg_kernel_time_shared > 0.0f) {
        double speedup_global = cpu_time / avg_kernel_time_global;
        double speedup_shared = cpu_time / avg_kernel_time_shared;

        cout << fixed << setprecision(3);
        cout << "Speedup (CPU / GPU global avg): " << speedup_global << "x" << endl;
        cout << "Speedup (CPU / GPU shared avg): " << speedup_shared << "x" << endl;

        if (avg_kernel_time_global > 0.0f) {
            double improvement = (avg_kernel_time_global - avg_kernel_time_shared) / avg_kernel_time_global * 100.0;
            cout << "Shared vs Global improvement: " << improvement << "%" << endl;
        }
    }

    // ===== СВОДКА ПО ПЕРЕДАЧЕ ДАННЫХ =====
    cout << endl;
    cout << "===== DATA TRANSFER =====" << endl;
    cout << fixed << setprecision(2);
    cout << "Data size A: " << Vsize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size B: " << Vsize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Result size: " << Rsize << " bytes" << endl;
    cout << setprecision(6);
    cout << "Allocation time: " << malloc_time << " ms" << endl;
    cout << "Host->device time: " << h2d_time << " ms" << endl;
    cout << "Device->host time: " << d2h_time << " ms" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_global);
    cudaFree(d_C_shared);

    free(h_A);
    free(h_B);

    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_malloc);
    cudaEventDestroy(stop_malloc);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel_global);
    cudaEventDestroy(stop_kernel_global);
    cudaEventDestroy(start_kernel_shared);
    cudaEventDestroy(stop_kernel_shared);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    return 0;
}
