#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// Размер блока (tile) для матричного умножения
#define BLOCK_DIM 16
// Тип элементов матриц
#define BASE_TYPE float

// Пустое warm-up ядро для инициализации CUDA контекста
__global__ void warmupKernel() {}

// Базовое ядро для умножения матриц без shared memory
// C = A * B
__global__ void matrixMultBasic(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Arows, int Acols, int Bcols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= Arows || col >= Bcols) {
        return;
    }

    BASE_TYPE sum = 0.0f;
    for (int k = 0; k < Acols; ++k) {
        sum += A[row * Acols + k] * B[k * Bcols + col];
    }

    C[row * Bcols + col] = sum;
}

// Оптимизированное ядро для умножения матриц с использованием shared memory
__global__ void matrixMultTiled(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Arows, int Acols, int Bcols) {
    __shared__ BASE_TYPE As[BLOCK_DIM][BLOCK_DIM];
    __shared__ BASE_TYPE Bs[BLOCK_DIM][BLOCK_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BLOCK_DIM + ty;
    int col = blockIdx.x * BLOCK_DIM + tx;

    BASE_TYPE sum = 0.0f;

    int numSteps = (Acols + BLOCK_DIM - 1) / BLOCK_DIM;

    for (int m = 0; m < numSteps; ++m) {
        int aCol = m * BLOCK_DIM + tx;
        if (row < Arows && aCol < Acols) {
            As[ty][tx] = A[row * Acols + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        int bRow = m * BLOCK_DIM + ty;
        if (bRow < Acols && col < Bcols) {
            Bs[ty][tx] = B[bRow * Bcols + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_DIM; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < Arows && col < Bcols) {
        C[row * Bcols + col] = sum;
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

// Инициализация матрицы случайными значениями [0;1)
void initMatrixRandom(BASE_TYPE* A, int rows, int cols) {
    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        A[i] = static_cast<BASE_TYPE>(rand()) / static_cast<BASE_TYPE>(RAND_MAX);
    }
}

// Умножение матриц на CPU
void matrixMultCPU(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Arows, int Acols, int Bcols) {
    for (int i = 0; i < Arows; ++i) {
        for (int j = 0; j < Bcols; ++j) {
            BASE_TYPE sum = 0.0f;
            for (int k = 0; k < Acols; ++k) {
                sum += A[i * Acols + k] * B[k * Bcols + j];
            }
            C[i * Bcols + j] = sum;
        }
    }
}

// Проверка корректности результата
bool compareMatrices(const BASE_TYPE* C_cpu, const BASE_TYPE* C_gpu, int rows, int cols, double eps = 1e-3){
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            double diff = fabs(static_cast<double>(C_cpu[idx]) - static_cast<double>(C_gpu[idx]));
            if (diff > eps) {
                cout << "Mismatch at (" << i << ", " << j << "): "
                     << "CPU=" << C_cpu[idx] << ", GPU=" << C_gpu[idx]
                     << ", diff=" << diff << ", eps=" << eps << endl;
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
    cout << "Max grid size: "
         << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
    cout << "Max block dimensions: "
         << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
    cout << endl;

    // ===== ПАРАМЕТРЫ ЗАДАЧИ =====
    int Arows = 500;
    int Acols = 400;
    int Brows = Acols;
    int Bcols = 300;

    int Arows_aligned = toMultiple(Arows, BLOCK_DIM);
    int Acols_aligned = toMultiple(Acols, BLOCK_DIM);
    int Brows_aligned = Acols_aligned;
    int Bcols_aligned = toMultiple(Bcols, BLOCK_DIM);

    const size_t Asize = static_cast<size_t>(Arows_aligned) * Acols_aligned * sizeof(BASE_TYPE);
    const size_t Bsize = static_cast<size_t>(Brows_aligned) * Bcols_aligned * sizeof(BASE_TYPE);
    const size_t Csize = static_cast<size_t>(Arows_aligned) * Bcols_aligned * sizeof(BASE_TYPE);

    const int iterations = 100;

    cout << "===== TASK PARAMETERS =====" << endl;
    cout << "A (original): " << Arows << " x " << Acols << endl;
    cout << "B (original): " << Brows << " x " << Bcols << endl;
    cout << "A (aligned):  " << Arows_aligned << " x " << Acols_aligned << endl;
    cout << "B (aligned):  " << Brows_aligned << " x " << Bcols_aligned << endl;
    cout << "C (aligned):  " << Arows_aligned << " x " << Bcols_aligned << endl;
    cout << "Block size:   " << BLOCK_DIM << " x " << BLOCK_DIM << endl;
    cout << "Iterations:   " << iterations << endl;

    cout << fixed << setprecision(2);
    cout << "Data size A: " << Asize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size B: " << Bsize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size C: " << Csize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Total data (A+B+C): "
         << (Asize + Bsize + Csize) / (1024.0 * 1024.0) << " MB" << endl << endl;

    // ===== CPU PHASE =====
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    cout << "===== CPU PHASE =====" << endl;

    BASE_TYPE* h_A = static_cast<BASE_TYPE*>(malloc(Asize));
    BASE_TYPE* h_B = static_cast<BASE_TYPE*>(malloc(Bsize));
    BASE_TYPE* h_C_cpu = static_cast<BASE_TYPE*>(malloc(Csize));
    BASE_TYPE* h_C_gpu_basic = static_cast<BASE_TYPE*>(malloc(Csize));
    BASE_TYPE* h_C_gpu_tiled = static_cast<BASE_TYPE*>(malloc(Csize));

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu_basic || !h_C_gpu_tiled) {
        cerr << "Host memory allocation failed" << endl;
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if (h_C_cpu) free(h_C_cpu);
        if (h_C_gpu_basic) free(h_C_gpu_basic);
        if (h_C_gpu_tiled) free(h_C_gpu_tiled);
        cudaEventDestroy(start_cpu);
        cudaEventDestroy(stop_cpu);
        return 1;
    }

    srand(0);
    initMatrixRandom(h_A, Arows_aligned, Acols_aligned);
    initMatrixRandom(h_B, Brows_aligned, Bcols_aligned);

    cudaEventRecord(start_cpu, 0);
    matrixMultCPU(h_A, h_B, h_C_cpu, Arows_aligned, Acols_aligned, Bcols_aligned);
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);

    float cpu_time = 0.0f;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    cout << setprecision(6);
    cout << "CPU matrix multiplication time: " << cpu_time << " ms" << endl << endl;

    // ===== GPU =====
    cout << "===== GPU =====" << endl;

    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_malloc, stop_malloc;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel_basic, stop_kernel_basic;
    cudaEvent_t start_kernel_tiled, stop_kernel_tiled;
    cudaEvent_t start_d2h, stop_d2h;

    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_malloc);
    cudaEventCreate(&stop_malloc);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel_basic);
    cudaEventCreate(&stop_kernel_basic);
    cudaEventCreate(&start_kernel_tiled);
    cudaEventCreate(&stop_kernel_tiled);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    float total_gpu_time = 0.0f;
    float malloc_time = 0.0f;
    float h2d_time = 0.0f;
    float kernel_time_basic = 0.0f;
    float kernel_time_tiled = 0.0f;
    float d2h_time = 0.0f;

    BASE_TYPE* d_A = nullptr;
    BASE_TYPE* d_B = nullptr;
    BASE_TYPE* d_C_basic = nullptr;
    BASE_TYPE* d_C_tiled = nullptr;

    cudaEventRecord(start_total, 0);

    // Warm-up
    warmupKernel<<<1, 1>>>();
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        cerr << "Warm-up kernel launch failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize after warm-up failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    // --- allocation ---
    cudaEventRecord(start_malloc, 0);

    cuda_error = cudaMalloc((void**)&d_A, Asize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_A) failed: " << cudaGetErrorString(cuda_error) << endl;
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_B, Bsize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_B) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_C_basic, Csize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_C_basic) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); cudaFree(d_B);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&d_C_tiled, Csize);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMalloc(d_C_tiled) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cudaEventRecord(stop_malloc, 0);
    cudaEventSynchronize(stop_malloc);
    cudaEventElapsedTime(&malloc_time, start_malloc, stop_malloc);
    cout << "GPU allocation time: " << malloc_time << " ms" << endl;

    // --- H2D ---
    cudaEventRecord(start_h2d, 0);

    cuda_error = cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy h_A -> d_A failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic); cudaFree(d_C_tiled);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cuda_error = cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy h_B -> d_B failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic); cudaFree(d_C_tiled);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d);
    cout << "Host->device copy time (A+B): " << h2d_time << " ms" << endl;

    // ===== Настройка grid / block =====
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 blocksPerGrid(
        (Bcols_aligned + BLOCK_DIM - 1) / BLOCK_DIM,
        (Arows_aligned + BLOCK_DIM - 1) / BLOCK_DIM
    );

    cout << "Grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << endl;
    cout << "Block: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << endl << endl;

    // --- kernel: basic ---
    cudaEventRecord(start_kernel_basic, 0);

    for (int iter = 0; iter < iterations; ++iter) {
        matrixMultBasic<<<blocksPerGrid, threadsPerBlock>>>(
            d_A, d_B, d_C_basic,
            Arows_aligned, Acols_aligned, Bcols_aligned
        );

        cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            cerr << "matrixMultBasic launch failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic); cudaFree(d_C_tiled);
            free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
            return 1;
        }
    }

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize after matrixMultBasic failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic); cudaFree(d_C_tiled);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cudaEventRecord(stop_kernel_basic, 0);
    cudaEventSynchronize(stop_kernel_basic);
    cudaEventElapsedTime(&kernel_time_basic, start_kernel_basic, stop_kernel_basic);

    float avg_kernel_time_basic = kernel_time_basic / iterations;

    cout << "Basic kernel total time (" << iterations << " iters): " << kernel_time_basic << " ms" << endl;
    cout << "Basic kernel avg time: " << avg_kernel_time_basic << " ms" << endl << endl;

    // --- kernel: tiled ---
    cudaEventRecord(start_kernel_tiled, 0);

    for (int iter = 0; iter < iterations; ++iter) {
        matrixMultTiled<<<blocksPerGrid, threadsPerBlock>>>(
            d_A, d_B, d_C_tiled,
            Arows_aligned, Acols_aligned, Bcols_aligned
        );

        cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            cerr << "matrixMultTiled launch failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic); cudaFree(d_C_tiled);
            free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
            return 1;
        }
    }

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        cerr << "cudaDeviceSynchronize after matrixMultTiled failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic); cudaFree(d_C_tiled);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cudaEventRecord(stop_kernel_tiled, 0);
    cudaEventSynchronize(stop_kernel_tiled);
    cudaEventElapsedTime(&kernel_time_tiled, start_kernel_tiled, stop_kernel_tiled);

    float avg_kernel_time_tiled = kernel_time_tiled / iterations;

    cout << "Tiled kernel total time (" << iterations << " iters): " << kernel_time_tiled << " ms" << endl;
    cout << "Tiled kernel avg time: " << avg_kernel_time_tiled << " ms" << endl << endl;

    // --- D2H ---
    cudaEventRecord(start_d2h, 0);

    cuda_error = cudaMemcpy(h_C_gpu_basic, d_C_basic, Csize, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy d_C_basic -> h_C_gpu_basic failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic); cudaFree(d_C_tiled);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cuda_error = cudaMemcpy(h_C_gpu_tiled, d_C_tiled, Csize, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        cerr << "cudaMemcpy d_C_tiled -> h_C_gpu_tiled failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_basic); cudaFree(d_C_tiled);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_basic); free(h_C_gpu_tiled);
        return 1;
    }

    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h);
    cout << "Device->host copy time (C basic + C tiled): " << d2h_time << " ms" << endl;

    // --- total GPU time ---
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_gpu_time, start_total, stop_total);

    cout << "Total GPU time: " << total_gpu_time << " ms" << endl << endl;

    // ===== RESULT CHECK =====
    cout << "===== RESULT CHECK =====" << endl;

    bool ok_basic = compareMatrices(h_C_cpu, h_C_gpu_basic, Arows_aligned, Bcols_aligned);
    bool ok_tiled = compareMatrices(h_C_cpu, h_C_gpu_tiled, Arows_aligned, Bcols_aligned);

    if (ok_basic) {
        cout << "Basic kernel result matches CPU (within tolerance)." << endl;
    } else {
        cout << "Basic kernel result differs from CPU." << endl;
    }

    if (ok_tiled) {
        cout << "Tiled kernel result matches CPU (within tolerance)." << endl;
    } else {
        cout << "Tiled kernel result differs from CPU." << endl;
    }

    // ===== PERFORMANCE SUMMARY =====
    cout << endl;
    cout << "===== PERFORMANCE SUMMARY =====" << endl;
    cout << "CPU matrix multiplication: " << cpu_time << " ms" << endl;
    cout << "GPU total time: " << total_gpu_time << " ms" << endl;
    cout << "GPU basic kernel avg: " << avg_kernel_time_basic << " ms" << endl;
    cout << "GPU tiled kernel avg: " << avg_kernel_time_tiled << " ms" << endl;

    if (cpu_time > 0.0f && avg_kernel_time_basic > 0.0f && avg_kernel_time_tiled > 0.0f) {
        double speedup_basic = cpu_time / avg_kernel_time_basic;
        double speedup_tiled = cpu_time / avg_kernel_time_tiled;
        double shared_gain = avg_kernel_time_basic / avg_kernel_time_tiled;

        cout << fixed << setprecision(3);
        cout << "Speedup (CPU / GPU basic avg): " << speedup_basic << "x" << endl;
        cout << "Speedup (CPU / GPU tiled avg): " << speedup_tiled << "x" << endl;
        cout << "Shared memory gain (basic / tiled): " << shared_gain << "x" << endl;
    }

    // ===== DATA TRANSFER =====
    cout << endl;
    cout << "===== DATA TRANSFER =====" << endl;
    cout << fixed << setprecision(2);
    cout << "Data size A: " << Asize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size B: " << Bsize / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Data size C: " << Csize / (1024.0 * 1024.0) << " MB" << endl;
    cout << setprecision(6);
    cout << "Allocation time: " << malloc_time << " ms" << endl;
    cout << "Host->device time: " << h2d_time << " ms" << endl;
    cout << "Device->host time: " << d2h_time << " ms" << endl;
 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_basic);
    cudaFree(d_C_tiled);

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu_basic);
    free(h_C_gpu_tiled);

    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_malloc);
    cudaEventDestroy(stop_malloc);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel_basic);
    cudaEventDestroy(stop_kernel_basic);
    cudaEventDestroy(start_kernel_tiled);
    cudaEventDestroy(stop_kernel_tiled);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    return 0;
}
