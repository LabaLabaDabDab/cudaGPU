#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <type_traits>

using namespace std;

#define N 1000000000
#define PI 3.14159265358979323846f

#define USE_DOUBLE

#ifdef USE_DOUBLE
    #define dtype double
#else
    #define dtype float
#endif


__global__ void kernel_sin(dtype* arr, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = sin((i % 360) * PI / (dtype)180.0);
    }
}

__global__ void kernel_sinf(dtype* arr, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = sinf((i % 360) * PI / (dtype)180.0);
    }
}

__global__ void kernel_fast_sinf(dtype* arr, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = __sinf((i % 360) * PI / (dtype)180.0);
    }
}


int main() {
    int device = 0;
    dim3 blockSize(1024);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    double err = 0.0;

    dtype* dev_arr = nullptr;
    dtype* arr = nullptr;
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

    cout << "Device name: " << prop.name << endl;
    cout << "Array type: " << (std::is_same<dtype, double>::value ? "double" : "float") << endl;
    cout << "Number of multiprocessors: " << prop.multiProcessorCount << endl;
    cout << "Global memory size: " << prop.totalGlobalMem << " bytes" << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Max grid size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
    cout << "Max block dimensions: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
    cout << endl;

    cuda_error = cudaMalloc((void**)&dev_arr, N * sizeof(dtype));
    if (cuda_error != cudaSuccess) {
        cout << "cudaMalloc failed: " << cudaGetErrorString(cuda_error) << endl;
        cout << "Probably not enough GPU memory for N = " << N
             << " elements of type " << (std::is_same<dtype, double>::value ? "double" : "float") << endl;
        return 1;
    }

    arr = (dtype*)malloc(N * sizeof(dtype));
    if (!arr) {
        cout << "Failed to allocate host memory" << endl;
        cudaFree(dev_arr);
        return 1;
    }

    auto compute_error = [&](const char* title) {
        err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            dtype expected = sin((i % 360) * PI / (dtype)180.0);
            err += fabs(expected - arr[i]);
        }
        err /= N;

        cout << title << " mean absolute error = " << scientific << setprecision(10) << err << endl;
    };

    // ===== 1) kernel_sin (sin) =====
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        kernel_sin<<<gridSize, blockSize>>>(dev_arr, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(arr, dev_arr, (size_t)N * sizeof(dtype), cudaMemcpyDeviceToHost);

        compute_error("sin   ");
        cout << "Execution time (kernel_sin): " << fixed << setprecision(3) << ms << " ms" << endl << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ===== 2) kernel_sinf (sinf) =====
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        kernel_sinf<<<gridSize, blockSize>>>(dev_arr, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(arr, dev_arr, (size_t)N * sizeof(dtype), cudaMemcpyDeviceToHost);

        compute_error("sinf  ");
        cout << "Execution time (kernel_sinf): " << fixed << setprecision(3) << ms << " ms" << endl << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ===== 3) kernel_fast_sinf (__sinf) =====
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        kernel_fast_sinf<<<gridSize, blockSize>>>(dev_arr, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(arr, dev_arr, (size_t)N * sizeof(dtype), cudaMemcpyDeviceToHost);

        compute_error("__sinf");
        cout << "Execution time (kernel_fast_sinf): " << fixed << setprecision(3) << ms << " ms" << endl << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(dev_arr);
    free(arr);
    return 0;
}
