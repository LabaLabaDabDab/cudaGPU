#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <png.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>

using namespace std;

#define BLOCK_DIM 16

struct Image {
    int width;
    int height;
    png_bytep data;
};

enum FilterType {
    FILTER_BLUR,
    FILTER_EDGE,
    FILTER_DENOISE
};

__constant__ float d_kernel[9];


Image readPNG(const char* filename) {
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, filename)) {
        fprintf(stderr, "Error reading PNG: %s\n", image.message);
        exit(1);
    }

    image.format = PNG_FORMAT_RGBA;
    png_bytep buffer = (png_bytep)malloc(PNG_IMAGE_SIZE(image));
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed for image data\n");
        png_image_free(&image);
        exit(1);
    }

    if (!png_image_finish_read(&image, nullptr, buffer, 0, nullptr)) {
        fprintf(stderr, "Error reading PNG data: %s\n", image.message);
        free(buffer);
        png_image_free(&image);
        exit(1);
    }

    Image result;
    result.width = image.width;
    result.height = image.height;
    result.data = buffer;

    png_image_free(&image);
    return result;
}


void writePNG(const char* filename, const Image& img) {
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;
    image.width = img.width;
    image.height = img.height;
    image.format = PNG_FORMAT_RGBA;

    printf("Writing PNG file: %s\n", filename);

    if (!png_image_write_to_file(&image, filename, 0, img.data, 0, nullptr)) {
        fprintf(stderr, "Error writing PNG: %s\n", image.message);
        png_image_free(&image);
        exit(1);
    }

    png_image_free(&image);
    printf("PNG written successfully\n");
}


void freeImage(Image& img) {
    if (img.data) {
        free(img.data);
        img.data = nullptr;
    }
    img.width = img.height = 0;
}


__global__ void kernel_filter(const unsigned char* input, unsigned char* output, int width, int height, int channels){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;

    for (int ky = -1; ky <= 1; ++ky) {
        int sy = (int)y + ky;
        if (sy < 0) sy = 0;
        if (sy >= height) sy = height - 1;

        for (int kx = -1; kx <= 1; ++kx) {
            int sx = (int)x + kx;
            if (sx < 0) sx = 0;
            if (sx >= width) sx = width - 1;

            int sidx = (sy * width + sx) * channels;
            float k = d_kernel[(ky + 1) * 3 + (kx + 1)];

            accumR += k * input[sidx + 0];
            accumG += k * input[sidx + 1];
            accumB += k * input[sidx + 2];
        }
    }

    int r = (int)roundf(accumR);
    int g = (int)roundf(accumG);
    int b = (int)roundf(accumB);

    if (r < 0) r = 0; if (r > 255) r = 255;
    if (g < 0) g = 0; if (g > 255) g = 255;
    if (b < 0) b = 0; if (b > 255) b = 255;

    output[idx + 0] = (unsigned char)r;
    output[idx + 1] = (unsigned char)g;
    output[idx + 2] = (unsigned char)b;

    if (channels > 3) {
        output[idx + 3] = input[idx + 3];
    }
}


FilterType parse_filter(const string& name)
{
    if (name == "blur") return FILTER_BLUR;
    if (name == "edge") return FILTER_EDGE;
    if (name == "denoise") return FILTER_DENOISE;

    cerr << "Unknown filter: " << name << endl;
    cerr << "Use one of: blur | edge | denoise" << endl;
    exit(1);
}


void get_kernel(FilterType type, float kernel[9])
{
    if (type == FILTER_BLUR) {
        float v = 1.0f / 9.0f;
        for (int i = 0; i < 9; ++i) kernel[i] = v;
    } else if (type == FILTER_EDGE) {
        float tmp[9] = {
             0.0f, -1.0f,  0.0f,
            -1.0f,  4.0f, -1.0f,
             0.0f, -1.0f,  0.0f
        };
        for (int i = 0; i < 9; ++i) kernel[i] = tmp[i];
    } else if (type == FILTER_DENOISE) {
        float tmp[9] = {
            1.f, 2.f, 1.f,
            2.f, 4.f, 2.f,
            1.f, 2.f, 1.f
        };
        for (int i = 0; i < 9; ++i) kernel[i] = tmp[i] / 16.0f;
    }
}


int main(int argc, char** argv)
{
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " input.png output.png [blur|edge|denoise]" << endl;
        return 1;
    }

    const char* input_name = argv[1];
    const char* output_name = argv[2];
    string filter_name = argv[3];

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

    cout << "Device name: " << prop.name << endl;
    cout << "Number of multiprocessors: " << prop.multiProcessorCount << endl;
    cout << "Global memory size: " << prop.totalGlobalMem << " bytes" << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Max grid size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
    cout << "Max block dimensions: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
    cout << endl;

    Image input_img = readPNG(input_name);
    int width = input_img.width;
    int height = input_img.height;
    int channels = 4;

    cout << "Loaded image: " << input_name << " (" << width << " x " << height << "), channels = " << channels << endl;

    FilterType filter_type = parse_filter(filter_name);
    float h_kernel[9];
    get_kernel(filter_type, h_kernel);

    cuda_error = cudaMemcpyToSymbol(d_kernel, h_kernel, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        cout << "cudaMemcpyToSymbol failed: " << cudaGetErrorString(cuda_error) << endl;
        freeImage(input_img);
        return 1;
    }

    size_t num_pixels = static_cast<size_t>(width) * height;
    size_t num_bytes = num_pixels * channels * sizeof(unsigned char);

    unsigned char* dev_in  = nullptr;
    unsigned char* dev_out = nullptr;

    cuda_error = cudaMalloc((void**)&dev_in, num_bytes);
    if (cuda_error != cudaSuccess) {
        cout << "cudaMalloc(dev_in) failed: " << cudaGetErrorString(cuda_error) << endl;
        freeImage(input_img);
        return 1;
    }

    cuda_error = cudaMalloc((void**)&dev_out, num_bytes);
    if (cuda_error != cudaSuccess) {
        cout << "cudaMalloc(dev_out) failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(dev_in);
        freeImage(input_img);
        return 1;
    }

    cudaEvent_t evH2DStart, evH2DStop;
    cudaEvent_t evKernelStart, evKernelStop;
    cudaEvent_t evD2HStart, evD2HStop;
    cudaEvent_t evTotalStart, evTotalStop;

    cudaEventCreate(&evH2DStart);
    cudaEventCreate(&evH2DStop);
    cudaEventCreate(&evKernelStart);
    cudaEventCreate(&evKernelStop);
    cudaEventCreate(&evD2HStart);
    cudaEventCreate(&evD2HStop);
    cudaEventCreate(&evTotalStart);
    cudaEventCreate(&evTotalStop);

    // ===== Host -> Device copy =====
    cudaEventRecord(evTotalStart, 0);
    cudaEventRecord(evH2DStart, 0);

    cuda_error = cudaMemcpy(dev_in, input_img.data, num_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(evH2DStop, 0);
    cudaEventSynchronize(evH2DStop);

    if (cuda_error != cudaSuccess) {
        cout << "cudaMemcpy host->device failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(dev_in);
        cudaFree(dev_out);
        freeImage(input_img);
        return 1;
    }

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize(
        (width  + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // ===== Kernel execution =====
    cudaEventRecord(evKernelStart, 0);
    kernel_filter<<<gridSize, blockSize>>>(dev_in, dev_out, width, height, channels);
    cudaEventRecord(evKernelStop, 0);
    cudaEventSynchronize(evKernelStop);

    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        cout << "Kernel launch failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(dev_in);
        cudaFree(dev_out);
        freeImage(input_img);
        return 1;
    }

    unsigned char* host_output = (unsigned char*)malloc(num_bytes);
    if (!host_output) {
        cerr << "Failed to allocate host_output" << endl;
        cudaFree(dev_in);
        cudaFree(dev_out);
        freeImage(input_img);
        return 1;
    }

    // ===== Device -> Host copy =====
    cudaEventRecord(evD2HStart, 0);

    cuda_error = cudaMemcpy(host_output, dev_out, num_bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(evD2HStop, 0);
    cudaEventRecord(evTotalStop, 0);

    cudaEventSynchronize(evD2HStop);
    cudaEventSynchronize(evTotalStop);

    if (cuda_error != cudaSuccess) {
        cout << "cudaMemcpy device->host failed: " << cudaGetErrorString(cuda_error) << endl;
        cudaFree(dev_in);
        cudaFree(dev_out);
        freeImage(input_img);
        free(host_output);
        return 1;
    }

    // ===== All Time =====
    float timeH2D    = 0.0f;
    float timeKernel = 0.0f;
    float timeD2H    = 0.0f;
    float timeTotal  = 0.0f;

    cudaEventElapsedTime(&timeH2D, evH2DStart, evH2DStop);
    cudaEventElapsedTime(&timeKernel, evKernelStart, evKernelStop);
    cudaEventElapsedTime(&timeD2H, evD2HStart, evD2HStop);
    cudaEventElapsedTime(&timeTotal, evTotalStart, evTotalStop);

    cout << fixed << setprecision(3);
    cout << "Host -> Device copy time: " << timeH2D << " ms" << endl;
    cout << "Kernel execution time: " << timeKernel << " ms" << endl;
    cout << "Device -> Host copy time: " << timeD2H << " ms" << endl;
    cout << "Total GPU time: " << timeTotal << " ms" << endl;

    cudaEventDestroy(evH2DStart);
    cudaEventDestroy(evH2DStop);
    cudaEventDestroy(evKernelStart);
    cudaEventDestroy(evKernelStop);
    cudaEventDestroy(evD2HStart);
    cudaEventDestroy(evD2HStop);
    cudaEventDestroy(evTotalStart);
    cudaEventDestroy(evTotalStop);

    Image output_img;
    output_img.width  = width;
    output_img.height = height;
    output_img.data   = host_output;

    writePNG(output_name, output_img);

    cudaFree(dev_in);
    cudaFree(dev_out);
    freeImage(input_img);
    freeImage(output_img);

    return 0;
}
