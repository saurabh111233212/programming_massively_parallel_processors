#include "timer.h"
#include <stdio.h>

__global__ void grayscale_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int i = row * width + col;
    if (row < height && col < width) {
        gray[i] = 0.3 * red[i] + 0.6 * green[i] + 0.1 * blue[i];
    }
}


void rgb_to_gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    unsigned char* red_d; unsigned char* blue_d; unsigned char* green_d; unsigned char* gray_d;
    // allocate
    cudaMalloc((void**) &red_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &green_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &blue_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &gray_d, width*height*sizeof(unsigned char));
    //copy CPU ->GPU
    cudaMemcpy(red_d, red, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // call the kernel
    GPUTimer timer;
    gpuStartTime(&timer);
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    grayscale_kernel<<< numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);
    gpuStopTime(&timer);
    gpuPrintElapsedTime(timer, "Kernel time");

    // copy GPU -> CPU
    cudaMemcpy(gray, gray_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //de-allocate
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
}


void rgb_to_gray_cpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            unsigned int i = row * width + col;
            gray[i] = 0.3 * red[i] + 0.6 * green[i] + 0.1 * blue[i];
        }
    }
}

int main(int argc, char** argv) {
    unsigned int width = 1<<10;
    unsigned int height = 1<<10;
    unsigned char* red = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char* blue = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char* green = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char* gray = (unsigned char*)malloc(width * height * sizeof(unsigned char));


    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int i = row * width + col;
            red[i] = rand() % 256;
            blue[i] = rand() % 256;
            green[i] = rand() % 256;
        }
    }

    CPUTimer cpuTimer;
    GPUTimer gpuTimer;

    cpuStartTime(&cpuTimer);
    rgb_to_gray_cpu(red, green, blue, gray, width, height);
    cpuStopTime(&cpuTimer);
    cpuPrintElapsedTime(cpuTimer, "CPU time");

    gpuStartTime(&gpuTimer);
    rgb_to_gray_gpu(red, green, blue, gray, width, height);
    gpuStopTime(&gpuTimer);
    gpuPrintElapsedTime(gpuTimer, "GPU time");


    free(red);
    free(blue);
    free(green);
    free(gray);

}