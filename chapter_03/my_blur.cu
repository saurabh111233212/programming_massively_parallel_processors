#include "timer.h"
#include <stdio.h>

__device__ __host__ unsigned int calcSum(unsigned char* img, unsigned int width, unsigned int height, unsigned int radius, int outRow, int outCol) {
    unsigned int sum = 0;
    for (int inRow = outRow - radius; inRow < outRow + radius; inRow++) {
        for (int inCol = outCol - radius; inCol < outCol + radius; inCol++) {
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                sum += img[inRow * width + inCol];
            }
        }
    }

    return sum;
}

__global__ void blur_kernel(unsigned char* img, unsigned char* out, unsigned int width, unsigned int height) {
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockDim.x + threadIdx.x;
    unsigned int radius = 1;

    if (outRow < height && outCol < width) {
        unsigned int sum = calcSum(img, width, height, radius, outRow, outCol);
        out[outRow*width + outCol] = (unsigned char) (sum / ((2 * radius + 1) * (2 * radius + 1)));

    }
}

void blur_cpu(unsigned char* img, unsigned char* out, unsigned int width, unsigned int height) {
    unsigned int radius = 1;

    for (int outRow = 0; outRow < height; outRow++) {
        for (int outCol = 0; outCol < width; outCol++) {
            unsigned int sum = calcSum(img, width, height, radius, outRow, outCol);
            out[outRow*width + outCol] = (unsigned char) (sum / ((2 * radius + 1) * (2 * radius + 1)));
        }
    }
}


void blur_gpu(unsigned char* img, unsigned char* out, unsigned int width, unsigned int height) {
    unsigned char* img_d; unsigned char* out_d; 
    cudaMalloc((void**) &img_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &out_d, width*height*sizeof(unsigned char));

    cudaMemcpy(img_d, img, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

    GPUTimer gpuTimer;
    gpuStartTime(&gpuTimer);
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks(((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x), (height + numThreadsPerBlock.y - 1)/ numThreadsPerBlock.y);
    blur_kernel<<< numBlocks, numThreadsPerBlock >>>(img_d, out_d, width, height);
    gpuStopTime(&gpuTimer);
    gpuPrintElapsedTime(gpuTimer, "Kernel time");

    cudaMemcpy(out, out_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);


    cudaFree(img_d);
    cudaFree(out_d);
}


int main(int argc, char** argv) {
    unsigned int width = 1<<10;
    unsigned int height = 1<<10;

    unsigned char* img = (unsigned char*) malloc(width * height * sizeof(unsigned char));
    unsigned char* out = (unsigned char*) malloc(width * height * sizeof(unsigned char));

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int i = row * width + col;
            img[i] = rand() % 256;
        }
    }

    CPUTimer cpuTimer;
    GPUTimer gpuTimer;

    cpuStartTime(&cpuTimer);
    blur_cpu(img, out, width, height);
    cpuStopTime(&cpuTimer);
    cpuPrintElapsedTime(cpuTimer, "CPU time");

    gpuStartTime(&gpuTimer);
    blur_gpu(img, out, width, height);
    gpuStopTime(&gpuTimer);
    gpuPrintElapsedTime(gpuTimer, "GPU time");


    free(img);
}