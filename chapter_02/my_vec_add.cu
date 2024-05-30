#include "timer.h"
#include <stdio.h>

void vec_add_cpu(float* x, float* y, float* z, int N) {
    for (unsigned int i = 0; i < N; i++) {
        z[i] = x[i] + y[i];
    }
}

__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        z[i] = x[i] + y[i];
    }
}

void vec_add_gpu(float* x, float* y, float* z, int N) {
    // Alocate GPU memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**) &x_d, N*sizeof(float));
    cudaMalloc((void**) &y_d, N*sizeof(float));
    cudaMalloc((void**) &z_d, N*sizeof(float));

    //copy to GPU
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // call the kernel
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vecadd_kernel<<< numBlocks, numThreadsPerBlock >>>(x_d, y_d, z_d, N);

    // copy from GPU
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // deallocate the GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(int argc, char**argv) {
    GPUTimer gpuTimer; 
    CPUTimer cpuTimer;
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1<<25);

    float *x = (float*)malloc(N*sizeof(float));
    float *y = (float*)malloc(N*sizeof(float));
    float *z = (float*)malloc(N*sizeof(float));

    for (unsigned int i = 0; i < N; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    cpuStartTime(&cpuTimer);
    vec_add_cpu(x, y, z, N);
    cpuStopTime(&cpuTimer);
    cpuPrintElapsedTime(cpuTimer, "CPU time");

    gpuStartTime(&gpuTimer);
    vec_add_gpu(x, y, z, N);
    gpuStopTime(&gpuTimer);
    gpuPrintElapsedTime(gpuTimer, "GPU time");

    free(x);
    free(y);
    free(z);

    return 0;
}
