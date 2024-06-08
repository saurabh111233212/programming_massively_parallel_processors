#include "timer.h"
#include <stdio.h>

__global__ void matmul_kernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        int i = row * n + col;
        float res = 0;
        for (int j = 0; j < n; j++) {
            res += A[row * n + j] * B[j * n + col];
        }

        C[i] = res;
    }
}

void matmul_gpu(float* A, float* B, float* C, int n) {
    // allocate
    float* A_d; float* B_d; float* C_d;
    size_t size = n * n * (sizeof(float));
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // move host -> GPU
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    
    // call the kernel
    GPUTimer kernel_timer;
    gpuStartTime(&kernel_timer);
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((n + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (n + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    matmul_kernel <<< numBlocks, numThreadsPerBlock>>> (A_d, B_d, C_d, n);
    gpuStopTime(&kernel_timer);
    gpuPrintElapsedTime(kernel_timer, "Kernel time");


    // move GPU -> host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);


    // free
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


void matmul_cpu(float* A, float* B, float* C, int n) {
    for (int out_row = 0; out_row < n; out_row++) {
        for (int out_col = 0; out_col < n; out_col++) {
            int out_index = out_row * n + out_col;
            float res = 0;
            for (int i = 0; i < n; i++) {
                res += A[out_row * n + i] * B[i * n + out_col];
            }

            C[out_index] = res;
        }
    }
}



void check_results(float* A, float* B, int n, float tolerance) {
    for (int i = 0; i < n * n; i++) {
        if (abs(A[i] - B[i]) > tolerance) {
            printf("Results do not match! Index %d, CPU result = %f, GPU result = %f\n", i, A[i], B[i]);
            return;
        }
    }
    printf("Results match within tolerance of %f\n", tolerance);
}



int main(int argc, char** argv) {
    unsigned int n = 1<<9;


    float* A = (float*) malloc(n * n * sizeof(float));
    float* B = (float*) malloc(n * n * sizeof(float));

    float* C_gpu = (float*) malloc(n * n * sizeof(float));
    float* C_cpu = (float*) malloc(n * n * sizeof(float));

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            int i = row * n + col;
            A[i] = rand() / RAND_MAX;
            B[i] = rand() / RAND_MAX;
        }
    }

    CPUTimer cpuTimer;
    GPUTimer gpuTimer;

    cpuStartTime(&cpuTimer);
    matmul_cpu(A, B, C_cpu, n);
    cpuStopTime(&cpuTimer);
    cpuPrintElapsedTime(cpuTimer, "CPU time");

    gpuStartTime(&gpuTimer);
    matmul_gpu(A, B, C_gpu, n);
    gpuStopTime(&gpuTimer);
    gpuPrintElapsedTime(gpuTimer, "GPU time");

    check_results(C_cpu, C_gpu, n, 1e-6);


    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);
}