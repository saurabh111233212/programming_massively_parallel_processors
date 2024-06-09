#include "timer.h"
#include <stdio.h>

#define TILE_DIM 32

__global__ void tiled_matmul_kernel(float* A, float* B, float* C, int n) {
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y  * blockIdx.y + blockIdx.y;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float res = 0.0f;

    for (unsigned int tile = 0; tile < n / TILE_DIM; tile++) {
        // populate the shared memory + synchronize 
        A_s[threadIdx.y][threadIdx.x] = A[(row * n) + tile*TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(threadIdx.y + tile*TILE_DIM)*n + col];
        __syncthreads();

        // loop over the tile, store intermediate results
        for(int i = 0; i < TILE_DIM; i++) {
            res += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads();
    }

    C[row*n + col] = res;
}

void matmul_gpu(float* A, float* B, float* C, int n) {
    // allocate on device
    float* A_d; float* B_d; float* C_d;
    unsigned int size = n*n*sizeof(float);
    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    // move host -> device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // call kernel
    GPUTimer kernel_timer;
    gpuStartTime(&kernel_timer);
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((n + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (n + numThreadsPerBlock.y - 1)/ numThreadsPerBlock.y);
    tiled_matmul_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, n);
    gpuStopTime(&kernel_timer);
    gpuPrintElapsedTime(kernel_timer, "Kernel time");


    // move device -> host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // free on device
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

int main() {
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