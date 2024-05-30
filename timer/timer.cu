// timer.cu
#include <stdio.h>
#include "timer.h"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"


void gpuStartTime(GPUTimer* timer) {
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->end);
    cudaEventRecord(timer->start, 0);
}

void gpuStopTime(GPUTimer* timer) {
    cudaEventRecord(timer->end, 0);
    cudaEventSynchronize(timer->end);  // wait for event to be recorded
    cudaDeviceSynchronize();  // wait for device to finish
}

void gpuPrintElapsedTime(GPUTimer timer, const char* message) {
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, timer.start, timer.end);
    printf(ANSI_COLOR_GREEN "%s: %f milliseconds\n" ANSI_COLOR_RESET, message, elapsed_time);
    cudaEventDestroy(timer.start);
    cudaEventDestroy(timer.end);
}


void cpuStartTime(CPUTimer* timer) {
    timer->start = clock();
}

void cpuStopTime(CPUTimer* timer) {
    timer->end = clock();
}

void cpuPrintElapsedTime(CPUTimer timer, const char* message) {
    double elapsed_time = ((double)(timer.end - timer.start) / CLOCKS_PER_SEC) * 1000;
    printf(ANSI_COLOR_CYAN "%s: %f milliseconds\n" ANSI_COLOR_RESET, message, elapsed_time);
}
