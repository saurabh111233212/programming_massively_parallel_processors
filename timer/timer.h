// timer.h
#ifndef TIMER_H
#define TIMER_H

#include <cuda.h>

typedef struct {
    cudaEvent_t start;
    cudaEvent_t end;
} GPUTimer;

typedef struct {
    clock_t start;
    clock_t end;} CPUTimer;

void gpuStartTime(GPUTimer* timer);
void gpuStopTime(GPUTimer* timer);
void gpuPrintElapsedTime(GPUTimer timer, const char* message);

void cpuStartTime(CPUTimer* timer);
void cpuStopTime(CPUTimer* timer);
void cpuPrintElapsedTime(CPUTimer timer, const char* message);

#endif // TIMER_H