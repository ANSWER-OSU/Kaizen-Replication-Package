#ifndef DWTHAAR1D_H_
#define DWTHAAR1D_H_

// Headers
#include <iostream>
#include <chrono>
#include <omp.h>

// CUDA Runtime
#include <cuda_runtime.h>

// CUDA Device Functions
__global__ void calApproxFinalOnDevice(float *inData, float *outData, unsigned int signalLength);

// Host Functions
int getLevels(unsigned int length, unsigned int* levels);
void calApproxFinalOnHost(float *inData, float *hOutData, unsigned int signalLength);

int runKernel(
    float *inData, 
    float *dOutData, 
    float *hOutData, 
    float *dPartialOutData,
    const unsigned int signalLength,
    float *inDataBuf,
    float *partialOutDataBuf,
    float *outDataBuf);

#endif