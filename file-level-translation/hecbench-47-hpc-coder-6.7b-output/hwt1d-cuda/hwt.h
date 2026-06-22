#ifndef DWTHAAR1D_H_
#define DWTHAAR1D_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <omp.h>

int getLevels(unsigned int length, unsigned int* levels) {
    *levels = floor(log2(length)) - 1;
    return 0;
}

void calApproxFinalOnHost(float *inData, float *hOutData, unsigned int signalLength) {
    // Calculate and store the approximation of the Haar wavelet transform of the signal on the host
    for (unsigned int k = 1; k <= signalLength; k++) {
        hOutData[k] = 0.5 * (inData[2 * (k - 1)] + inData[2 * (k - 1) + 1]);
    }
}

int runKernel(
    float *inData, 
    float *dOutData, 
    float *hOutData, 
    float *dPartialOutData,
    const unsigned int signalLength,
    float *inDataBuf,
    float *partialOutDataBuf,
    float *outDataBuf) {
    
    // Call the host implementation of the wavelet transform
    calApproxFinalOnHost(inData, hOutData, signalLength);

    // Copy data to the device and execute the kernel in parallel
#pragma omp parallel for
    for (unsigned int k = 1; k <= signalLength; k++) {
        dOutData[k] = -0.5 * (hOutData[2 * (k - 1)] + hOutData[2 * (k - 1) + 1]);
        dPartialOutData[k] = 0.5 * (hOutData[2 * (k - 1)] - hOutData[2 * (k - 1) + 1]);
    }

    // Copy data back from the device
    for (unsigned int k = 1; k <= signalLength; k++) {
        outDataBuf[k] = dOutData[k];
    }

    return 0;
}

#endif