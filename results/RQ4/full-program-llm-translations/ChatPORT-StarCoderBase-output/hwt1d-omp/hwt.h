#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <omp.h>

int getLevels(unsigned int length, unsigned int* levels) {
  unsigned int l = 0;
  while (length > 1) {
    levels[l] = length;
    length = (length + 1) / 2;
    l++;
  }
  return l;
}

void calApproxFinalOnHost(float *inData, float *hOutData, unsigned int signalLength) {
  float sum;
  for (int i = 0; i < signalLength; i++) {
    sum = 0.f;
    #pragma omp parallel for reduction("+ : sum") num_threads(8)
    for (int j = 0; j < i + 1; j++) {
      sum += inData[j] * inData[i - j - 1];
    }
    hOutData[i] = sum;
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
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  double kernel_time;
  int ret = 0;
  unsigned int level = 0;
  unsigned int OMP_NUM_THREADS = omp_get_num_threads();
  unsigned int levels[16];
  int numTeams = 1;
  int numThreads = OMP_NUM_THREADS;
  while (level < 16) {
    if (numThreads > 1) {
      numTeams = (signalLength + numThreads - 1) / numThreads;
    }
    #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
    for (int i = 0; i < signalLength; i++) {
      partialOutDataBuf[i] = 0.f;
    }
    #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
    for (int i = 0; i < signalLength; i++) {
      if (i < (signalLength - level - 1)) {
        partialOutDataBuf[i] += inData[i] * inData[i + level + 1];
      }
    }
    #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
    for (int i = 0; i < signalLength; i++) {
      dPartialOutData[i] = partialOutDataBuf[i];
    }
    start = std::chrono::steady_clock::now();
    #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
    for (int team_id = 0; team_id < numTeams; team_id++) {
      for (int thread_id = 0; thread_id < numThreads; thread_id++) {
        int tid = team_id * numThreads + thread_id;
        if (tid < signalLength) {
          dOutData[tid] = 0.f;
          for (int j = 0; j < i + 1; j++) {
            dOutData[tid] += dPartialOutData[tid + j * numTeams] * inData[i - j - 1];
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    kernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (kernel_time > 600) {
      numThreads = numThreads / 2;
    } else {
      level++;
      if (level < 16) {
        numThreads = OMP_NUM_THREADS;
      }
    }
  }
  if (level == 16) {
    ret = runKernel(inData, dOutData, hOutData, dPartialOutData, signalLength, inDataBuf, partialOutDataBuf, outDataBuf);
  }
  return ret;
}