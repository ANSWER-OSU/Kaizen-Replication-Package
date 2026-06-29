#include <iostream>
#include <chrono>

void ComputeChannelSumNCHW (
    const int N,
    const int C,
    const int HxW,
    const double* X,
    double* sum,
    double* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; c++) {
      for (int n = 0; n < N; n++) {
        for (int hw = 0; hw < HxW; hw++) {
          const int index = (n * C + c) * HxW + hw;
          sum[c] += X[index];
          sumsq[c] += X[index] * X[index];
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void ComputeChannelSumNHWC (
    const int N,
    const int C,
    const int HxW,
    const double* X,
    double* sum,
    double* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for
    for (int c = 0; c < C; c++) {
      for (int i = 0; i < N * HxW; i++) {
        const int index = i * C + c;
        sum[c] += X[index];
        sumsq[c] += X[index] * X[index];
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}