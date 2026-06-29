#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>

#define GPU_THREADS 256

template <typename T>
void eval_swish (const int numTeams, const int numThreads, const int N, const int repeat) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < N; i++) {
    y[i] = x[i] / (T(1) + exp(-x[i]));
  }
}

template <typename T>
void SwishGradientKernel( const int numTeams, const int numThreads, const int N, const T* X, const T* Y, const T* dY, T* dX) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < N; i++) {
    dX[i] = dY[i] * (Y[i] + (T(1) - Y[i]) / (T(1) + exp(-X[i]));
  }
}

template <typename T>
void eval_swish_gradient (const int numTeams, const int numThreads, const int N, const int repeat) {
  size_t size_bytes = N * sizeof(T);

  T *h_X  = (T*) malloc (size_bytes);
  T *h_Y  = (T*) malloc (size_bytes);
  T *h_dY = (T*) malloc (size_bytes);
  T *h_dX = (T*) malloc (size_bytes);
  T *r_Y  = (T*) malloc (size_bytes);
  T *r_dX = (T*) malloc (size_bytes);

  std::default_random_engine gen (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);
  for (int i = 0; i < N; i++) {
    h_X[i] = distr(gen);
    h_dY[i] = distr(gen);
  }

  T *d_X, *d_Y, *d_dX, *d_dY;
  cudaMalloc((void**)&d_X, size_bytes);
  cudaMemcpy(d_X, h_X, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_Y, size_bytes);

  cudaMalloc((void**)&d_dY, size_bytes);
  cudaMemcpy(d_dY, h_dY, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_dX, size_bytes);

  const int numBlocks = (N + GPU_THREADS - 1) / GPU_THREADS;
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < repeat; i++) {
    SwishKernel<T> << <numBlocks, GPU_THREADS >> > (N, d_X, d_Y);
  }
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < repeat; i++) {
    SwishGradientKernel<T> << <numBlocks, GPU_THREADS >> > (N, d_X, d_Y, d_dY, d_dX);
  }

  // verify
  cudaMemcpy(h_dX, d_dX, size_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Y, d_Y, size_bytes, cudaMemcpyDeviceToHost);

  reference (N, h_X, r_Y, r_dX, h_dY);

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_dX[i] - r_dX[i]) > 1e-3 || fabs(h_Y[i] - r_Y[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_dX);
  cudaFree(d_dY);

  free(h_X);
  free(h_Y);
  free(h_dX);
  free(h_dY);
  free(r_dX);
  free(r_Y);
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of teams> <number of threads per team> <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int numTeams = atoi(argv[1]);
  const int numThreadsPerTeam = atoi(argv[2]);
  const int N = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  eval_swish<float>(numTeams, numThreadsPerTeam, N, repeat);
  eval_swish_gradient<float>(numTeams, numThreadsPerTeam, N, repeat);

  return 0;
}