#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define NUM_THREADS 256
#define GridDimMaxY 65536

template <typename T, bool kNFirst>
void ChannelShuffleNCHWKernel(const int numTeams, const int numThreads, const int O, const int G, const int K, const int HxW, const T* X, T* Y) {
  #pragma omp target teams distribute parallel for collapse(3) num_teams(numTeams) num_threads(numThreads)
  for (int n = 0; n < O; n++) {
    for (int k = 0; k < K; k++) {
      for (int i = 0; i < HxW; i++) {
        const int offset = (kNFirst ? s : n) * NUM_THREADS + i;
        if (offset < HxW) {
          Y[(n * C + O * k + g) * HxW + offset] = X[(n * C + g * K + k) * HxW + offset];
        }
      }
    }
  }
}

template <typename T, int kSharedSize>
void ChannelShuffleNHWCKernel(const int numTeams, const int numThreads, const int O, const int G, const int K, const T* X, T* Y) {
  #pragma omp target teams num_teams(numTeams)
  {
    T sdata[kSharedSize];
    #pragma omp parallel num_threads(numThreads)
    {
      const int C = G * K;
      const int offset = omp_get_team_num() * C;
      for (int i = omp_get_thread_num(); i < C; i += omp_get_num_threads()) {
        sdata[i] = X[offset + i];
      }
      #pragma omp barrier
      for (int i = omp_get_thread_num(); i < C; i += omp_get_num_threads()) {
        const int g = i % G;
        const int k = i / G;
        Y[offset + i] = sdata[g * K + k];
      }
    }
  }
}

template <typename T>
bool ChannelShuffleNCHW (const int numTeams, const int numThreads, T *X, int N, int C, int G, int numel, T *Y, long &time, int repeat) {
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int S = (HxW + NUM_THREADS - 1) / NUM_THREADS;

  auto start = std::chrono::steady_clock::now();

  if (N <= GridDimMaxY) {
    ChannelShuffleNCHWKernel<T, false>(numTeams, numThreads, N, G, K, HxW, X, Y);
  } else {
    ChannelShuffleNCHWKernel<T, true>(numTeams, numThreads, N, G, K, HxW, X, Y);
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

template <typename T>
bool ChannelShuffleNHWC (const int numTeams, const int numThreads, T *X, int N, int C, int G, int numel, T *Y, long &time, int repeat) {
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int outer_size = N * HxW;

  auto start = std::chrono::steady_clock::now();

  if (C <= 32) {
    ChannelShuffleNHWCKernel<T, 32>(numTeams, numThreads, outer_size, G, K, X, Y);
  } else if (C <= 128) {
    ChannelShuffleNHWCKernel<T, 128>(numTeams, numThreads, outer_size, G, K, X, Y);
  } else if (C <= 512) {
    ChannelShuffleNHWCKernel<T, 512>(numTeams, numThreads, outer_size, G, K, X, Y);
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <numTeams> <numThreads> <group size> <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int numTeams = atoi(argv[1]);
  const int numThreads = atoi(argv[2]);
  const int G = atoi(argv[3]);
  const int W = atoi(argv[4]);
  const int H = atoi(argv[5]);
  const int repeat = atoi(argv[6]);

  long time;
  float *h_X, *h_Y, *h_Y_ref;
  float *d_X, *d_Y;
  int error;

  // limited by the global device memory
  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      const int numel = N * C * W * H; // assume no integer overflow
      size_t data_size_bytes = numel * sizeof(float);

      auto errorX = cudaMalloc((void**)&d_X, data_size_bytes);
      auto errorY = cudaMalloc((void**)&d_Y, data_size_bytes);
      if (errorX != cudaSuccess || errorY != cudaSuccess) {
        if (errorX == cudaSuccess) cudaFree(d_X);
        if (errorY == cudaSuccess) cudaFree(d_Y);
        printf("Device memory allocation failed. Exit\n");
        goto end;
      }

      h_X = (float*) malloc(data_size_bytes);
      for (int i = 0; i < numel; i++) h_X[i] = (float) i / numel;

      h_Y = (float*) malloc(data_size_bytes);
      h_Y_ref = (float*) malloc(data_size_bytes);

      cudaMemcpy(d_X, h_X, data_size_bytes, cudaMemcpyHostToDevice);

      ChannelShuffleNHWC (numTeams, numThreads, N, C, G, numel, d_Y, time, repeat);
      ChannelShuffleNHWC_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
      cudaMemcpy(h_Y, d_Y, data_size_bytes, cudaMemcpyDeviceToHost);
      error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NHWC) check\n");
      else
        printf("Average time of channel shuffle (NHWC): %f (ms)\n", (time * 1e-6f) / repeat);

      ChannelShuffleNCHW (numTeams, numThreads, d_X, N, C, G, numel, d_Y, time, repeat);
      ChannelShuffleNCHW_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
      cudaMemcpy(h_Y, d_Y, data_size_bytes, cudaMemcpyDeviceToHost);
      error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NCHW) check\n");
      else
        printf("Average time of channel shuffle (NCHW): %f (ms)\n", (time * 1e-6f) / repeat);

      cudaFree(d_X);
      cudaFree(d_Y);
      free(h_X);
      free(h_Y);
      free(h_Y_ref);
    }
  }

  end: return 0;
}