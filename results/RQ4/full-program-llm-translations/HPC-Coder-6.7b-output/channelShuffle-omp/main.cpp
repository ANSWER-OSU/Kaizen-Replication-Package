#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define NUM_THREADS 256

template <typename T, bool kNFirst>
void ChannelShuffleNCHWKernel(
    const int G,
    const int K,
    const int HxW,
    const T* X,
          T* Y)
{
  const int C = G * K;
  const int n = kNFirst ? omp_get_thread_num() / (K * G) : omp_get_thread_num() / (HxW * NUM_THREADS);
  const int s = kNFirst ? omp_get_thread_num() / K : omp_get_thread_num() % HxW;
  const int g = omp_get_thread_num() % G;
  const int k = omp_get_thread_num() / (K * G);
  const int offset = s * NUM_THREADS + omp_get_thread_num() % NUM_THREADS;
  if (offset < HxW) {
    Y[(n * C + g * K + k) * HxW + offset] = X[(n * C + g * K + k) * HxW + offset];
  }
}

template <typename T, int kSharedSize>
void ChannelShuffleNHWCKernel(const int G, const int K, const T* X, T* Y)
{
  T sdata[kSharedSize];
  const int C = G * K;
  const int offset = omp_get_thread_num() * C;
  for (int i = omp_get_thread_num() % C; i < C; i += omp_get_thread_num()) {
    sdata[i] = X[offset + i];
  }
  #pragma omp barrier
  for (int i = omp_get_thread_num() % C; i < C; i += omp_get_thread_num()) {
    const int g = i % G;
    const int k = i / G;
    Y[offset + i] = sdata[g * K + k];
  }
}

template <typename T>
bool ChannelShuffleNCHW (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int S = (HxW + NUM_THREADS - 1) / NUM_THREADS;

  auto start = std::chrono::steady_clock::now();

  #pragma omp parallel shared(X, Y, G, K, HxW) num_threads(S * N)
  {
    for (int i = 0; i < repeat; i++) {
      ChannelShuffleNCHWKernel<float, false>(G, K, HxW, X, Y);
    }
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

template <typename T>
bool ChannelShuffleNHWC (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int inner_size = N * HxW;
#define NUM_OMP_THREADS omp_get_max_threads()
#define BASE omp_get_thread_num()
#define LOCAL_ID omp_get_thread_num()

  auto start = std::chrono::steady_clock::now();

  if (C <= 32) {
    for (int i = 0; i < repeat; i++) {
      #pragma omp parallel num_threads(NUM_OMP_THREADS)
      {
        for (int j = BASE; j < inner_size; j += NUM_OMP_THREADS)
          ChannelShuffleNHWCKernel<float, 32>(G, K, X, Y);
      }
    }
  } else if (C <= 128) {
    for (int i = 0; i < repeat; i++) {
      #pragma omp parallel num_threads(NUM_OMP_THREADS)
      {
        for (int j = BASE; j < inner_size; j += NUM_OMP_THREADS)
          ChannelShuffleNHWCKernel<float, 128>(G, K, X, Y);
      }
    }
  } else if (C <= 512) {
    for (int i = 0; i < repeat; i++) {
      #pragma omp parallel num_threads(NUM_OMP_THREADS)
      {
        for (int j = BASE; j < inner_size; j += NUM_OMP_THREADS)
          ChannelShuffleNHWCKernel<float, 512>(G, K, X, Y);
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <group size> <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int G = atoi(argv[1]);
  const int W = atoi(argv[2]);
  const int H = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  long time;
  float *h_X, *h_Y, *h_Y_ref;
  size_t data_size_bytes;
  int error;

  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      const int numel = N * C * W * H; // assume no integer overflow
      data_size_bytes = numel * sizeof(float);

      h_X = (float*) malloc(data_size_bytes);
      for (int i = 0; i < numel; i++) h_X[i] = (float) i / numel;

      h_Y = (float*) malloc(data_size_bytes);
      h_Y_ref = (float*) malloc(data_size_bytes);

      ChannelShuffleNHWC (h_X, N, C, G, numel, h_Y, time, repeat);
      ChannelShuffleNHWC_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
      error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NHWC) check\n");
      else
        printf("Average time of channel shuffle (NHWC): %f (ms)\n", (time * 1e-6f) / repeat);

      ChannelShuffleNCHW (h_X, N, C, G, numel, h_Y, time, repeat);
      ChannelShuffleNCHW_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
      error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NCHW) check\n");
      else
        printf("Average time of channel shuffle (NCHW): %f (ms)\n", (time * 1e-6f) / repeat);

      free(h_X);
      free(h_Y);
      free(h_Y_ref);
    }
  }

  return 0;
}