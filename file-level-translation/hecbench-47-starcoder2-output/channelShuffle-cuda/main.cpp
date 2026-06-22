#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

#define NUM_THREADS 256

template <typename T>
bool ChannelShuffleNCHW(T* X, int N, int C, int G, int numel, T* Y,
                        long& time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int S = (HxW + NUM_THREADS - 1) / NUM_THREADS;

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
      const int thread_id = omp_get_thread_num();
      const int s = thread_id / NUM_THREADS;
      const int n = thread_id % NUM_THREADS;
      const int g = blockIdx.z % G;
      const int k = blockIdx.z / G;
      const int offset = s * NUM_THREADS + thread_id;
      if (offset < HxW) {
        Y[(n * C + blockIdx.z) * HxW + offset] =
            X[(n * C + g * K + k) * HxW + offset];
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

template <typename T>
bool ChannelShuffleNHWC(T* X, int N, int C, int G, int numel, T* Y,
                        long& time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int outer_size = N * HxW;

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < outer_size; i++) {
      const int g = i % G;
      const int k = i / G;
      Y[i] = X[g * K + k];
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
  float* h_X, *h_Y, *h_Y_ref;

  // limited by the global device memory
  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      const int numel = N * C * W * H; // assume no integer overflow
      size_t data_size_bytes = numel * sizeof(float);

      h_X = (float*)malloc(data_size_bytes);
      for (int i = 0; i < numel; i++) h_X[i] = (float)i / numel;

      h_Y = (float*)malloc(data_size_bytes);
      h_Y_ref = (float*)malloc(data_size_bytes);

      ChannelShuffleNHWC(h_X, N, C, G, numel, h_Y, time, repeat);
      ChannelShuffleNHWC_cpu(h_X, N, C, G, numel, h_Y_ref, time, repeat);
      const int error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NHWC) check\n");
      else
        printf("Average time of channel shuffle (NHWC): %f (ms)\n", (time * 1e-6f) / repeat);

      ChannelShuffleNCHW(h_X, N, C, G, numel, h_Y, time, repeat);
      ChannelShuffleNCHW_cpu(h_X, N, C, G, numel, h_Y_ref, time, repeat);
      const int error = memcmp(h_Y, h_Y_ref, data_size_bytes);
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