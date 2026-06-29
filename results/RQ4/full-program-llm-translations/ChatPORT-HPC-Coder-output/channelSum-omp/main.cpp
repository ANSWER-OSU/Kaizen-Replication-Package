#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cub/cub.cuh>
#include "reference.h"

#define NUM_THREADS 256

// choose integer type to avoid floating-point rounding errors
typedef int scalar_t;

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void ChannelSumNCHW(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  T m_val = 0;
  T v_val = 0;
  const int c = blockIdx.x;

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    for (int hw = threadIdx.y; hw < HxW; hw += blockDim.y) {
      const int index = (n * C + c) * HxW + hw;
      m_val += X[index];
      v_val += X[index] * X[index];
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    BlockReduce<T, kBlockDimX, kBlockDimY>(m_val, v_val);
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T, int kBlockSize>
__global__ void ChannelSumNHWC(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  const int inner_size = N * HxW;
  const int c = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int index = i * C + c;
    m_val += X[index];
    v_val += X[index] * X[index];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    BlockReduce<T, kBlockSize>(m_val, v_val);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T, int kBlockSize>
__global__ void ChannelSumNHWC_grid(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  const int inner_size = N * HxW;
  T m_val = 0;
  T v_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int index = i * C + blockIdx.x;
    m_val += X[index];
    v_val += X[index] * X[index];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    BlockReduce<T, kBlockSize>(m_val, v_val);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sum[blockIdx.x] = m_val;
    sumsq[blockIdx.x] = v_val;
  }
}

template <typename T, int kBlockSize>
void ComputeChannelSumNHWC_grid (
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* sum,
    T* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelSumNHWC_grid<T, kBlockSize><<<C, kBlockSize>>>(N, C, HxW, X, sum, sumsq);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

template <typename T, int kBlockDimX, int kBlockDimY>
void ComputeChannelSumNCHW (
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* sum,
    T* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelSumNCHW<T, kBlockDimX, kBlockDimY><<<C, dim3(NUM_THREADS, 1)>>>(N, C, HxW, X, sum, sumsq);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clockYou can find the above OpenMP translation of this CUDA code here:
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int W = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  long time;

  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      int numel = N * C * W * H; // assume no integer overflow

      size_t data_size_bytes = numel * sizeof(scalar_t);
      size_t output_size_bytes = C * sizeof(scalar_t);

      scalar_t *h_X = (scalar_t*) malloc (data_size_bytes);
      scalar_t *h_sum = (scalar_t*) malloc (output_size_bytes);
      scalar_t *h_sumsq = (scalar_t*) malloc (output_size_bytes);
      scalar_t *r_sum = (scalar_t*) malloc (output_size_bytes);

      srand(numel);
      for (int i = 0; i < numel; i++) h_X[i] = rand() % 256;

      scalar_t *d_X, *d_sum, *d_sumsq;
      auto errorX = cudaMalloc((void**)&d_X, data_size_bytes);
      auto errorY = cudaMalloc((void**)&d_sum, output_size_bytes);
      auto errorZ = cudaMalloc((void**)&d_sumsq, output_size_bytes);
      if (errorX != cudaSuccess ||
          errorY != cudaSuccess ||
          errorZ != cudaSuccess) {
        if (errorX == cudaSuccess) cudaFree(d_X);
        if (errorY == cudaSuccess) cudaFree(d_sum);
        if (errorZ == cudaSuccess) cudaFree(d_sumsq);
        printf("Device memory allocation failed. Exit\n");
        goto end;
      }

      cudaMemcpy(d_X, h_X, data_size_bytes, cudaMemcpyHostToDevice);

      ComputeChannelSumNHWC_grid <scalar_t, NUM_THREADS> (N, C, W*H, d_X, d_sum, d_sumsq, time, repeat);

      cudaMemcpy(h_sum, d_sum, output_size_bytes, cudaMemcpyDeviceToHost);
      ref_nhwc (N, C, W*H, h_X, r_sum, h_sumsq);
      bool ok = check(C, h_sum, r_sum);

      printf("Average time of channel sum (nhwc, grid): %f (ms)\n", (time * 1e-6f) / repeat);
      printf("Verification %s for channel sum (nhwc, grid)\n", ok ? "PASS" : "FAIL");

      ComputeChannelSumNCHW <scalar_t, 2, 128> (N, C, W*H, d_X, d_sum, d_sumsq, time, repeat);

      cudaMemcpy(h_sum, d_sum, output_size_bytes, cudaMemcpyDeviceToHost);
      ref_nchw (N, C, W*H, h_X, r_sum, h_sumsq);
      ok = check(C, h_sum, r_sum);
      
      printf("Average time of channel sum (nchw, 2-128): %f (ms)\n", (time * 1e-6f) / repeat);
      printf("Verification %s for channel sum (nchw, 2-128)\n", ok ? "PASS" : "FAIL");

      cudaFree(d_X);
      cudaFree(d_sum);
      cudaFree(d_sumsq);

      free(h_X);
      free(h_sum);
      free(r_sum);
      free(h_sumsq);
    }
  }
  
  end: return 0;
}