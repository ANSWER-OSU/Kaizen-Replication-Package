#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define NUM_THREADS 256

// choose integer type to avoid floating-point rounding errors
typedef int scalar_t;

template <typename T, int kBlockDimX, int kBlockDimY>
__attribute__ ((always_inline)) void BlockReduce(T &input1, T &input2) {
  T private_val1[kBlockDimX][kBlockDimY];
  T private_val2[kBlockDimX][kBlockDimY];
  T *p1 = private_val1[0];
  T *p2 = private_val2[0];
  T block_sum_1 = 0;
  T block_sum_2 = 0;
  T thread_sum_1 = 0;
  T thread_sum_2 = 0;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  #pragma omp simd reduction(+:thread_sum_1, thread_sum_2)
  for (int i = tid; i < NUM_THREADS; i += blockDim.x) {
    p1[i] = input1 + i;
    p2[i] = input2 + i;
  }

  #pragma omp for
  for (int i = tid; i < NUM_THREADS; i += blockDim.x) {
    block_sum_1 += p1[i];
    block_sum_2 += p2[i];
  }

  if (tid < kBlockDimX * kBlockDimY) {
    T tmp_sum_1 = p1[0];
    T tmp_sum_2 = p2[0];
    for (int i = 1; i < NUM_THREADS; i++) {
      tmp_sum_1 += p1[i];
      tmp_sum_2 += p2[i];
    }
    input1 = tmp_sum_1;
    input2 = tmp_sum_2;
  }
}

template <typename T>
__attribute__ ((always_inline)) void BlockReduce(T &input1, T &input2) {
  T private_val1[NUM_THREADS];
  T private_val2[NUM_THREADS];
  T *p1 = private_val1;
  T *p2 = private_val2;
  T block_sum_1 = 0;
  T block_sum_2 = 0;
  T thread_sum_1 = 0;
  T thread_sum_2 = 0;
  int tid = omp_get_thread_num();

  #pragma omp simd reduction(+:thread_sum_1, thread_sum_2)
  for (int i = tid; i < NUM_THREADS; i += 1) {
    p1[i] = input1 + i;
    p2[i] = input2 + i;
  }

  #pragma omp for
  for (int i = tid; i < NUM_THREADS; i += 1) {
    block_sum_1 += p1[i];
    block_sum_2 += p2[i];
  }

  T tmp_sum_1 = p1[0];
  T tmp_sum_2 = p2[0];
  for (int i = 1; i < NUM_THREADS; i++) {
    tmp_sum_1 += p1[i];
    tmp_sum_2 += p2[i];
  }
  input1 = tmp_sum_1;
  input2 = tmp_sum_2;
}

template <typename T>
__attribute__ ((always_inline)) void ChannelSumNHWC(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* __restrict__ sum,
    T* __restrict__ sumsq,
    int tid)
{
  T m_val = 0;
  T v_val = 0;
  const int c = tid % C;
  const int n = tid / C;

  // sum batches from different channels
  for (int hw = 0; hw < HxW; hw++) {
    const int index = (n * C + c) * HxW + hw;
    m_val += X[index];
    v_val += X[index] * X[index];
  }
  BlockReduce<T>(m_val, v_val);
  if (tid == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
__attribute__ ((always_inline)) void ChannelSumNCHW(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* __restrict__ sum,
    T* __restrict__ sumsq,
    int tid)
{
  T m_val = 0;
  T v_val = 0;
  const int c = tid % C;

  // sum batches from different channels
  for (int n = 0; n < N; n++) {
    for (int hw = 0; hw < HxW; hw++) {
      const int index = (n * C + c) * HxW + hw;
      m_val += X[index];
      v_val += X[index] * X[index];
    }
  }
  BlockReduce<T,8,16,16>(m_val, v_val);
  if (tid % (C * 8) == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
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
    #pragma omp parallel num_threads(C)
    ChannelSumNCHW(N, C, HxW, X, sum, sumsq, omp_get_thread_num());
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

template <typename T>
void ComputeChannelSumNHWC (
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

  #pragma omp parallel for num_threads(C)
  for (int c = 0; c < C; c++) {
    int c_index = c * N * HxW;
    T m_val = 0;
    T v_val = 0;
    for (int i = 0; i < N * HxW; i++) {
      m_val += X[c_index + i];
      v_val += X[c_index + i] * X[c_index + i];
    }
    BlockReduce<T>(m_val, v_val);
    sum[c] = m_val;
    sumsq[c] = v_val;
  }

  auto end = std::chrono::steady_clock::now();
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

      rational_simulate(data_size_bytes, "X in bytes");

      ComputeChannelSumNHWC (N, C, W*H, h_X, h_sum, h_sumsq, time, repeat);

      rational_simulate(output_size_bytes, "sum and sumsq size in bytes");
      bool ok = check(C, h_sum, r_sum);
      printf("Verification %s for channel sum (nhwc) with OPENMP.\n", ok ? "PASS" : "FAIL");
      printf("Average time of channel sum (nhwc): %f (ms)\n", (time * 1e-6f) / repeat);


      ComputeChannelSumNCHW (N, C, W*H, h_X, h_sum, h_sumsq, time, repeat);

      rational_simulate(output_size_bytes, "sum and sumsq size in bytes");
      ok = check(C, h_sum, r_sum);
      printf("Verification %s for channel sum (nchw) with OPENMP.\n", ok ? "PASS" : "FAIL");
      printf("Average time of channel sum (nchw): %f (ms)\n", (time * 1e-6f) / repeat);

      free(h_X);
      free(h_sum);
      free(r_sum);
      free(h_sumsq);
    }
  }

  return 0;
}