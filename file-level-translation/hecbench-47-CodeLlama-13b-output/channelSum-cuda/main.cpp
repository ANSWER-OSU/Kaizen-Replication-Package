#include <iostream>
#include <chrono>
#include <omp.h>
#include <cassert>

#define N 64
#define C 32
#define W 224
#define H 224

using scalar_t = int;

template <typename T>
__device__ void BlockReduce(T &input1, T &input2) {
  using BlockReduce = cub::BlockReduce<T, 2, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 128>;
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;
  input1 = BlockReduce(temp_storage1).Sum(input1);
  input2 = BlockReduce(temp_storage2).Sum(input2);
}

template <typename T>
__device__ void BlockReduce(T &input1, T &input2) {
  using BlockReduce = cub::BlockReduce<T, 64>;
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;
  input1 = BlockReduce(temp_storage1).Sum(input1);
  input2 = BlockReduce(temp_storage2).Sum(input2);
}


template <typename T>
void ref_nhwc(T N, T C, T HxW, const T* X, T* sum, T* sumsq) {
  const int inner_size = N * HxW;
#pragma omp parallel for collapse(2) reduction(+:sum, sumsq)  
  for (int i = 0; i < inner_size; i++) {
    for (int c = 0; c < C; c++) {
      T m_val = 0;
      T v_val = 0;
      for (int n = 0; n < N; n++) {
	const int index = i * C + c;
	m_val += static_cast<float>(X[index]);
	v_val += static_cast<float>(X[index]) * static_cast<float>(X[index]);
      }
      sum[c] += m_val;
      sumsq[c] += v_val;
    }
  }
}

template <typename T>
void ref_nchw(T N, T C, T HxW, const T* X, T* sum, T* sumsq) {
#pragma omp parallel for collapse(1) reduction(+:sum, sumsq) 
  for (int c = 0; c < C; c++) {
    T m_val = 0;
    T v_val = 0;
    for (int n = 0; n < N; n++) {
      for (int hw = 0; hw < HxW; hw++) {
	const int index = (n * C + c) * HxW + hw;
	m_val += static_cast<float>(X[index]);
	v_val += static_cast<float>(X[index]) * static_cast<float>(X[index]);
      }
    }
    sum[c] += m_val;
    sumsq[c] += v_val;
  }
}

int main(int argc, char** argv) {
  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {
      int numel = N * C * W * H;
      scalar_t *h_X = (scalar_t*)malloc(numel * sizeof(scalar_t);
      scalar_t *h_sum = (scalar_t*)malloc(C * sizeof(scalar_t) * sizeof(scalar_t);
      scalar_t *h_sumsq = (scalar_t*)malloc(C * sizeof(scalar_t) * sizeof(scalar_t);
      scalar_t *r_sum = (scalar_t*)malloc(C * sizeof(scalar_t) * sizeof(scalar_t);

      srand(numel);
      for (int i = 0; i < numel; i++) {
	h_X[i] = rand() % 256;
      }

      long time;
      auto start = std::chrono::steady_clock::now();

      ref_nchw(N, C, W*H, h_X, h_sum, h_sumsq);
      auto end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      ref_nhwc(N, C, W*H, h_X, r_sum, h_sumsq);
      bool ok = check(C, h_sum, r_sum);
      
      printf("Average time of channel sum (nhwc): %f (ms)\n", (time * 1e-6f) / repeat);
      printf("Verification %s for channel sum (nhwc)\n", ok ? "PASS" : "FAIL");

      free(h_X);
      free(h_sum);
      free(r_sum);
      free(h_sumsq);
    }
  }
  return 0;
}