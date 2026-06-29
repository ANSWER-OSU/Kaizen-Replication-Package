/*
   Kernels for data permute 
 */

#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include <omp.h>

// OpenMP code reference
void permuate_openmp(float *inp, float *q, float *k, float *v, int B, int T, int C, int NH) {
  int i = 0;
#pragma omp parallel for collapse(4)
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < NH; n++) {
      for (int t = 0; t < T; t++) {
        for (int c = n * (C/NH); c < (n+1)*(C/NH); c++) {
          q[i] = inp[b * T * 3 * C + t * 3 * C + c];
          k[i] = inp[b * T * 3 * C + t * 3 * C + C + c];
          v[i] = inp[b * T * 3 * C + t * 3 * C + 2*C + c];
          i++;
        }}}}
}

void permute (float* out, const float* inp,
              int B, int T, int C, int NH,
              const int omp_threads)
{
  // inp is (B, T, 3C) QKV
  int HS = C / NH; // head size

  // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
  float *Q, *K, *V;
  Q = out + 0 * B * T * C;
  K = out + 1 * B * T * C;
  V = out + 2 * B * T * C;
  int total_threads = B * T * C;
  int num_threads = ceil_div(total_threads, omp_threads);

  #pragma omp parallel num_threads(omp_threads)
  {
  int tid = omp_get_thread_num();
  int st = tid * num_threads;
  int end = (tid + 1) * num_threads;

  if (end > total_threads) {
    end = total_threads;
  }

  for (int idx = st; idx < end; ++idx) {
    int b = idx / (C * T);
    int rest = idx % (C * T);
    int nh_ = rest / (T * HS);
    rest = rest % (T * HS);
    int n = rest / HS;
    int d_ = rest % HS;

    int inp_idx = \
        (b * T * 3 * C)
        +   (n * 3 * C)
        +       (0 * C)
        +          (nh_ * HS)
        +                d_;

    Q[idx] = inp[inp_idx];
    K[idx] = inp[inp_idx + C];
    V[idx] = inp[inp_idx + 2 * C];
  }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int B = atoi(argv[1]);
  const int repeat_times = atoi(argv[2]);

  const int T = 1024;
  const int C = 768;
  const int NH = 12;

  size_t S = (size_t)B * T * C;

  // create host memory of random numbers
  float* inp = make_random_float(S * 3);
  float* out = make_random_float(S * 3);
  float* q = make_random_float(S);
  float* k = make_random_float(S);
  float* v = make_random_float(S);

  permuate_openmp(inp, q, k, v, B, T, C, NH);

  // benchmark speed of the kernel
  int omp_threads[] = {1, 2, 4, 8, 16};

  for (size_t j = 0; j < sizeof(omp_threads) / sizeof(int); j++) {
    int threads = omp_threads[j];
    float elapsed_time = benchmark_func(repeat_times, permute,
        out, inp, B, T, C, NH, threads);

    printf("threads %d | time %f ms\n", threads, elapsed_time);
  }

  return 0;
}