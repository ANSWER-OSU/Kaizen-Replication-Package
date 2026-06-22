#include <stdio.h>
#include <stdlib.h>
#include "common.h"

// OpenMP code reference
void permuate_cpu(float *inp, float *q, float *k, float *v, int B, int T, int C, int NH) {
  #pragma omp target teams distribute parallel for collapse(3) num_teams(B) num_threads(32)
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < NH; n++) {
      for (int t = 0; t < T; t++) {
        for (int c = n * (C/NH); c < (n+1)*(C/NH); c++) {
          q[b * T * 3 * C + t * 3 * C + c] = inp[b * T * 3 * C + t * 3 * C + c];
          k[b * T * 3 * C + t * 3 * C + c] = inp[b * T * 3 * C + t * 3 * C + C + c];
          v[b * T * 3 * C + t * 3 * C + c] = inp[b * T * 3 * C + t * 3 * C + 2*C + c];
        }
      }
    }
  }
}

void permute (float* out, const float* inp, int B, int T, int C, int NH, const int numTeams, const int numThreads) {
  #pragma omp target teams distribute parallel for collapse(3) num_teams(B) num_threads(numThreads)
  for (int b = 0; b < B; b++) {
    for (int nh = 0; nh < NH; nh++) {
      for (int t = 0; t < T; t++) {
        for (int d = 0; d < C; d++) {
          int idx = b * T * C + t * C + d;
          int inp_idx = (b * T * 3 * C) + (nh * 3 * C) + (t * 3 * C) + (d * C);
          out[idx] = inp[inp_idx];
        }
      }
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

  permuate_cpu(inp, q, k, v, B, T, C, NH);

  // move to GPU
  float *d_inp, *d_out;
  cudaCheck(cudaMalloc(&d_inp, S * 3 * sizeof(float)));
  cudaCheck(cudaMemcpy(d_inp, inp, S * 3 * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMalloc(&d_out, S * 3 * sizeof(float)));

  int numTeams = B;
  int numThreads = 32;
  int block_sizes[] = {32, 64, 128, 256, 512};

  for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    printf("Checking block size %d.\n", block_size);
    permute (d_out, d_inp, B, T, C, NH, numTeams, numThreads);
    validate_result(d_out, q, "q", S, 1e-6f);
    validate_result(d_out+B*T*C, k, "k", S, 1e-6f);
    validate_result(d_out+2*B*T*C, v, "v", S, 1e-6f);
  }
  printf("All results match. Starting benchmarks.\n\n");

  // benchmark speed of the kernel
  for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    float elapsed_time = benchmark_kernel(repeat_times, permute,
        d_out, d_inp, B, T, C, NH, numTeams, numThreads, block_size);

    printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
  }

  // free memory
  free(inp);
  free(q);
  free(k);
  free(v);
  free(out);
  cudaCheck(cudaFree(d_inp));
  cudaCheck(cudaFree(d_out));

  return 0;
}