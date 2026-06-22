#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

void permuate_cpu(float *inp, float *q, float *k, float *v, int B, int T, int C, int NH) {
  int i = 0;
  #pragma omp parallel for num_threads(3)
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < NH; n++) {
      for (int t = 0; t < T; t++) {
        for (int c = n * (C/NH); c < (n+1)*(C/NH); c++) {
          q[i] = inp[b * T * 3 * C + t * 3 * C + c];
          k[i] = inp[b * T * 3 * C + t * 3 * C + C + c];
          v[i] = inp[b * T * 3 * C + t * 3 * C + 2*C + c];
          i++;
        }
      }
    }
  }
}

void permute (float* out, const float* inp, 
              int B, int T, int C, int NH, 
              const int threads, const int numTeams) {
  const int HS = C / NH;
  #pragma omp target teams num_teams(numTeams)
  {
    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    #pragma omp parallel num_threads(threads)
    {
      int t = omp_get_thread_num();
      int b = omp_get_team_num();

      int C1 = NH * HS;

      float *Q, *K, *V;
      Q = out + 0 * B * T * C1;
      K = out + 1 * B * T * C1;
      V = out + 2 * B * T * C1;
      
      for (int nh_ = 0; nh_ < C1; nh_++) {
        for (int n_ = 0; n_ < NH; n_++) {
          for (int t_ = 0; t_ < T; t_++) {
            int inp_idx = \
                    (b * T * 3 * C)
                    +   (n_ * 3 * C)
                    +       (t_ * 3 * C)
                    +           (nh_ * HS)
                    +                  t;
            Q[b * C1 * T + nh_ * T + t_] = inp[inp_idx];
            K[b * C1 * T + nh_ * T + t_] = inp[inp_idx + C];
            V[b * C1 * T + nh_ * T + t_] = inp[inp_idx + 2 * C];
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: %s <batch size> <T> <C> <repeat>\n", argv[0]);
    return 1;
  }
  const int B = atoi(argv[1]);
  const int T = atoi(argv[2]);
  const int C = atoi(argv[3]);
  const int repeat_times = atoi(argv[4]);

  const int NH = 12;
  const int HS = C / NH;

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

  // check result
  permute (d_out, d_inp, B, T, C, NH, 32, 1);
  validate_result(d_out, q, "q", S, 1e-6f);
  validate_result(d_out+B*T*C, k, "k", S, 1e-6f);
  validate_result(d_out+2*B*T*C, v, "v", S, 1e-6f);

  printf("All results match. Starting benchmarks.\n\n");

  // Benchmark the kernel
  float elapsed_time = benchmark_kernel(repeat_times, permute,
      d_out, d_inp, B, T, C, NH, 32, 1);

  printf("Threads: 32, NumTeams: 1, time %f ms\n", elapsed_time);

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