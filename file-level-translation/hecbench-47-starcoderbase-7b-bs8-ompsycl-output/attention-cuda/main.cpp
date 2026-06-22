#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>

#include "kernels.h"
#include "reference.h"

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int impl_num, const int repeat) {
  size_t q_size = (size_t)d;
  size_t r_size = (size_t)n;
  size_t kv_size = (size_t)d * n;
  size_t r_size_bytes = r_size * sizeof(float);
  size_t q_size_bytes = q_size * sizeof(float);
  size_t kv_size_bytes = kv_size * sizeof(float);

  // input
  float *d_key;
  cudaMalloc((void**)&d_key, kv_size_bytes);
  cudaMemcpy(d_key, key, kv_size_bytes, cudaMemcpyHostToDevice);

  float *d_value;
  cudaMalloc((void**)&d_value, kv_size_bytes);
  cudaMemcpy(d_value, value, kv_size_bytes, cudaMemcpyHostToDevice);

  float *d_query;
  cudaMalloc((void**)&d_query, q_size_bytes);
  cudaMemcpy(d_query, query, q_size_bytes, cudaMemcpyHostToDevice);

  // intermediate
  float *d_dot_product;
  cudaMalloc((void**)&d_dot_product, r_size_bytes);

  float *d_exp_sum;
  cudaMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (q_size_bytes);
  float *d_output;
  cudaMalloc((void**)&d_output, q_size_bytes);

  #pragma omp target teams distribute parallel for reduction(+ : d_exp_sum) num_teams(r) num_threads(256)
  for (int k = 0; k < repeat; k++) {
    attention_kernel1_warpReduce(d_key, d_query, d_dot_product, d_exp_sum, n, d);
    attention_kernel2_blockReduce(d_exp_sum, d_dot_product, d_value, d_output, n, d);
  }

  #pragma omp target teams distribute parallel for num_teams(q_size) num_threads(256)
  for (int i = 0; i < q_size; i++) {
    output[i] = 0;
    #pragma omp parallel for reduction (+ : output[i]) num_teams(r) num_threads(256)
    for (int j = 0; j < r; j++) {
      output[i] += value[j * d + i] * dot_product[j];
    }
  }

  cudaFree(d_value);
  cudaFree(d_output);
  cudaFree(d_key);
  cudaFree(d_dot_product);
  cudaFree(d_exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <rows> <columns> <implementation> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: fused kernels with block reduce\n");
    printf("implementation 2: fused kernels with warp reduce\n");
    printf("implementation 3: fused kernels with mixed reduce\n");
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int r = atoi(argv[4]);

  // input
  size_t q_size = (size_t)d;
  size_t kv_size = (size_t)d * n;
  size_t q_size_bytes = q_size * sizeof(float);
  size_t kv_size_bytes = kv_size * sizeof(float);

  float* key = (float*) malloc (kv_size_bytes);
  float* value = (float*) malloc (kv_size_bytes);
  float* query = (float*) malloc (q_size_bytes);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  for (size_t i = 0; i < kv_size; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
    query[i % d] = dist(gen);
  }

  float* hout = attention_host(key, value, query, n, d);
  float* dout = attention_device(key, value, query, n, d, k, r);

  bool ok = true;
  for (int i = 0; i < d; i++) {
    if (fabsf(hout[i] - dout[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}