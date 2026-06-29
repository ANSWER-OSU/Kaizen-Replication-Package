#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <random>

// Thread block size
#define TPB 256

__global__ void attention_kernel1(const float* key, const float* query, float* dot_product, float* exp_sum, const int n, const int d) {
  float k = key[blockIdx.x * d + threadIdx.x];
  float q = query[threadIdx.x];
  __shared__ float shared_mem[TPB];
  shared_mem[threadIdx.x] = k * q;
  __syncthreads();
  for(int stride = TPB / 2; stride > 0; stride >>= 1) {
    if(threadIdx.x < stride) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    dot_product[blockIdx.x] = shared_mem[0];
  }
}

__global__ void attention_kernel2(const float* exp_sum, const float* dot_product, float* score, const int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    score[idx] = exp(dot_product[idx] - exp_sum[0]);
  }
}

__global__ void attention_kernel3(const float* score, const float* value, float* output, const int n, const int d) {
  float s = score[threadIdx.x];
  float v = value[blockIdx.x * d + threadIdx.x];
  __shared__ float shared_mem[TPB];
  shared_mem[threadIdx.x] = s * v;
  __syncthreads();
  for(int stride = TPB / 2; stride > 0; stride >>= 1) {
    if(threadIdx.x < stride) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[blockIdx.x] = shared_mem[0];
  }
}

float* attention_host(const float* key, const float* value, const float* query,
                        const int n, const int d, const int impl_num, const int repeat)
{
  size_t q_size = (size_t)d;
  size_t r_size = (size_t)n;
  size_t kv_size = (size_t)d * n;

  float* dot_product = (float*)malloc(r_size * sizeof(float));
  float* exp_sum = (float*)malloc(sizeof(float));
  float* output = (float*)malloc(q_size * sizeof(float));

  for(int k = 0; k < repeat; k++) {
    memset(exp_sum, 0, sizeof(float));
    #pragma omp parallel for reduction(+ : exp_sum[:1])
    for(int i = 0; i < n; i++) {
      double sum = 0;
      for(int j = 0; j < d; j++) {
        sum += key[i * d + j] * query[j];
      }
      dot_product[i] = sum;
    }
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
      exp_sum[0] += dot_product[i];
    }
    float sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for(int i = 0; i < n; i++) {
      sum += exp(dot_product[i] - exp_sum[0]);
    }
    memset(dot_product, 0, r_size * sizeof(float));

    for(int i = 0; i < d; i++) {
      double sum = 0;
      for(int j = 0; j < n; j++) {
        sum += exp(dot_product[j] - exp_sum[0]) * value[j * d + i];
      }
      output[i] = sum;
    }
  }

  free(dot_product);
  free(exp_sum);
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