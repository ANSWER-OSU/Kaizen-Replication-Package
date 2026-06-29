#include <omp.h>
#include <math.h>

void attention_kernel1 (
    const float* key,
    const float* query,
    float* dot_product,
    float* exp_sum,
    const int n,
    const int d)
{
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    #pragma omp atomic
    exp_sum[0] += expf(sum);
  }
}

void attention_kernel2 (
    const float* exp_sum,
    const float* dot_product,
    float* score,
    const int n)
{
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    score[i] = expf(dot_product[i]) / exp_sum[0];
}

void attention_kernel3 (
    const float* score,
    const float* value,
    float* output,
    const int n,
    const int d)
{
#pragma omp parallel for
  for (int j = 0; j < d; j++) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}