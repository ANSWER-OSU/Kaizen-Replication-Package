#include <immintrin.h>
#include <omp.h>

void attention_kernel1(
    const float *__restrict__ key,
    const float *__restrict__ query,
    float *__restrict__ dot_product,
    float *__restrict__ exp_sum,
    const int n,
    const int d)
{
  omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    float sum = 0;
#pragma omp simd
    for (int j = 0; j < d; j++)
    {
      sum += key[i * d + j] * query[j];
    }
    dot_product[i] = sum;
    atomicAdd(exp_sum, expf(sum));
  }
}

void attention_kernel2(
    const float *__restrict__ exp_sum,
    const float *__restrict__ dot_product,
    float *__restrict__ score,
    const int n)
{
  omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    float local_score = expf(dot_product[i]) / exp_sum[0];
#pragma omp atomic
    score[i] += local_score;
  }
}

void attention_kernel3(
    const float *__restrict__ score,
    const float *__restrict__ value,
    float *__restrict__ output,
    const int n,
    const int d)
{
  omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for simd collapse(2)
  for (int j = 0; j < d; j++)
  {
    for (int i = 0; i < n; i++)
    {
      output[j] += score[i] * value[i * d + j];
    }
  }
}