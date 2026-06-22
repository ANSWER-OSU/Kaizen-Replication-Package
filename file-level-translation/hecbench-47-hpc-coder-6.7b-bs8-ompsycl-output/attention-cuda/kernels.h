#include <omp.h>
#include <cub/cub.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

void attention_kernel1 (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  #pragma omp target teams distribute num_teams(n)
  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    #pragma omp atomic update
    (*exp_sum) += __expf(sum);
  }
}

void attention_kernel2 (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    float*__restrict__ score,
    const int n)
{
  #pragma omp target teams distribute num_teams(1)
  for (int i = 0; i < n; i++)
    score[i] = __expf(dot_product[i]) / exp_sum[0];
}

void attention_kernel3 (
    const float*__restrict__ score,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  #pragma omp target teams distribute num_teams(1) num_threads(d)
  for (int j = 0; j < d; j++) {
    float sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}

void attention_kernel1_blockReduce (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  const int threadsPerBlock = 256;
  const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  cub::DeviceReduce::Sum(nullptr, threadsPerBlock * numBlocks, key, dot_product, n, d, cub::DeviceReduceOptimization());
  cub::DeviceReduce::Sum(nullptr, threadsPerBlock * numBlocks, query, dot_product, n, d, cub::DeviceReduceOptimization());
  #pragma omp target teams distribute num_teams(numBlocks)
  for (int i = 0; i < numBlocks; i++) {
    float sum = 0;
    #pragma omp parallel for reduction(+ : sum) num_threads(threadsPerBlock)
    for (int j = 0; j < d; j++)
      sum += key[i * threadsPerBlock * d + j] * query[i * threadsPerBlock + j];
    dot_product[i] = sum;
    #pragma omp atomic update
    (*exp_sum) += __expf(sum);
  }
}

void attention_kernel1_warpReduce (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  namespace cg = cooperative_groups;
  const int threadsPerBlock = 256;
  const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  cub::DeviceReduce::Sum(nullptr, threadsPerBlock * numBlocks, key, dot_product, n, d, cub::DeviceReduceOptimization());
  cub::DeviceReduce::Sum(nullptr, threadsPerBlock * numBlocks, query, dot_product, n, d, cub::DeviceReduceOptimization());
  #pragma omp target teams distribute num_teams(numBlocks)
  for (int i = 0; i < numBlocks; i++) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    float sum = 0;
    for (int j = warp.thread_rank(); j < d; j += warp.size()) {
      sum += key[i * d + j] * query[j];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0) {
      dot_product[i] = sum;
      #pragma omp atomic update
      (*exp_sum) += __expf(sum);
    }
  }
}

void attention_kernel2_blockReduce (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  const int threadsPerBlock = 256;
  const int numBlocks = (d + threadsPerBlock - 1) / threadsPerBlock;
  #pragma omp target teams distribute num_teams(numBlocks)
  for (int j = 0; j < numBlocks; j++) {
    float sum = 0;
    #pragma omp parallel for reduction(+ : sum) num_threads(threadsPerBlock)
    for (int i = 0; i < n; i++) {
      float score = __expf(dot_product[i]) / exp_sum[0];
      sum += score * value[i * d + j];
    }
    output[j] = sum;
  }
}

void attention_kernel2_warpReduce (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  namespace cg = cooperative_groups;
  const int threadsPerBlock = 256;
  const int numBlocks = (d + threadsPerBlock - 1) / threadsPerBlock;
  #pragma omp target teams distribute num_teams(numBlocks)
  for (int j = 0; j < numBlocks; j++) {
    cg::thread_block block = cg::this_threadYou()
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    float sum = 0;
    for (int i = warp.thread_rank(); i < n; i += warp.size()) {
      float score = __expf(dot_product[i]) / exp_sum[0];
      sum += score * value[i * d + j];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0)
      output[j] = sum;
  }
}