void attention_kernel1 (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    #pragma omp atomic
    *exp_sum += __expf(sum);
  }
}

void attention_kernel2 (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    float*__restrict__ score,
    const int n)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < n; i++)
    score[i] = __expf(dot_product[i]) / exp_sum[0];
}

void attention_kernel3 (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ score,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads) collapse(2)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++)
      output[j] += score[i] * value[i * d + j];
}

void attention_kernel1_blockReduce (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for (int i = 0; i < n; i++) {
    float sum = 0;
    #pragma omp parallel for reduction (+:sum) num_threads(numThreads)
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    #pragma omp atomic
    *exp_sum += __expf(sum);
  }
}

void attention_kernel1_warpReduce (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for (int i = 0; i < n; i++) {
    float sum = 0;
    #pragma omp parallel for reduction (+:sum) num_threads(numThreads)
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    #pragma omp atomic
    *exp_sum += __expf(sum);
  }
}

void attention_kernel2_blockReduce (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for (int j = 0; j < d; j++) {
    float sum = 0;
    #pragma omp parallel for reduction (+:sum) num_threads(numThreads)
    for (int i = 0; i < n; i++) {
      float score = __expf(dot_product[i]) / exp_sum[0];
      sum += score * value[i * d + j];
    }
    output[j] = sum;
  }
}

void attention_kernel2_warpReduce (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for (int j = 0; j < d; j++) {
    float sum = 0;
    #pragma omp parallel for reduction (+:sum) num_threads(numThreads)
    for (int i = 0; i < n; i++) {
      float score = __expf(dot_product[i]) / exp_sum[0];
      sum += score * value[i * d + j];
    }
    output[j] = sum;
  }
}