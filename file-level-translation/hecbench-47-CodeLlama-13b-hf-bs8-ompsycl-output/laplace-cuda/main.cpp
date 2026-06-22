void cudamc_kernel(const int numTeams, const int numThreads, const int N, const int *__restrict__ a, const int *__restrict__ b, int *__restrict__ c) {
  #pragma omp target teams num_teams(numTeams)
  {
    #pragma omp parallel num_threads(numThreads)
    {
      int i = omp_get_thread_num();
      if (i < N) c[i] = a[i] + b[i];
    }
  }
}