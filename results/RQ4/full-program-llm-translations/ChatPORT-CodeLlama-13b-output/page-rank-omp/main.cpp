void map(const int numTeams, const int numThreads, const int *__restrict__ pages, const float *__restrict__ page_ranks, float *__restrict__ maps, const unsigned int *__restrict__ noutlinks, const int n) {
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      maps[i * n + j] = pages[i * n + j] * page_ranks[i] / noutlinks[i];
     }
  }
}

void reduce(const int numTeams, const int numThreads, float *__restrict__ page_ranks, const float *__restrict__ maps, const int n, float *__restrict__ dif) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int j = 0; j < n; j++) {
    float new_rank = 0.0f;
    for (int i = 0; i < n; i++) {
      new_rank += maps[i * n + j];
    }
    new_rank = ((1.f - D_FACTOR) / n) + (D_FACTOR * new_rank);
    dif[j] = fmaxf(fabsf(new_rank - page_ranks[j]), dif[j]);
    page_ranks[j] = new_rank;
  }
}