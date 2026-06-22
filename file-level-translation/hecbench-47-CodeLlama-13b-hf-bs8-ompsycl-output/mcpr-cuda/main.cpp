void compute_probs(const double *__restrict__ alphas, const double *__restrict__ rands, double *__restrict__ probs, const int n, const int K, const int M) {
  #pragma omp target teams distribute parallel for num_teams(n) num_threads(256)
  for (int i = 0; i < n; i++) {
    int index_min = i * K;
    double local_min = alphas[index_min] / (1.0 + rands[index_min]);
    for (int j = 1; j < K; j++) {
      int index = i * K + j;
      double value = alphas[index] / (1.0 + rands[index]);
      if (value < local_min) local_min = value;
    }
    probs[i] = 1.0 - local_min;
  }
}

void compute_probs_unitStrides(const double *__restrict__ alphas, const double *__restrict__ rands, double *__restrict__ probs, const int n, const int K, const int M) {
  #pragma omp target teams distribute parallel for num_teams(n) num_threads(256)
  for (int i = 0; i < n; i++) {
    int index_min = i * K;
    double local_min = alphas[index_min] / (1.0 + rands[index_min]);
    for (int j = 1; j < K; j++) {
      int index = j;
      double value = alphas[index] / (1.0 + rands[index]);
      if (value < local_min) local_min = value;
    }
    probs[i] = 1.0 - local_min;
  }
}

void compute_probs_unitStrides_sharedMem(const double *__restrict__ alphas, const double *__restrict__ rands, double *__restrict__ probs, const int n, const int K, const int M) {
  #pragma omp target teams distribute parallel for num_teams(n) num_threads(256)
  for (int i = 0; i < n; i++) {
    double local_min = alphas[i] / (1.0 + rands[i]);
    #pragma omp parallel for reduction(min : local_min) num_threads(256)
    for (int j = 1; j < K; j++) {
      int index = j;
      double value = alphas[index] / (1.0 + rands[index]);
      if (value < local_min) local_min = value;
    }
    probs[i] = 1.0 - local_min;
  }
}