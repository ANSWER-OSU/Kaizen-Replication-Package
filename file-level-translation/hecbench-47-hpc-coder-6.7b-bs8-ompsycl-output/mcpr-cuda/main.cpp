#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include "kernels.h"
#include "reference.h"

// transpose
double* t(const double *idata, const int width, const int height) {
  double *odata = (double*) malloc (sizeof(double) * width * height);
  #pragma omp parallel for collapse(2)
  for (int yIndex = 0; yIndex < height; yIndex++) {
    for (int xIndex = 0; xIndex < width; xIndex++) {
      int index_in  = xIndex + width * yIndex;
      int index_out = yIndex + height * xIndex;
      odata[index_out] = idata[index_in];
    }
  }
  return odata;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <path to filename> <repeat>\n", argv[0]);
    return 1;
  }
  char *filename = argv[1];
  const int repeat = atoi(argv[2]);

  // n and K should match the dimension of the dataset in the csv file
  const int n = 26280, K = 21, M = 10000;

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Error: failed to open file alphas.csv. Exit\n");
    return 1;
  }

  int alphas_size = n * K; // n rows and K cols
  int alphas_size_byte = n * K * sizeof(double);

  int rands_size = M * K;  // M rows and K cols
  int rands_size_byte = M * K * sizeof(double);

  double *alphas, *rands, *probs, *probs_ref;
  alphas = (double*) malloc (alphas_size_byte);
  rands = (double*) malloc (rands_size_byte);
  probs = (double*) malloc (alphas_size_byte);
  probs_ref = (double*) malloc (alphas_size_byte);

  // load the csv file
  for (int i = 0; i < alphas_size; i++)
    fscanf(fp, "%lf", &alphas[i]);
  fclose(fp);

  // normal distribution (mean: 0 and var: 1)
  std::mt19937 gen(19937);
  std::normal_distribution<double> norm_dist(0.0,1.0);
  for (int i = 0; i < rands_size; i++) rands[i] = norm_dist(gen);

  reference(alphas, rands, probs_ref, n, K, M);

  double *d_alphas, *d_rands, *d_probs;

  #pragma omp target enter data map(to: rands[0:rands_size], alphas[0:alphas_size])
  #pragma omp target enter data map(alloc: probs[0:alphas_size])

  // kernel 1
  #pragma omp target teams distribute parallel for num_teams(ceil(1.0 * n / 192)) num_threads(192)
  for (int i = 0; i < n; i++) {
    compute_probs(alphas, rands, probs, n, K, M, i);
  }

  #pragma omp target update from(probs[0:alphas_size])

  double *t_rands = t(rands, K, M);
  double *t_alphas = t(alphas, K, n);

  reference_unitStrides(t_alphas, t_rands, probs_ref, n, K, M);

  #pragma omp target data map(to: t_rands[0:rands_size], t_alphas[0:alphas_size])
  {
    #pragma omp target teams distribute parallel for num_teams(ceil(1.0 * n / 96)) num_threads(96)
    for (int i = 0; i < n; i++) {
      compute_probs_unitStrides(t_alphas, t_rands, probs, n, K, M, i);
    }
  }

  const int sm_size = sizeof(double) * K * 96 * 2;

  #pragma omp target teams distribute parallel for num_teams(ceil(1.0 * n / 96)) num_threads(96) \
    shared(alphas, rands, probs) private(i) collapse(2)
  for (int j = 0; j < M; j++) {
    for (int i = 0; i < n; i++) {
      compute_probs_unitStrides_sharedMem(alphas, rands, probs, n, K, M, i, j);
    }
  }

  #pragma omp target update from(probs[0:alphas_size])

  verify(probs, probs_ref, alphas_size);

  #pragma omp target exit data map(release: rands[0:rands_size], alphas[0:alphas_size], probs[0:alphas_size])

  // free memory
  free(alphas);
  free(rands);
  free(t_alphas);
  free(t_rands);
  free(probs);
  free(probs_ref);
  return 0;
}