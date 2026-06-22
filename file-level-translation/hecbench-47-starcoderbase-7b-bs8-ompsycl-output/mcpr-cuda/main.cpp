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
  #pragma omp target teams distribute parallel for collapse(2) num_teams(height) num_threads(width)
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
  #pragma omp target allocate num_teams(8)
  {
    #pragma omp target teams distribute num_teams(8)
    for (int i = 0; i < 8; i++) {
      d_rands[i * K * M] = 0;
      #pragma omp parallel for collapse(2) num_threads(K)
      for (int j = 0; j < M; j++) {
        for (int k = 0; k < K; k++) {
          int index = j * K + k;
          d_rands[index * M + i] = rands[index];
        }
      }
    }
    #pragma omp target teams distribute num_teams(8)
    for (int i = 0; i < 8; i++) {
      d_alphas[i * K * n] = 0;
      #pragma omp parallel for collapse(2) num_threads(K)
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < K; k++) {
          int index = j * K + k;
          d_alphas[index * n + i] = alphas[index];
        }
      }
    }
    #pragma omp target teams distribute num_teams(8)
    for (int i = 0; i < 8; i++) {
      d_probs[i * K * n] = 0;
    }
  }

  // kernel 1
  int threads_per_block = 192;
  #pragma omp target teams distribute parallel for num_teams(8) num_threads(192)
  for (int i = 0; i < 8; i++) {
    compute_probs<<<1, 192>>>(d_alphas + i * K * n, d_rands + i * K * M, d_probs + i * K * n, n, K, M);
  }

  // kernel 2
  double *t_rands = t(rands, K, M);
  double *t_alphas = t(alphas, K, n);

  reference_unitStrides(t_alphas, t_rands, probs_ref, n, K, M);

  #pragma omp target teams distribute parallel for num_teams(8) num_threads(192)
  for (int i = 0; i < 8; i++) {
    compute_probs_unitStrides<<<1, 192>>>(d_alphas + i * K * n, d_rands + i * K * M, d_probs + i * K * n, n, K, M);
  }

  // kernel 3
  threads_per_block = 96;
  #pragma omp target teams distribute parallel for num_teams(8) num_threads(96)
  for (int i = 0; i < 8; i++) {
    const int sm_size = sizeof(double) * K * threads_per_block * 2;
    compute_probs_unitStrides_sharedMem<<<1, 96, sm_size, 0>>>(d_alphas + i * K * n, d_rands + i * K * M, d_probs + i * K * n, n, K, M);
  }

  // copy out results
  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(K)
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < K; k++) {
      int index = j * K + k;
      probs[index * n + omp_get_team_num()] = d_probs[index * n + omp_get_team_num()];
    }
  }

  // free memory
  #pragma omp target teams distribute num_teams(8)
  for (int i = 0; i < 8; i++) {
    #pragma omp parallel for collapse(2) num_threads(K)
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < K; k++) {
        int index = j * K + k;
        d_rands[index * M + i] = 0;
        d_alphas[index * n + i] = 0;
        d_probs[index * n + i] = 0;
      }
    }
  }
  #pragma omp target teams num_teams(8)
  {
    #pragma omp parallel for collapse(2) num_threads(K)
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < K; k++) {
        int index = j * K + k;
        rands[index] = 0;
        alphas[index] = 0;
        probs[index * n + omp_get_team_num()] = 0;
      }
    }
  }
  free(t_alphas);
  free(t_rands);
  cudaFree(d_alphas);
  cudaFree(d_rands);
  cudaFree(d_probs);
  free(alphas);
  free(rands);
  free(probs);
  free(probs_ref);
  return 0;
}