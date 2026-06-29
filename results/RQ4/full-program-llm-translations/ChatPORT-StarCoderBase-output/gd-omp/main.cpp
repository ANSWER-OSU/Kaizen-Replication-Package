#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "utils.h"
#include "reference.h"

void compute (
    const float * __restrict__ x,
          float * __restrict__ grad,
    const int   * __restrict__ A_row_ptr,
    const int   * __restrict__ A_col_index,
    const float * __restrict__ A_value,
    const int   * __restrict__ A_y_label,
    float * __restrict__ total_obj_val,
    int   * __restrict__ correct,
    int m ) {
  #pragma omp target teams distribute parallel for num_teams(m) num_threads(BLOCK_DIM)
  for (int i = 0; i < m; i++) {
    // Simple sparse matrix multiply x' = A * x
    float xp = 0.f;
    for ( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j) {
      xp += A_value[j] * x[A_col_index[j]];
    }

    // compute objective
    float v = logf(1.f + expf(-xp * A_y_label[i]));
    #pragma omp atomic update
    *total_obj_val += v;

    // compute errors
    float prediction = 1.f / (1.f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) {
      #pragma omp atomic update
      *correct += 1;
    }

    // compute gradient at x
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.f + accum);
    for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j) {
      float temp = -accum * A_value[j] * A_y_label[i];
      #pragma omp atomic update
      grad[A_col_index[j]] += temp;
    }
  }
}

int main(int argc, const char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <path to file> <lambda> <alpha> <repeat>\n", argv[0]);
    return 1;
  }
  const std::string file_path = argv[1];
  const float lambda = atof(argv[2]);
  const float alpha = atof(argv[3]);
  const int iters = atoi(argv[4]);

  //store the problem data in variable A and the data is going to be normalized
  Classification_Data_CRS A;
  get_CRSM_from_svm(A, file_path);

  const int m = A.m; // observations
  const int n = A.n; // features

  std::vector<float> x(n, 0.f);
  std::vector<float> grad (n);

  #pragma omp target teams distribute parallel for num_teams(m) num_threads(BLOCK_DIM)
  for (int i = 0; i < n; i++) {
    x[i] = 0.f;
    grad[i] = 0.f;
  }

  float total_obj_val = 0;
  int correct = 0;
  #pragma omp parallel for num_teams(m) num_threads(BLOCK_DIM)
  for (int k = 0; k < iters; k++) {
    // compute the total objective, correct rate, and gradient
    compute(A.row_ptr.data(), A.col_index.data(), A.values.data(), A.y_label.data(), m, n, lambda, alpha, &total_obj_val, &correct);

    // update x (gradient does not need to be updated)
    #pragma omp parallel for num_teams(m) num_threads(BLOCK_DIM)
    for (int i = 0; i < n; i++) {
      x[i] -= alpha * grad[i] / (float)m + lambda * x[i];
    }
  }

  reference(A, x, grad, m, n, iters, alpha, lambda, total_obj_val, (float)correct / (float)m);

  return 0;
}