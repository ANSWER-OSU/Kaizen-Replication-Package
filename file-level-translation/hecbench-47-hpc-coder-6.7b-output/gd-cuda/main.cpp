#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cmath>
#include "utils.h"
#include "reference.h"

void
update(float *__restrict__ x,
       const float * __restrict__ grad,
       float* l2_norm,
       int m, int n, float lambda, float alpha)
{
  float s;
#pragma omp parallel for reduction(+:s)
  for (int i = 0; i < n; i++) {
    s += x[i] * x[i];
    float g = grad[i] / (float)m + lambda * x[i];
    x[i] = x[i] - alpha * g;
  }
#pragma omp atomic
  *l2_norm += s;
}

void
compute(const float * __restrict__ x,
          float * __restrict__ grad,
    const int   * __restrict__ A_row_ptr,
    const int   * __restrict__ A_col_index,
    const float * __restrict__ A_value,
    const int   * __restrict__ A_y_label,
    float * __restrict__ total_obj_val,
    int   * __restrict__ correct,
    int m )
{
  for (int i = 0; i < m; i+=omp_get_num_threads()) {
    // Simple sparse matrix multiply x' = A * x
    float xp = 0.f;
#pragma omp simd
    for( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      xp += A_value[j] * x[A_col_index[j]];
    }

    // compute objective
    float v = logf(1.f + expf(-xp * A_y_label[i]));
#pragma omp atomic
    *total_obj_val += v;

    // compute errors
    float prediction = 1.f / (1.f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) #pragma omp atomic
      (*correct)++;

    // compute gradient at x
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.f + accum);
#pragma omp simd
    for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      float temp = -accum * A_value[j] * A_y_label[i];
#pragma omp atomic
      grad[A_col_index[j]] += temp;
    }
  }
}

void
update(float * __restrict__ x,
       const float * __restrict__ grad,
       int m, int n, float lambda, float alpha)
{
  for (int i = 0; i < n; i++) {
    float g = grad[i] / (float)m + lambda * x[i];
    x[i] = x[i] - alpha * g;
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
  const int iters = atof(argv[4]);

  //store the problem data in variable A and the data is going to be normalized
  Classification_Data_CRS A;
  get_CRSM_from_svm(A, file_path);

  const int m = A.m; // observations
  const int n = A.n; // features

  std::vector<float> x(n, 0.f);
  std::vector<float> grad (n);

  float total_obj_val = 0.f;
  float l2_norm = 0.f;
  int correct = 0;

  long long train_start = get_time();

  float obj_val;
  float train_error;

  #pragma omp parallel shared(x, grad, A, lambda, alpha, total_obj_val, l2_norm, correct)
  {
    for (int k = 0; k < iters; k++) {

      // reset the training status
      total_obj_val = 0.f;
      l2_norm = 0.f;
      correct = 0;

      //reset gradient vector
      fill(grad.begin(), grad.end(), 0.f);

      // compute the total objective, correct rate, and gradient
      compute(
        x.data(),
        grad.data(),
        A.row_ptr.data(),
        A.col_index.data(),
        A.values.data(),
        A.y_label.data(),
        &total_obj_val,
        &correct,
        m
      );

      // update x (gradient does not need to be updated)
      update(x.data(), grad.data(), &l2_norm, m, n, lambda, alpha);
    }
  }

  long long train_end = get_time();
  printf("Training time takes %lf (s) for %d iterations\n\n",
         (train_end - train_start) * 1e-6, iters);

  obj_val = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
  train_error = 1.f-(correct/(float)m);

  printf("object value = %f train_error = %f\n", obj_val, train_error);

  reference(A, x, grad, m, n, iters, alpha, lambda, obj_val, train_error);

  return 0;
}