#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "utils.h"

void update(float *x, const float *grad, float *l2_norm, int m, int n, float lambda, float alpha) {
  *l2_norm = 0.0f;
  for (int i = 0; i < n; i++) {
    *l2_norm += x[i] * x[i];
    float g = grad[i] / (float)m + lambda * x[i];
    x[i] = x[i] - alpha * g;
  }
}

void compute(const float *x, float *grad, const int *A_row_ptr, const int *A_col_index, const float *A_value, const int *A_y_label, float *total_obj_val, int *correct, int m) {
  *total_obj_val = 0.0f;
  *correct = 0;

  for (int i = 0; i < m; i++) {
    float xp = 0.0f;
    for (int j = A_row_ptr[i]; j < A_row_ptr[i+1]; j++) {
      xp += A_value[j] * x[A_col_index[j]];
    }
    float v = logf(1.0f + expf(-xp * A_y_label[i]));
    *total_obj_val += v;
    float prediction = 1.0f / (1.0f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) {
      (*correct)++;
    }
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.0f + accum);
    for (int j = A_row_ptr[i]; j < A_row_ptr[i+1]; j++) {
      float temp = -accum * A_value[j] * A_y_label[i];
      grad[A_col_index[j]] += temp;
    }
  }
}