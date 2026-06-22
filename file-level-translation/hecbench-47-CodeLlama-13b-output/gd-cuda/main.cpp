#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <openmp.h>

__global__ void
update(float *__restrict__ x,
       const float * __restrict__ grad,
       int m, int n, float lambda, float alpha)
{
  typedef omp::BlockReduce<float, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage t;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float s = 0;
  if (i < n) {
    s = BlockReduce(t).Sum(x[i]*x[i]);
    float g = grad[i] / (float)m + lambda * x[i];
    x[i] = x[i] - alpha * g;
  }
  if (threadIdx.x == 0) omp_atomic_add(l2_norm, s);
}

__global__ void
compute (
    const float * __restrict__ x,
          float * __restrict__ grad,
    const int   * __restrict__ A_row_ptr,
    const int   * __restrict__ A_col_index,
    const float * __restrict__ A_value,
    const int   * __restrict__ A_y_label,
    float * __restrict__ total_obj_val,
    int   * __restrict__ correct,
    int m )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < m) {
    // Simple sparse matrix multiply x' = A * x
    float xp = 0.f;
    for( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      xp += A_value[j] * x[A_col_index[j]];
    }

    // compute objective
    float v = logf(1.f + expf(-xp * A_y_label[i]));
    omp_atomic_add(total_obj_val, v);

    // compute errors
    float prediction = 1.f / (1.f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) omp_atomic_add(correct, 1);

    // compute gradient at x
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.f + accum);
    for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      float temp = -accum * A_value[j] * A_y_label[i];
      omp_atomic_add(&grad[A_col_index[j]], temp);
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
  const int iters = atof(argv[4]);

  //store the problem data in variable A and the data is going to be normalized
  Classification_Data_CRS A;
  get_CRSM_from_svm(A, file_path);

  const int m = A.m; // observations
  const int n = A.n; // features

  std::vector<float> x(n, 0.f);
  std::vector<float> grad (n);

  float *d_x;
  omp_alloc((void**)&d_x, n * sizeof(float));
  omp_memcpy(d_x, x.data(), n * sizeof(float), omp_MemcpyHostToDevice);

  float *d_grad;
  omp_alloc((void**)&d_grad, n * sizeof(float));

  float *d_total_obj_val;
  omp_alloc((void**)&d_total_obj_val, sizeof(float));

  float *d_l2_norm;
  omp_alloc((void**)&d_l2_norm, sizeof(float));

  int *d_correct;
  omp_alloc((void**)&d_correct, sizeof(int));

  int *d_row_ptr;
  omp_alloc((void**)&d_row_ptr, A.row_ptr.size() * sizeof(int));
  omp_memcpy(d_row_ptr, A.row_ptr.data(),
             A.row_ptr.size() * sizeof(int), omp_MemcpyHostToDevice);

  int *d_col_index;
  omp_alloc((void**)&d_col_index, A.col_index.size() * sizeof(int));
  omp_memcpy(d_col_index, A.col_index.data(),
             A.col_index.size() * sizeof(int), omp_MemcpyHostToDevice);

  float *d_value;
  omp_alloc((void**)&d_value, A.values.size() * sizeof(float));
  omp_memcpy(d_value, A.values.data(),
             A.values.size() * sizeof(float), omp_MemcpyHostToDevice);

  int *d_y_label;
  omp_alloc((void**)&d_y_label, A.y_label.size() * sizeof(int));
  omp_memcpy(d_y_label, A.y_label.data(),
             A.y_label.size() * sizeof(int), omp_MemcpyHostToDevice);

  dim3 grid ((m+255)/256);
  dim3 block (256);

  dim3 grid2 ((n+255)/256);
  dim3 block2 (256);

  float obj_val;
  float train_error;

  omp_synchronize();
  long long train_start = get_time();

  float total_obj_val;
  float l2_norm;
  int correct;

  for (int k = 0; k < iters; k++) {

    // reset the training status
    omp_memset(d_total_obj_val, 0, sizeof(float));
    omp_memset(d_l2_norm, 0, sizeof(float));
    omp_memset(d_correct, 0, sizeof(int));

    //reset gradient vector
    omp_memset(d_grad, 0, n * sizeof(float));

    // compute the total objective, correct rate, and gradient
    compute<<<grid, block>>>(
        d_x,
        d_grad,
        d_row_ptr,
        d_col_index,
        d_value,
        d_y_label,
        d_total_obj_val,
        d_correct,
        m
      );

    // update x (gradient does not need to be updated)
    update<<<grid2, block2>>>(d_x, d_grad, d_l2_norm, m, n, lambda, alpha);
  }
  omp_synchronize();

  long long train_end = get_time();
  printf("Training time takes %lf (s) for %d iterations\n\n",
         (train_end - train_start) * 1e-6, iters);

  omp_memcpy(&total_obj_val, d_total_obj_val, sizeof(float), omp_MemcpyDeviceToHost);
  omp_memcpy(&l2_norm, d_l2_norm, sizeof(float), omp_MemcpyDeviceToHost);
  omp_memcpy(&correct, d_correct, sizeof(int), omp_MemcpyDeviceToHost);

  obj_val = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
  train_error = 1.f-(correct/(float)m);

  printf("object value = %f train_error = %f\n", obj_val, train_error);

  reference(A, x, grad, m, n, iters, alpha, lambda, obj_val, train_error);

  omp_free(d_row_ptr);
  omp_free(d_col_index);
  omp_free(d_value);
  omp_free(d_y_label);
  omp_free(d_x);
  omp_free(d_grad);
  omp_free(d_total_obj_val);
  omp_free(d_l2_norm);
  omp_free(d_correct);

  return 0;
}