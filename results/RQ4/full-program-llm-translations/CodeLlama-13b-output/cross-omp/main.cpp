#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include <omp.h>

template <typename T, typename StrideType>
__global__ void cross_kernel(
    int numel,
          T* out,
    const T* x1,
    const T* x2,
    StrideType ostride,
    StrideType x1stride,
    StrideType x2stride)
{
  #pragma omp for schedule(static)
  for (int i = 0; i < numel; i++) {

    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const T val0 = (x1_row[1 * x1stride] * x2_row[2 * x2stride] -
                    x1_row[2 * x1stride] * x2_row[1 * x2stride]);

    const T val1 = (x1_row[2 * x1stride] * x2_row[0 * x2stride] -
                    x1_row[0 * x1stride] * x2_row[2 * x2stride]);

    const T val2 = (x1_row[0 * x1stride] * x2_row[1 * x2stride] -
                    x1_row[1 * x1stride] * x2_row[0 * x2stride]);

    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

template <typename T, typename StrideType>
__global__ void cross2_kernel(
    int numel,
          T* out,
    const T* x1,
    const T* x2,
    StrideType ostride,
    StrideType x1stride,
    StrideType x2stride)
{
  #pragma omp for schedule(static)
  for (int i = 0; i < numel; i++) {

    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const T x1_c0 = x1_row[0 * x1stride];
    const T x1_c1 = x1_row[1 * x1stride];
    const T x1_c2 = x1_row[2 * x1stride];
    const T x2_c0 = x2_row[0 * x2stride];
    const T x2_c1 = x2_row[1 * x2stride];
    const T x2_c2 = x2_row[2 * x2stride];

    const T val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1 ;

    const T val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2 ;

    const T val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0 ;

    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

template <typename T>
__global__ void cross3_kernel(
    int numel,
          T* out,
    const T* x1,
    const T* x2)
{
  #pragma omp for schedule(static)
  for (int i = 0; i < numel; i++) {

    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const T x1_c0 = x1_row[0];
    const T x1_c1 = x1_row[1];
    const T x1_c2 = x1_row[2];
    const T x2_c0 = x2_row[0];
    const T x2_c1 = x2_row[1];
    const T x2_c2 = x2_row[2];

    const T val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1 ;

    const T val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2 ;

    const T val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0 ;

    out_row[0] = val0;
    out_row[1] = val1;
    out_row[2] = val2;
  }
}

template <typename T>
void eval(const int nrows, const int repeat) {
  const int num_elems = nrows * 3;
  const int size_bytes = num_elems * sizeof(T); 

  T *a, *b, *o, *o2, *o3;
  a = (T*) malloc (size_bytes);
  b = (T*) malloc (size_bytes);
  o = (T*) malloc (size_bytes);
  o2 = (T*) malloc (size_bytes);
  o3 = (T*) malloc (size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<T> distr (-2.f, 2.f);
  for (int i = 0; i < num_elems; i++) {
    a[i] = distr(g);
    b[i] = distr(g);
  }

  T *d_o;
  d_o = (T*) malloc (size_bytes);

  const int num_threads = omp_get_max_threads();
  const int num_blocks = (nrows + num_threads - 1) / num_threads;
  const int block_size = (nrows + num_blocks - 1) / num_blocks;

  dim3 grid ((nrows + block_size - 1) / block_size);
  dim3 block (block_size);

  #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_blocks; i++) {
    cross_kernel<<<grid, block>>>(nrows, d_o, a, b, 1, 1, 1);
  }
  
  #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_blocks; i++) {
    cross2_kernel<<<grid, block>>>(nrows, d_o, a, b, 1, 1, 1);
  }
  
  #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_blocks; i++) {
    cross3_kernel<<<grid, block>>>(nrows, d_o, a, b);
  }

  #pragma omp barrier

  // Check the results
  bool ok = true;
  for (int i = 0; i < num_elems; i++) {
    if (fabs(o[i] - o2[i]) > 1e-3 || fabs(o[i] - o3[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(a);
  free(b);
  free(o);
  free(o2);
  free(o3);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of rows in a 2D tensor> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  printf("=========== Data type is FP32 ==========\n");
  eval<float>(nrows, repeat);

  printf("=========== Data type is FP64 ==========\n");
  eval<double>(nrows, repeat);

  return 0;
}