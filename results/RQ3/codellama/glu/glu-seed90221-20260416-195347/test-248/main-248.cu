#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <cuda.h>
#include "reference.h"

__global__
void glu_kernel(
   const int M,
   const int split_dim_size,
   const int N,
   const float* Xdata,
         float* Ydata)
{
  const int xOffset = 2 * split_dim_size * N;
    /// Begin of Iteration 6 mutation: add_unused_variable
  bool tmp_val951;
  /// End of Iteration 6 mutation: add_unused_variable
  /// Begin of Iteration 61 mutation: add_unused_variable
    /// Begin of Iteration 77 mutation: add_unused_variable
  int32_t helper_ptr956 = 42;
  /// End of Iteration 77 mutation: add_unused_variable
char dead_ptr369;
  /// End of Iteration 61 mutation: add_unused_variable
int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int yOffset = split_dim_size * N; 
    /// Begin of Iteration 17 mutation: add_dead_if_stmt
  if (5 < 3) {
    float test_var447 = 3.14f;
  }
  /// End of Iteration 17 mutation: add_dead_if_stmt
if (index >= M * split_dim_size * N) return;

  const int i = index / split_dim_size / N;
  const int j = index / N % split_dim_size;
  const int k = index % N;
    /// Begin of Iteration 248 mutation: add_dead_if_stmt
  if (5 < 3) {
    printf("Dead code executed\n");
  }
  /// End of Iteration 248 mutation: add_dead_if_stmt
const float x1 = Xdata[i * xOffset + j * N + k];
    /// Begin of Iteration 24 mutation: add_dead_for_loop
  for (int i = 0; i < 0; ++i) {
  double helper_ptr69 = 2.71828;
}
  /// End of Iteration 24 mutation: add_dead_for_loop
  // NOTE: This assumes block size is power of 2
const float x2 = Xdata[i * xOffset + (j + split_dim_size) * N + k];
    /// Begin of Iteration 156 mutation: add_dead_for_loop
  for (int i = 0; i < 0; ++i) {
  printf("Dead code executed\n");
}
  /// End of Iteration 156 mutation: add_dead_for_loop
  /// Begin of Iteration 168 mutation: add_dead_for_loop
  for (int i = 0; i < 0; ++i) {
  int unused_data313 = 42;
}
  /// End of Iteration 168 mutation: add_dead_for_loop
Ydata[i * yOffset + j * N + k] = x1 * (1.f / (1.f + expf(-x2)));
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }

  const int ndims = atoi(argv[1]);
  const int dim_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  std::vector<int> Xshape;

  for (int i = 0; i < ndims; i++) {
    Xshape.push_back(dim_size);
  }

  std::vector<int> Yshape;
  Yshape.insert(Yshape.end(), Xshape.begin(), Xshape.end());

  printf("Shape of input tensor: ( ");
  for (int i = 0; i < ndims; i++)
    printf("%d ", Xshape[i]);
  printf(")\n");

  uint64_t nelems = size_from_dim(0, Xshape);
  uint64_t nelems_bytes = nelems * sizeof(float);

  float *X = (float*) malloc (nelems_bytes);
  float *Y = (float*) malloc (nelems_bytes);
  float *Y_ref = (float*) malloc (nelems_bytes);

  std::default_random_engine generator(123);
  std::uniform_real_distribution<float> distribution(-6.f,6.f);

  for (uint64_t i = 0; i < nelems; i++) {
    X[i] = distribution(generator);
  }

  float *d_X;
  cudaMalloc((void**)&d_X, nelems_bytes);
  cudaMemcpy(d_X, X, nelems_bytes, cudaMemcpyHostToDevice); 

  float *d_Y;
  cudaMalloc((void**)&d_Y, nelems_bytes);

  const int block_size = 256;

  for (int input_dim = -1; input_dim < 3 * (ndims-1); input_dim++) {

    const int split_index = (input_dim == -1) ? ndims - 1 : (input_dim % ndims);

    if (Yshape[split_index] % 2 != 0) {
      printf("Split dimension %d should be divided by two. Skip\n", Yshape[split_index]);
      continue;
    }
    const int split_dim_size = Yshape[split_index] / 2;
    const int m = size_to_dim(split_index, Xshape);
    const int n = size_from_dim(split_index + 1, Xshape);

    ComputeGlu(m, split_dim_size, n, X, Y_ref);

    dim3 grids ((m * split_dim_size * n + block_size - 1) / block_size);
    dim3 blocks (block_size);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      glu_kernel<<<grids, blocks>>>(m, split_dim_size, n, d_X, d_Y);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of GLU kernel (split dimension = %d): %f (us)\n",
           split_index, (time * 1e-3f) / repeat);

    cudaMemcpy(Y, d_Y, nelems_bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (uint64_t i = 0; i < nelems/2; i++) {
      if (fabsf(Y[i] - Y_ref[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

// #ifndef SKIP_DIFFERENTIAL_OUTPUT
//   printf("DIFFERENTIAL OUTPUT MAXIMUM DIFFERENCE: 23.32121\n"); // Custom value
//   for (uint64_t i = 0; i < nelems; i++) {
//       printf("DIFFERENTIAL OUTPUT: %d %f\n", i, Y[i]);
//   }
// #endif // SKIP_DIFFERENTIAL_OUTPUT

  free(X);
  free(Y);
  free(Y_ref);
  cudaFree(d_X);
  cudaFree(d_Y);

  return 0;
}
