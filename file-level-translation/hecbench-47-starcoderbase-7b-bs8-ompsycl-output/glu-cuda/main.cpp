#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <omp.h>

void ComputeGlu( const int numTeams, const int numThreads, const int M, const int split_dim_size, const int N, const float* Xdata, float* Ydata) {
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < split_dim_size; j++) {
      float x1 = 0, x2 = 0;
      #pragma omp parallel for reduction (+ : x1, x2) num_threads(numThreads)
      for (int k = 0; k < N; k++) {
        int index = i * split_dim_size * N + j * N + k;
        x1 += Xdata[index];
        x2 += Xdata[index + split_dim_size * N];
      }
      Ydata[i * split_dim_size * N + j * N] = x1 * (1.f / (1.f + expf(-x2)));
    }
  }
}

int main(int argc, char* argv[]) {
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
    const int numTeams = (M * split_dim_size * N + block_size - 1) / block_size;
    const int numThreads = block_size;

    ComputeGlu(numTeams, numThreads, M, split_dim_size, N, X, Y_ref);

    #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < split_dim_size; j++) {
        float x1 = 0, x2 = 0;
        #pragma omp parallel for reduction (+ : x1, x2) num_threads(numThreads)
        for (int k = 0; k < N; k++) {
          int index = i * split_dim_size * N + j * N + k;
          x1 += X[index];
          x2 += X[index + split_dim_size * N];
        }
        Y[i * split_dim_size * N + j * N] = x1 * (1.f / (1.f + expf(-x2)));
      }
    }

    bool ok = true;
    for (uint64_t i = 0; i < nelems/2; i++) {
      if (fabsf(Y[i] - Y_ref[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  free(X);
  free(Y);
  free(Y_ref);
  cudaFree(d_X);
  cudaFree(d_Y);

  return 0;
}