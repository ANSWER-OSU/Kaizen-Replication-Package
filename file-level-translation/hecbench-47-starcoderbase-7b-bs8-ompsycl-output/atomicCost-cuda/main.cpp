#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

static void CheckError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
  }
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))

#define BLOCK_SIZE 256

// measure cost of additions without atomics
template <typename T>
void woAtomicOnGlobalMem(const int numTeams, const int numThreads, T* result, int size) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for ( unsigned int i = 0; i < numTeams; i++ ) {
    for ( unsigned int j = i * size; j < (i + 1) * size; j++){
      result[i] += j % 2;
    }
  }
}

// measure cost of additions with atomics
template <typename T>
void wiAtomicOnGlobalMem(const int numTeams, const int numThreads, T* result, int size) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for ( unsigned int i = 0; i < numTeams; i++ ) {
    #pragma omp atomic update
    result[i] += size * i % 2;
  }
}

template <typename T>
void atomicCost (int length, int size, int repeat) {
  printf("\n\n");
  printf("Each thread sums up %d elements\n", size);

  int numTeams = length / size;
  int numThreads = numTeams * BLOCK_SIZE;
  assert(length % size == 0);
  assert(numTeams % BLOCK_SIZE == 0);

  size_t result_size = sizeof(T) * numTeams;

  T* result_wi = (T*) malloc (result_size);
  T* result_wo = (T*) malloc (result_size);

  T *d_result_wi, *d_result_wo;
  CHECK_ERROR( cudaMalloc((void **)&d_result_wi, result_size) );
  CHECK_ERROR( cudaMemset(d_result_wi, 0, result_size) );
  CHECK_ERROR( cudaMalloc((void **)&d_result_wo, result_size) );
  CHECK_ERROR( cudaMemset(d_result_wo, 0, result_size) );

  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(BLOCK_SIZE)
  for ( unsigned int i = 0; i < numTeams; i++ ) {
    result_wi[i] = 0;
    result_wo[i] = 0;
  }

  wiAtomicOnGlobalMem<T>(numTeams, BLOCK_SIZE, d_result_wi, size);
  woAtomicOnGlobalMem<T>(numTeams, BLOCK_SIZE, d_result_wo, size);

  CHECK_ERROR( cudaDeviceSynchronize() );
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++) {
    wiAtomicOnGlobalMem<T>(numTeams, BLOCK_SIZE, d_result_wi, size);
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
         time * 1e-3f / repeat);
  CHECK_ERROR( cudaMemcpy(result_wi, d_result_wi, result_size, cudaMemcpyDeviceToHost) );

  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++) {
    woAtomicOnGlobalMem<T>(numTeams, BLOCK_SIZE, d_result_wo, size);
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
         time * 1e-3f / repeat);
  CHECK_ERROR( cudaMemcpy(result_wo, d_result_wo, result_size, cudaMemcpyDeviceToHost) );

  int diff = memcmp(result_wi, result_wo, result_size);
  printf("%s\n", diff ? "FAIL" : "PASS");

  free(result_wi);
  free(result_wo);
  cudaFree(d_result_wi);
  cudaFree(d_result_wo);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: the number of elements to sum per thread (1 - 16)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int length = 922521600;
  assert(length % BLOCK_SIZE == 0);

  printf("\nFP64 atomic add\n");
  atomicCost<double>(length, nelems, repeat);

  printf("\nINT32 atomic add\n");
  atomicCost<int>(length, nelems, repeat);

  printf("\nFP32 atomic add\n");
  atomicCost<float>(length, nelems, repeat);

  return 0;
}