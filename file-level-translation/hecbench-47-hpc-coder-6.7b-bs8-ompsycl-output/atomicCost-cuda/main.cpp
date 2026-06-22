#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

#define BLOCK_SIZE 256

// measure cost of additions without atomics
template <typename T>
void woAtomicOnGlobalMem(T* result, int size, const int numTeams, const int numThreads)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for ( int i = 0; i < numTeams; i++ ) {
    #pragma omp parallel for num_threads(numThreads)
    for ( int j = 0; j < numThreads; j++ ) {
      int tid = i * numThreads + j;
      for ( unsigned int k = tid * size; k < (tid + 1) * size; k++){
        result[tid] += k % 2;
      }
    }
  }
}

// measure cost of additions with atomics
template <typename T>
void wiAtomicOnGlobalMem(T* result, int size, const int numTeams, const int numThreads)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for ( int i = 0; i < numTeams; i++ ) {
    #pragma omp parallel for num_threads(numThreads)
    for ( int j = 0; j < numThreads; j++ ) {
      int tid = i * numThreads + j;
      for ( unsigned int k = tid * size; k < (tid + 1) * size; k++){
        #pragma omp atomic
        result[tid] += k % 2;
      }
    }
  }
}

template <typename T>
void atomicCost (int length, int size, int repeat, int numThreads)
{
  printf("\n\n");
  printf("Each thread sums up %d elements\n", size);

  int numTeams = length / size;
  assert(length % size == 0);
  assert(numTeams % numThreads == 0);

  size_t result_size = sizeof(T) * numTeams;

  T* result_wi = (T*) malloc (result_size);
  T* result_wo = (T*) malloc (result_size);
  memset(result_wi, 0, result_size);
  memset(result_wo, 0, result_size);

  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    wiAtomicOnGlobalMem<T>(result_wi, size, numTeams, numThreads);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    woAtomicOnGlobalMem<T>(result_wo, size, numTeams, numThreads);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  int diff = memcmp(result_wi, result_wo, result_size);
  printf("%s\n", diff ? "FAIL" : "PASS");

  free(result_wi);
  free(result_wo);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <N> <repeat> <numThreads>\n", argv[0]);
    printf("N: the number of elements to sum per thread (1 - 16)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int numThreads = atoi(argv[3]);

  const int length = 922521600;
  assert(length % BLOCK_SIZE == 0);

  printf("\nFP64 atomic add\n");
  atomicCost<double>(length, nelems, repeat, numThreads);

  printf("\nINT32 atomic add\n");
  atomicCost<int>(length, nelems, repeat, numThreads);

  printf("\nFP32 atomic add\n");
  atomicCost<float>(length, nelems, repeat, numThreads);

  return 0;
}