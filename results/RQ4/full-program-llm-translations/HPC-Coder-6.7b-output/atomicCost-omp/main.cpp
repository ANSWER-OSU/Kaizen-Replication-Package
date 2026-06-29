#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <omp.h>

#define BLOCK_SIZE 256

// measure cost of additions without atomics
template <typename T>
void woAtomicOnGlobalMem(T* result, int size)
{
  for ( unsigned int i = 0; i < size; i++){
    result[i] += i % 2;
  }
}

// measure cost of additions with atomics
template <typename T>
void wiAtomicOnGlobalMem(T* result, int size)
{
  #pragma omp parallel shared(result, size)
  {
    T temp;
    #pragma omp for schedule(static)
    for ( int i = 0; i < size; i++ ) {
      temp = i % 2;
      #pragma omp atomic
      result[i] += temp;
    }
  }
}

template <typename T>
void atomicCost(int length, int size, int repeat) {
  printf("\n\n");
  printf("Each thread sums up %d elements\n", size);

  int num_threads = length / size;
  assert(length % size == 0);
  assert(num_threads % omp_get_max_threads() == 0);

  size_t result_size = sizeof(T) * num_threads;

  T* result_wi = (T*)malloc(result_size); 
  T* result_wo = (T*)malloc(result_size); 

  #pragma omp parallel shared(result_wi, result_wo, size, length)
  {
    #pragma omp for
    for(int i = 0; i < num_threads; i++) {
      result_wi[i] = 0;
      result_wo[i] = 0;
    }
  }

  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < repeat; i++){ 
    #pragma omp parallel
    {
      wiAtomicOnGlobalMem(result_wi, size);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();
  for(int i = 0; i < repeat; i++){ 
    #pragma omp parallel
    {
      woAtomicOnGlobalMem(result_wo, size);
    }
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  int found_diff = abs(result_wi[0] - result_wo[0]) != 0;
  for (int i = 1; i < num_threads; i++) {
    found_diff |= abs(result_wi[i] - result_wo[i]) != 0;
  }

  printf("%s\n", found_diff ? "FAIL" : "PASS");
  
  free(result_wi);
  free(result_wo);
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
  assert(length % omp_get_max_threads() == 0);

  printf("\nINT32 atomic add\n");
  atomicCost<int>(length, nelems, repeat);

  printf("\nFP32 atomic add\n");
  atomicCost<float>(length, nelems, repeat);

  return 0;
}