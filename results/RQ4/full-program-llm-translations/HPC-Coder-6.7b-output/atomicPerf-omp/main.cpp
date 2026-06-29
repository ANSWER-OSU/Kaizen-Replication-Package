#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define BLOCK_SIZE 256

#include "reference.h"


template <typename T>
void BlockRangeAtomicOnGlobalMem(T* data, int n)
{
  #pragma omp parallel
  {
    unsigned int tid = omp_get_thread_num();
    for ( unsigned int i = tid; i < n; i += BLOCK_SIZE){
        atomicAdd(&data[tid], (T)1);
    }
  }
}

template <typename T>
void WarpRangeAtomicOnGlobalMem(T* data, int n)
{
  #pragma omp parallel
  {
    unsigned int tid = omp_get_thread_num();
    for ( unsigned int i = tid; i < n; i += 32){
        atomicAdd(&data[i & 31], (T)1);
    }
  }
}

template <typename T>
void SingleRangeAtomicOnGlobalMem(T* data, int offset, int n)
{
  #pragma omp parallel
  {
    unsigned int tid = omp_get_thread_num();
    for ( unsigned int i = tid; i < n; i += BLOCK_SIZE){
        atomicAdd(&data[offset], (T)1);
    }
  }
}

template <typename T>
void BlockRangeAtomicOnSharedMem(T* data, int n)
{
  #pragma omp parallel
  {
    unsigned int tid = omp_get_thread_num();
    T smem_data[BLOCK_SIZE];
    for ( unsigned int i = tid; i < n; i += BLOCK_SIZE){
        atomicAdd(&smem_data[tid], (T)1);
    }
    if (omp_get_num_threads() == 1) {
        data[tid] = smem_data[tid];
    }
  }
}

template <typename T>
void WarpRangeAtomicOnSharedMem(T* data, int n)
{
  #pragma omp parallel
  {
    unsigned int tid = omp_get_thread_num();
    T smem_data[32];
    for ( unsigned int i = tid; i < n; i += 32){
        atomicAdd(&smem_data[i & 31], (T)1);
    }
    if (omp_get_num_threads() == 1 && tid < 32) {
        data[tid] = smem_data[tid];
    }
  }
}

template <typename T>
void SingleRangeAtomicOnSharedMem(T* data, int offset, int n)
{
  #pragma omp parallel
  {
    unsigned int tid = omp_get_thread_num();
    T smem_data[BLOCK_SIZE];
    for ( unsigned int i = tid; i < n; i += BLOCK_SIZE){
        atomicAdd(&smem_data[offset], (T)1);
    }
    if (omp_get_num_threads() == 1 && tid == 0) {
        data[tid] = smem_data[tid];
    }
  }
}

template <typename T>
void atomicPerf (int n, int t, int repeat)
{
  std::vector<T> data(t);
  std::vector<T> h_data(t);
  std::vector<T> r_data(t);
  for(int i=0; i<t; i++) {
    data[i] = (T) (i%1024+1);
  }

  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++) {
    BlockRangeAtomicOnGlobalMem(data.data(), n);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  memcpy(h_data.data(), data.data(), t*sizeof(T));
  for(int i=0; i<repeat; i++)
    BlockRangeAtomicOnGlobalMem_ref(r_data.data(), n);
  int fail = memcmp(h_data.data(), r_data.data(), t*sizeof(T));
  printf("%s\n", fail ? "FAIL" : "PASS");

  for(int i=0; i<repeat; i++) {
    WarpRangeAtomicOnGlobalMem(data.data(), n);
  }
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  memcpy(h_data.data(), data.data(), t*sizeof(T));
  for(int i=0; i<repeat; i++)
    WarpRangeAtomicOnGlobalMem_ref(r_data.data(), n);
  fail = memcmp(h_data.data(), r_data.data(), t*sizeof(T));
  printf("%s\n", fail ? "FAIL" : "PASS");

  for(int i=0; i<repeat; i++) {
    SingleRangeAtomicOnGlobalMem(data.data(), i % BLOCK_SIZE, n);
  }
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  memcpy(h_data.data(), data.data(), t*sizeof(T));
  for(int i=0; i<repeat; i++)
    SingleRangeAtomicOnGlobalMem_ref(r_data.data(), i % BLOCK_SIZE, n);
  fail = memcmp(h_data.data(), r_data.data(), t*sizeof(T));
  printf("%s\n", fail ? "FAIL" : "PASS");

  for(int i=0; i<repeat; i++) {
    BlockRangeAtomicOnSharedMem(data.data(), n);
  }
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  fail = memcmp(data.data(), h_data.data(), t*sizeof(T));
  printf("%s\n", fail ? "FAIL" : "PASS");

  for(int i=0; i<repeat; i++) {
    WarpRangeAtomicOnSharedMem(data.data(), n);
  }
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  fail = memcmp(data.data(), h_data.data(), t*sizeof(T));
  printf("%s\n", fail ? "FAIL" : "PASS");

  for(int i=0; i<repeat; i++) {
    SingleRangeAtomicOnSharedMem(data.data(), i % BLOCK_SIZE, n);
  }
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  fail = memcmp(data.data(), h_data.data(), t*sizeof(T));
  printf("%s\n", fail ? "FAIL" : "PASS");
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int n = 3*4*7*8*9*256; // number of threads
  const int len = 1024; // data array length
  
  printf("\nFP64 atomic add\n");
  atomicPerf<double>(n, len, repeat); 

  printf("\nINT32 atomic add\n");
  atomicPerf<int>(n, len, repeat); 

  printf("\nFP32 atomic add\n");
  atomicPerf<float>(n, len, repeat); 

  return 0;
}