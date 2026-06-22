#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#define BLOCK_SIZE 256

#include "reference.h"

template <typename T>
void BlockRangeAtomicOnGlobalMem(T* data, int n)
{
  #pragma omp target teams distribute num_teams(n/BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++){
    #pragma omp atomic
    data[i]++;  // arbitrary number to add
  }
}

template <typename T>
void WarpRangeAtomicOnGlobalMem(T* data, int n)
{
  #pragma omp target teams distribute num_teams(n/BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++){
    #pragma omp atomic
    data[i&0x1F]++; // arbitrary number to add
  }
}

template <typename T>
void SingleRangeAtomicOnGlobalMem(T* data, int offset, int n)
{
  #pragma omp target teams distribute num_teams(n/BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++){
    #pragma omp atomic
    data[offset]++;    // arbitrary number to add
  }
}

template <typename T>
void BlockRangeAtomicOnSharedMem(T* data, int n)
{
  T smem_data[BLOCK_SIZE];
  #pragma omp target teams distribute num_teams(n/BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++){
    #pragma omp atomic
    smem_data[i%BLOCK_SIZE]++;
  }
  if (omp_get_team_num() == omp_get_num_teams())
    data[omp_get_thread_num()] = smem_data[omp_get_thread_num()];
}

template <typename T>
void WarpRangeAtomicOnSharedMem(T* data, int n)
{
  T smem_data[32];
  #pragma omp target teams distribute num_teams(n/BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++){
    #pragma omp atomic
    smem_data[i&0x1F]++;
  }
  if (omp_get_team_num() == omp_get_num_teams() && omp_get_thread_num() < 0x1F)
    data[omp_get_thread_num()] = smem_data[omp_get_thread_num()];
}

template <typename T>
void SingleRangeAtomicOnSharedMem(T* data, int offset, int n)
{
  T smem_data[BLOCK_SIZE];
  #pragma omp target teams distribute num_teams(n/BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++){
    #pragma omp atomic
    smem_data[offset]++;
  }
  if (omp_get_team_num() == omp_get_num_teams() && omp_get_thread_num() == 0)
    data[omp_get_thread_num()] = smem_data[omp_get_thread_num()];
}

template <typename T>
void atomicPerf (int n, int t, int repeat)
{
  size_t data_size = sizeof(T) * t;

  T* data = (T*) malloc (data_size);
  T* h_data = (T*) malloc (data_size);
  T* r_data = (T*) malloc (data_size);
  int fail;

  for(int i=0; i<t; i++) {
    data[i] = i%1024+1;
  }

  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    BlockRangeAtomicOnGlobalMem<T>(data, n);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  memcpy(h_data, data, data_size);
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    BlockRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");
  
  memcpy(data, h_data, data_size);
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    WarpRangeAtomicOnGlobalMem<T>(data, n);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  memcpy(h_data, data, data_size);
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    WarpRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  memcpy(data, h_data, data_size);
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    SingleRangeAtomicOnGlobalMem<T>(data, i%BLOCK_SIZE, n);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  memcpy(h_data, data, data_size);
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    SingleRangeAtomicOnGlobalMem_ref<T>(r_data, i%BLOCK_SIZE, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  memcpy(data, h_data, data_size);
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    BlockRangeAtomicOnSharedMem<T>(data, n);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  memcpy(data, h_data, data_size);
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    WarpRangeAtomicOnSharedMem<T>(data, n);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  memcpy(data, h_data, data_size);
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    SingleRangeAtomicOnSharedMem<T>(data, i%BLOCK_SIZE, n);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  free(data);
  free(h_data);
  free(r_data);
}

int main(int argc, char* argv[])
{
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