#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

#define BLOCK_SIZE 256

#include "reference.h"

template <typename T>
void BlockRangeAtomicOnGlobalMem(T* data, int n)
{
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    #pragma omp atomic
    data[threadIdx.x] += (T)1;  //arbitrary number to add
  }
}

template <typename T>
void WarpRangeAtomicOnGlobalMem(T* data, int n)
{
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    #pragma omp atomic
    data[i & 0x1F] += (T)1; //arbitrary number to add
  }
}

template <typename T>
void SingleRangeAtomicOnGlobalMem(T* data, int offset, int n)
{
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    #pragma omp atomic
    data[offset] += (T)1;    //arbitrary number to add
  }
}

template <typename T>
void BlockRangeAtomicOnSharedMem(T* data, int n)
{
  __shared__ T smem_data[BLOCK_SIZE];
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    #pragma omp atomic
    smem_data[threadIdx.x] += (T)1;
  }
  if (blockIdx.x == gridDim.x)
    data[threadIdx.x] = smem_data[threadIdx.x];
}

template <typename T>
void WarpRangeAtomicOnSharedMem(T* data, int n)
{
  __shared__ T smem_data[32];
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    #pragma omp atomic
    smem_data[(i & 0x1F)] += (T)1;
  }
  if (blockIdx.x == gridDim.x && threadIdx.x < 0x1F)
    data[threadIdx.x] = smem_data[threadIdx.x];
}

template <typename T>
void SingleRangeAtomicOnSharedMem(T* data, int offset, int n)
{
  __shared__ T smem_data[BLOCK_SIZE];
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    #pragma omp atomic
    smem_data [offset] += (T)1;
  }
  if (blockIdx.x == gridDim.x && threadIdx.x == 0)
    data[threadIdx.x] = smem_data[threadIdx.x];
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

  T* d_data;
  d_data = (T*) malloc (data_size);

  dim3 block (BLOCK_SIZE);
  dim3 grid (n / BLOCK_SIZE);

  memcpy(d_data, data, data_size);
  #pragma omp parallel for num_threads(32)
  for(int i=0; i<repeat; i++)
    BlockRangeAtomicOnGlobalMem<T>(d_data, n);
  memcpy(h_data, d_data, data_size);
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    BlockRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");
  
  memcpy(d_data, data, data_size);
  #pragma omp parallel for num_threads(32)
  for(int i=0; i<repeat; i++)
    WarpRangeAtomicOnGlobalMem<T>(d_data, n);
  memcpy(h_data, d_data, data_size);
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    WarpRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  memcpy(d_data, data, data_size);
  #pragma omp parallel for num_threads(32)
  for(int i=0; i<repeat; i++)
    SingleRangeAtomicOnGlobalMem<T>(d_data, i % BLOCK_SIZE, n);
  memcpy(h_data, d_data, data_size);
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    SingleRangeAtomicOnGlobalMem_ref<T>(r_data, i % BLOCK_SIZE, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  memcpy(d_data, data, data_size);
  #pragma omp parallel for num_threads(32)
  for(int i=0; i<repeat; i++)
    BlockRangeAtomicOnSharedMem<T>(d_data, n);
  fail = memcmp(d_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  memcpy(d_data, data, data_size);
  #pragma omp parallel for num_threads(32)
  for(int i=0; i<repeat; i++)
    WarpRangeAtomicOnSharedMem<T>(d_data, n);
  fail = memcmp(d_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  memcpy(d_data, data, data_size);
  #pragma omp parallel for num_threads(32)
  for(int i=0; i<repeat; i++)
    SingleRangeAtomicOnSharedMem<T>(d_data, i % BLOCK_SIZE, n);
  fail = memcmp(d_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  free(data);
  free(h_data);
  free(r_data);
  free(d_data); 
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