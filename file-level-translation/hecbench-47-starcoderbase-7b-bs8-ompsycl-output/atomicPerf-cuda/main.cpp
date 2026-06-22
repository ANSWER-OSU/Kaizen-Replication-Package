#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

#define BLOCK_SIZE 256

#include "reference.h"

static void CheckError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
  }
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))


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
  CHECK_ERROR( cudaMalloc((void **)&d_data, data_size) );

  #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
  for ( unsigned int i = 0; i < t; i++ ) {
    #pragma omp atomic update
    d_data[i] += (T)1;
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
    for ( unsigned int i = 0; i < t; i++ ) {
      #pragma omp atomic update
      d_data[i] += (T)1;
    }
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost) );
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
    for ( unsigned int i = 0; i < t; i++ ) {
      r_data[i] += (T)1;
    }
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");
  
  CHECK_ERROR( cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
    for ( unsigned int i = 0; i < t; i++ ) {
      d_data[i] += (T)1;
    }
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost) );
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
    for ( unsigned int i = 0; i < t; i++ ) {
      r_data[i] += (T)1;
    }
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  CHECK_ERROR( cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
    for ( unsigned int i = 0; i < t; i++ ) {
      #pragma omp atomic update
      d_data[i%BLOCK_SIZE] += (T)1;
    }
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost) );
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  CHECK_ERROR( cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
    for ( unsigned int i = 0; i < t; i++ ) {
      #pragma omp atomic update
      d_data[i] += (T)1;
    }
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost) );
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  CHECK_ERROR( cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
    for ( unsigned int i = 0; i < t; i++ ) {
      #pragma omp atomic update
      d_data[i] += (T)1;
    }
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost) );
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  CHECK_ERROR( cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    #pragma omp target teams distribute parallel for num_teams(n/BLOCK_SIZE) num_threads(BLOCK_SIZE)
    for ( unsigned int i = 0; i < t; i++ ) {
      #pragma omp atomic update
      d_data[i%BLOCK_SIZE] += (T)1;
    }
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost) );
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  free(data);
  free(h_data);
  free(r_data);
  cudaFree(d_data); 
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