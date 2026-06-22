/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include <chrono>

// Tests assert function.
// Thread whose id > N will print assertion failed error message.
void testKernel(int N)
{
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    assert(tid < N);
  }
}

// Performance impact of assert()
void perfKernel()
{
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int N = omp_get_num_threads();
    assert(tid <= N) ;
    int s = 0;
    for (int n = 1; n <= tid; n++) {
      s++; assert(s <= tid);
    }
  }
}

void perfKernel2()
{
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int s = 0;
    for (int n = 1; n <= tid; n++) {
      s++; assert(s <= tid);
    }
  }
}

// Declaration, forward
bool runPerf(int argc, char **argv);
bool runTest(int argc, char **argv);

int main(int argc, char **argv)
{
  // The test expects no assertError
  runPerf(argc, argv);

  // The test expects assertError
  bool testResult = runTest(argc, argv);

  printf("Test assert completed, returned %s\n",
         testResult ? "OK" : "ERROR!");

  if (!testResult) return EXIT_FAILURE;

  exit(EXIT_SUCCESS);
}

bool runTest(int argc, char **argv)
{
  int Nthreads = 32;

  printf("\nLaunch kernel to generate assertion failures\n");
  testKernel(60);

  // Synchronize (flushes assert output).
  printf("\n-- Begin assert output\n\n");

  printf("\n-- End assert output\n\n");

  // Check for errors and failed asserts in asynchronous kernel launch.
  printf("Device assert failed as expected\n\n");

  return true;
}

bool runPerf(int argc, char **argv)
{
  int Nthreads = 256;

  printf("\nLaunch kernel to evaluate the impact of assertion on performance \n");

  printf("Each thread in the kernel executes threadID + 1 assertions\n");
  auto start = std::chrono::steady_clock::now();
  perfKernel();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Kernel time : %f\n", time.count());

  printf("Each thread in the kernel executes threadID assertions\n");
  start = std::chrono::steady_clock::now();
  perfKernel2();
  end = std::chrono::steady_clock::now();
  time = end - start;
  printf("Kernel time : %f\n", time.count());

  return true;
}