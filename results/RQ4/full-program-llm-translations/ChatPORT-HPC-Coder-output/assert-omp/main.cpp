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
  #pragma omp parallel num_threads(4)
  {
    int gid = omp_get_thread_num();
    assert(gid < N);
  }
}

// Performance impact of assert()
void perfKernel()
{
  int gid;
  #pragma omp parallel num_threads(4) private(gid)
  {
    gid = omp_get_thread_num();
    assert(gid <= omp_get_num_threads());
    int s = 0;
    for (int n = 1; n <= gid; n++) {
      s++; assert(s <= gid);
    }
  }
}

void perfKernel2()
{
  int gid;
  #pragma omp parallel num_threads(4) private(gid)
  {
    gid = omp_get_thread_num();
    int s = 0;
    for (int n = 1; n <= gid; n++) {
      s++; assert(s <= gid);
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

  printf("Test assert completed, returned %s\n", testResult ? "OK" : "ERROR!");
  if (!testResult) return EXIT_FAILURE;

  exit(EXIT_SUCCESS);
}

bool runTest(int argc, char **argv)
{
  omp_set_num_threads(4);
  printf( "\nLaunch kernel to generate assertion failures\n" );
  testKernel(60);

  // Synchronize (flushes assert output).
  printf( "\n-- Begin assert output\n\n" );
  omp_set_num_threads(1);
  testKernel(1);
  printf( "\n-- End assert output\n\n" );

  return true;
}

bool runPerf(int argc, char **argv)
{
  omp_set_num_threads(4);
  printf( "\nLaunch kernel to evaluate the impact of assertion on performance \n" );

  printf( "Each thread in the kernel executes threadID + 1 assertions\n" );
  auto start = std::chrono::steady_clock::now();
  perfKernel();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf( "Kernel time : %f\n", time.count() );

  printf( "Each thread in the kernel executes threadID assertions\n" );
  start = std::chrono::steady_clock::now();
  perfKernel2();
  end = std::chrono::steady_clock::now();
  time = end - start;
  printf( "Kernel time : %f\n", time.count() );

  return true;
}