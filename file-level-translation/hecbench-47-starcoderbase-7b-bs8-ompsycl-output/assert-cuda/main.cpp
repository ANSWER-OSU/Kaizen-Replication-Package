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
void testKernel(const int numTeams, const int numThreads, int N) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int gid = 0; gid < N; gid++) {
    assert(gid < N);
  }
}

// Performance impact of assert()
void perfKernel(const int numTeams, const int numThreads) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int gid = 0; gid < 32 * 1000; gid++) {
    assert(gid <= 32 * 1000);
    int s = 0;
    for (int n = 1; n <= gid; n++) {
      s++;
      assert(s <= gid);
    }
  }
}

void perfKernel2(const int numTeams, const int numThreads) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int gid = 0; gid < 32 * 1000; gid++) {
    int s = 0;
    for (int n = 1; n <= gid; n++) {
      s++;
      assert(s <= gid);
    }
  }
}

// Declaration, forward
bool runPerf(int argc, char **argv);
bool runTest(int argc, char **argv);

int main(int argc, char **argv) {
  // The test expects no assertError
  runPerf(argc, argv);

  // The test expects assertError
  bool testResult = runTest(argc, argv);

  printf("Test assert completed, returned %s\n",
         testResult ? "OK" : "ERROR!");

  if (!testResult) return EXIT_FAILURE;

  exit(EXIT_SUCCESS);
}

bool runTest(int argc, char **argv) {
  int numTeams = 2;
  int numThreads = 32;
  int N = 60;

  testKernel(numTeams, numThreads, N);

  // Synchronize (flushes assert output)
  #pragma omp target teams distribute parallel for num_teams(2) num_threads(32)
  for (int i = 0; i < 2; i++) {
    printf("\n-- Begin assert output %d\n\n", i);
  }
  #pragma omp target teams distribute parallel for num_teams(2) num_threads(32)
  for (int i = 0; i < 2; i++) {
    printf("-- End assert output %d\n\n", i);
  }

  // Check for errors and failed asserts in asynchronous kernel launch
  int err = 0;
  #pragma omp target teams distribute parallel for num_teams(2) num_threads(32) reduction(+:err)
  for (int i = 0; i < 2; i++) {
    int tmp = omp_get_num_teams();
    if (tmp > 2) {
      err++;
    }
  }
  if (err > 0) {
    printf("Number of teams error expected, but saw %d\n", err);
    return false;
  }

  err = 0;
  #pragma omp target teams distribute parallel for num_teams(2) num_threads(32) reduction(+:err)
  for (int i = 0; i < 2; i++) {
    int tmp = omp_get_num_threads();
    if (tmp > 32) {
      err++;
    }
  }
  if (err > 0) {
    printf("Number of teams error expected, but saw %d\n", err);
    return false;
  }

  #pragma omp target teams distribute parallel for num_teams(2) num_threads(32)
  for (int i = 0; i < 2; i++) {
    int tmp = omp_get_team_num();
    if (tmp > 1) {
      err++;
    }
  }
  if (err > 0) {
    printf("Team number error expected, but saw %d\n", err);
    return false;
  }

  err = 0;
  #pragma omp target teams distribute parallel for num_teams(2) num_threads(32) reduction(+:err)
  for (int i = 0; i < 2; i++) {
    int tmp = omp_get_thread_num();
    if (tmp > 31) {
      err++;
    }
  }
  if (err > 0) {
    printf("Thread number error expected, but saw %d\n", err);
    return false;
  }

  return true;
}

bool runPerf(int argc, char **argv) {
  int numTeams = 1000;
  int numThreads = 256;

  perfKernel(numTeams, numThreads);
  perfKernel2(numTeams, numThreads);

  return true;
}