#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"

void MRCGradient (
    const int numTeams, const int numThreads,
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float*__restrict__ dX1, float*__restrict__ dX2)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < N; i++) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) {
      dX1[i] = dX2[i] = 0.f;
    } else {
      dX1[i] = -Y[i] * dOutput[i];
      dX2[i] = Y[i] * dOutput[i];
    }
  }
}

void MRCGradient2(
    const int numTeams, const int numThreads,
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float*__restrict__ dX1, float*__restrict__ dX2)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < N; i++) {
    float y = Y[i];
    float o = dOutput[i];
    float dist = -y * (X1[i] - X2[i]) + margin;
    dX1[i] = dist < 0.f ? 0.f : -y * o;
    dX2[i] = dist < 0.f ? 0.f : y * o;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  size_t size_bytes = length * sizeof(float);

  float *h_X1  = (float*) malloc (size_bytes);
  float *h_X2  = (float*) malloc (size_bytes);
  float *h_O   = (float*) malloc (size_bytes);
    int *h_Y   = (  int*) malloc (size_bytes);
  float *h_dX1 = (float*) malloc (size_bytes);
  float *h_dX2 = (float*) malloc (size_bytes);
  float *r_dX1 = (float*) malloc (size_bytes);
  float *r_dX2 = (float*) malloc (size_bytes);

  const float m = 0.01;  // margin

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);
  for (int i = 0; i < length; i++) {
    h_X1[i] = distr(g);
    h_X2[i] = distr(g);
    h_O[i] = distr(g);
    h_Y[i] = (distr(g) < 0) ? -1 : 1;
  }

  int numTeams, numThreads;
  #pragma omp parallel
  {
    numTeams  = omp_get_num_teams();
    numThreads = omp_get_num_threads();
  }

  float *d_X1, *d_X2, *d_O, *d_dX1, *d_dX2;
  int *d_Y;
  #pragma omp target teams distribute num_teams(numTeams)
  d_X1 = (float*) malloc (size_bytes, omp_target_allocator);
  #pragma omp target teams distribute num_teams(numTeams)
  d_X2 = (float*) malloc (size_bytes, omp_target_allocator);
  #pragma omp target teams distribute num_teams(numTeams)
  d_O = (float*) malloc (size_bytes, omp_target_allocator);
  #pragma omp target teams distribute num_teams(numTeams)
  d_Y = (  int*) malloc (size_bytes, omp_target_allocator);
  #pragma omp target teams distribute num_teams(numTeams)
  d_dX1 = (float*) malloc (size_bytes, omp_target_allocator);
  #pragma omp target teams distribute num_teams(numTeams)
  d_dX2 = (float*) malloc (size_bytes, omp_target_allocator);

  #pragma omp target update to (device)
  for (int i = 0; i < length; i++) {
    d_X1[i] = h_X1[i];
    d_X2[i] = h_X2[i];
    d_O[i] = h_O[i];
    d_Y[i] = h_Y[i];
  }

  // warmup
  for (int i = 0; i < repeat; i++) {
    MRCGradient (numTeams, numThreads, length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
    MRCGradient2 (numTeams, numThreads, length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) 
    MRCGradient (numTeams, numThreads, length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of MRC kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) 
    MRCGradient2 (numTeams, numThreads, length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of MRC2 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  // verify
  #pragma omp target update from (device)
  for (int i = 0; i < length; i++) {
    h_dX1[i] = d_dX1[i];
    h_dX2[i] = d_dX2[i];
  }

  reference (length, h_Y, h_X1, h_X2, h_O, m, r_dX1, r_dX2);

  bool ok = true;
  for (int i = 0; i < length; i++) {
    if (fabs(h_dX1[i] - r_dX1[i]) > 1e-3 || fabs(h_dX2[i] - r_dX2[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  #pragma omp target exit data delete ( d_X1 : length )
  #pragma omp target exit data delete ( d_X2 : length )
  #pragma omp target exit data delete ( d_O : length )
  #pragma omp target exit data delete ( d_Y : length )
  #pragma omp target exit data delete ( d_dX1 : length )
  #pragma omp target exit data delete ( d_dX2 : length )

  free(h_X1);
  free(h_X2);
  free(h_O);
  free(h_Y);
  free(h_dX1);
  free(h_dX2);

  return 0;
}