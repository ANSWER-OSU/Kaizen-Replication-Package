//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <chrono>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "Projectile.hpp"
#include "reference.h"

static const int num_elements = 10000000;
const int BLOCK_SIZE = 256;

// Function to calculate the range, maximum height and total flight time of a
// projectile
void CalculateRange(const int numTeams, const int numThreads, const Projectile *obj, Projectile *pObj) {
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < num_elements; i++) {
    float proj_angle = obj[i].getangle();
    float proj_vel = obj[i].getvelocity();
    float sin_value = sinf(proj_angle * kPIValue / 180.0f);
    float cos_value = cosf(proj_angle * kPIValue / 180.0f);
    float total_time = fabsf((2 * proj_vel * sin_value)) / kGValue;
    float max_range =  fabsf(proj_vel * total_time * cos_value);
    float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f * kGValue;  // h = v^2 * sin^2theta/2g

    pObj[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
  }
}

// in_vect and out_vect are the vectors with N Projectile numbers and are inputs to the
// parallel function
void GpuParallel(std::vector<Projectile>& in_vect,
                 std::vector<Projectile>& out_vect,
                 const int numTeams, const int numThreads, const int repeat)
{
  Projectile *bufin_vect, *bufout_vect;

  cudaMalloc((void**)&bufin_vect, sizeof(Projectile) * num_elements);
  cudaMalloc((void**)&bufout_vect, sizeof(Projectile) * num_elements);
  cudaMemcpy(bufin_vect, in_vect.data(), sizeof(Projectile) * num_elements, cudaMemcpyHostToDevice);

  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < repeat; i++)
    CalculateRange(numTeams, numThreads, bufin_vect, bufout_vect);

  cudaMemcpy(out_vect.data(), bufout_vect, sizeof(Projectile) * num_elements, cudaMemcpyDeviceToHost);
  cudaFree(bufin_vect);
  cudaFree(bufout_vect);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <# teams> <# threads> <repeat>\n", argv[0]);
    return 1;
  }
  const int numTeams = atoi(argv[1]);
  const int numThreads = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  float init_angle = 0.0f;
  float init_vel = 0.0f;
  vector<Projectile> input_vect, out_parallel_vect, out_scalar_vect;

  // Initialize the Input and Output vectors
  srand(2);
  for (int i = 0; i < num_elements; i++) {
    init_angle = rand() % 90 + 10;
    init_vel = rand() % 400 + 10;
    input_vect.push_back(Projectile(init_angle, init_vel, 1.0f, 1.0f, 1.0f));
    out_parallel_vect.push_back(Projectile());
    out_scalar_vect.push_back(Projectile());
  }

  GpuParallel(input_vect, out_parallel_vect, numTeams, numThreads, repeat);

  reference(input_vect.data(), out_scalar_vect.data(), num_elements);

  bool ok = true;
  for (int i = 0; i < num_elements; i++) {
    if (out_parallel_vect[i] != out_scalar_vect[i]) {
      ok = false;
      std::cout << out_parallel_vect[i] << std::endl;
      std::cout << out_scalar_vect[i] << std::endl;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}