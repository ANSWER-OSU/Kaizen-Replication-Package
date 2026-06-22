#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <iostream>
#include <vector>
#include <omp.h>
#include "Projectile.hpp"
#include "reference.h"

static const int num_elements = 10000000;
const int BLOCK_SIZE = 256;

static std::mutex my_mutex;

// Function to calculate the range, maximum height and total flight time of a
// projectile
void CalculateRange(const Projectile *obj, Projectile *pObj) {
  int tid = omp_get_thread_num();
  int num_threads = omp_get_num_threads();
  int i;
  for (i = tid; i < num_elements; i += num_threads) {
      float proj_angle = obj[i].getangle();
      float proj_vel = obj[i].getvelocity();
      float sin_value = sinf(proj_angle * kPIValue / 180.0f);
      float cos_value = cosf(proj_angle * kPIValue / 180.0f);
      float total_time = fabsf((2 * proj_vel * sin_value)) / kGValue;
      float max_range =  fabsf(proj_vel * total_time * cos_value);
      float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                    kGValue;
      pObj[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
  }
}

// in_vect and out_vect are the vectors with N Projectile numbers and are inputs to the
// parallel function
void OpenMPParallel(std::vector<Projectile>& in_vect,
               std::vector<Projectile>& out_vect,
               const int repeat)
{
  Projectile *bufin_vect, *bufout_vect;
  bufin_vect = static_cast<Projectile*>(malloc(sizeof(Projectile) * num_elements));
  bufout_vect = static_cast<Projectile*>(malloc(sizeof(Projectile) * num_elements));
  memcpy(bufin_vect, in_vect.data(), sizeof(Projectile) * num_elements);

  omp_set_num_threads(atoi(argv[1]));
  auto start = std::chrono::steady_clock::now();

  #pragma omp parallel
  {
    CalculateRange(bufin_vect, bufout_vect);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  memcpy(out_vect.data(), bufout_vect, sizeof(Projectile) * num_elements);
  free(bufin_vect);
  free(bufout_vect);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

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

  OpenMPParallel(input_vect, out_parallel_vect, repeat);

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