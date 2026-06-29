#include <chrono>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "Projectile.hpp"
#include "reference.h"

static const int BLOCK_SIZE = 256;

void GpuParallel(std::vector<Projectile>& in_vect,
                 std::vector<Projectile>& out_vect,
                 const int repeat)
{
  #pragma omp parallel for
  for (int i = 0; i < in_vect.size(); i++) {
    float proj_angle = in_vect[i].getangle();
    float proj_vel = in_vect[i].getvelocity();
    float sin_value = sinf(proj_angle * kPIValue / 180.0f);
    float cos_value = cosf(proj_angle * kPIValue / 180.0f);
    float total_time = fabsf((2 * proj_vel * sin_value)) / kGValue;
    float max_range =  fabsf(proj_vel * total_time * cos_value);
    float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                       kGValue;  // h = v^2 * sin^2theta/2g

    out_vect[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
  }
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
  for (int i = 0; i < in_vect.size(); i++) {
    init_angle = rand() % 90 + 10;
    init_vel = rand() % 400 + 10;
    in_vect.push_back(Projectile(init_angle, init_vel, 1.0f, 1.0f, 1.0f));
    out_parallel_vect.push_back(Projectile());
    out_scalar_vect.push_back(Projectile());
  }

  GpuParallel(in_vect, out_parallel_vect, repeat);

  reference(in_vect.data(), out_scalar_vect.data(), in_vect.size());

  bool ok = true;
  for (int i = 0; i < in_vect.size(); i++) {
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