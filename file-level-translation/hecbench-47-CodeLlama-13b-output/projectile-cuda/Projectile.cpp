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

void CalculateRange(const Projectile &obj, Projectile &pObj) {

  int i = omp_get_thread_num();
  if (i >= num_elements) return;
  float proj_angle = obj.getangle();
  float proj_vel = obj.getvelocity();
  float sin_value = sinf(proj_angle * kPIValue / 180.0f);
  float cos_value = cosf(proj_angle * kPIValue / 180.0f);
  float total_time = fabsf((2 * proj_vel * sin_value)) / kGValue;
  float max_range =  fabsf(proj_vel * total_time * cos_value);
  float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                     kGValue;  // h = v^2 * sin^2theta/2g

  pObj.setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
}

// in_vect and out_vect are the vectors with N Projectile numbers and are inputs to the
// parallel function
void GpuParallel(std::vector<Projectile>& in_vect,
                 std::vector<Projectile>& out_vect,
                 const int repeat)
{
  std::vector<Projectile> bufin_vect, bufout_vect;

  bufin_vect.resize(num_elements);
  bufout_vect.resize(num_elements);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_elements; i++)
    CalculateRange(in_vect[i], bufout_vect[i]);

  for (int i = 0; i < repeat; i++)
    CalculateRange(bufin_vect, bufout_vect);

  reference(bufin_vect, bufout_vect, num_elements);

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

  GpuParallel(input_vect, out_parallel_vect, repeat);

  for (int i = 0; i < num_elements; i++) {
    if (out_parallel_vect[i] != out_scalar_vect[i]) {
       std::cout << out_parallel_vect[i] << std::endl;
       std::cout << out_scalar_vect[i] << std::endl;
       break;
    }
  }
  return 0;
}