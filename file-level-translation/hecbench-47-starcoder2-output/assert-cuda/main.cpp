// This code is not tested and might contain compilation errors

#include <omp.h>
#include <stdio.h>
#include <assert.h>

// OpenMP program to test assert function
void test_program(int N)
{
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    assert(thread_id < N);
  }
}

// OpenMP program to measure the performance impact of assert()
void performance_program()
{
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    assert(thread_id <= omp_get_num_threads());

    int s = 0;
    for (int n = 1; n <= thread_id; n++) {
      s++;
      assert(s <= thread_id);
    }
  }
}

void performance_program_2()
{
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int s = 0;
    for (int n = 1; n <= thread_id; n++) {
      s++;
      assert(s <= thread_id);
    }
  }
}

// Main function to run the test and performance programs
int main(int argc, char **argv)
{
  int N_threads = 32;

  // Set the number of threads for OpenMP
  omp_set_num_threads(N_threads);

  // Test the assert function by running the test program
  test_program(60);

  // Measure the performance impact of assert()
  double start_time = omp_get_wtime();
  performance_program();
  double end_time = omp_get_wtime();
  printf("Kernel time: %f\n", end_time - start_time);

  start_time = omp_get_wtime();
  performance_program_2();
  end_time = omp_get_wtime();
  printf("Kernel time: %f\n", end_time - start_time);

  return 0;
}