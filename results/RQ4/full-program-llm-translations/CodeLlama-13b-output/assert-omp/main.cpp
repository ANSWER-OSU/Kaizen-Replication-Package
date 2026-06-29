#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>

// Tests assert function.
// Thread whose id > N will print assertion failed error message.
__global__ void testKernel(int N)
{
#pragma omp parallel for
  for (int gid = 0; gid < N; gid++) {
    assert(gid < N);
  }
}

// Performance impact of assert()
__global__ void perfKernel()
{
#pragma omp parallel for
  for (int gid = 0; gid < gridDim.x * blockDim.x; gid++) {
    assert(gid <= blockDim.x * gridDim.x);
    int s = 0;
    for (int n = 1; n <= gid; n++) {
      s++;
      assert(s <= gid);
    }
  }
}

__global__ void perfKernel2()
{
#pragma omp parallel for
  for (int gid = 0; gid < gridDim.x * blockDim.x; gid++) {
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
  int Nblocks = 2;
  int Nthreads = 32;
  omp_error_t error;

  // Kernel configuration, where a one-dimensional
  // grid and one-dimensional blocks are configured.
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(Nthreads);

  printf("\nLaunch kernel to generate assertion failures\n");
  testKernel<<<dimGrid, dimBlock>>>(60);

  // Synchronize (flushes assert output).
  printf("\n-- Begin assert output\n\n");
  error = omp_device_synchronize();
  printf("\n-- End assert output\n\n");

  // Check for errors and failed asserts in asynchronous kernel launch.
  if (error == omp_error_assert) {
    printf("Device assert failed as expected, "
           "OMP error message is: %s\n\n",
           omp_get_error_string(error));
  }

  return (error == omp_error_assert);
}

bool runPerf(int argc, char **argv)
{
  int Nblocks = 1000;
  int Nthreads = 256;

  dim3 dimGrid(Nblocks);
  dim3 dimBlock(Nthreads);

  printf("\nLaunch kernel to evaluate the impact of assertion on performance \n");

  printf("Each thread in the kernel executes threadID + 1 assertions\n");
  auto start = std::chrono::steady_clock::now();
  perfKernel<<<dimGrid, dimBlock>>>();
  omp_device_synchronize();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Kernel time : %f\n", time.count());

  printf("Each thread in the kernel executes threadID assertions\n");
  start = std::chrono::steady_clock::now();
  perfKernel2<<<dimGrid, dimBlock>>>();
  omp_device_synchronize();
  end = std::chrono::steady_clock::now();
  time = end - start;
  printf("Kernel time : %f\n", time.count());

  return true;
}