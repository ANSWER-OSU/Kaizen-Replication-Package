#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>

// Define the maximum number of threads
#define NUM_THREADS 8

// Forward declarations
bool runTest(int argc, char* argv[]);
void generateRandomData(float* data, int outerDimX, int outerDimY, int outerDimZ, float lowerBound, float upperBound);
bool compareData(float* data1, float* data2, int dimX, int dimY, int dimZ, int radius, float tolerance);
double fdtdReference(float* output, float* input, float* coeff, int dimX, int dimY, int dimZ, int radius, int timesteps);
double fdtdGPU(float* output, float* input, float* coeff, int dimX, int dimY, int dimZ, int radius, int timesteps, int argc, char* argv[]);
void showHelp(int argc, char* argv[]);

int main(int argc, char** argv)
{
  // Check help flag
  if (argc > 1 && std::string(argv[1]) == "--help")
  {
    showHelp(argc, argv);
    return 0;
  }

  // Execute
  bool bTestResult = runTest(argc, argv);

  // Finish
  if (bTestResult == true)
    printf("PASS\n");
  else
    printf("FAIL\n");

  return 0;
}

void showHelp(int argc, char* argv[])
{
  if (argc > 0)
    std::cout << std::endl << argv[0] << std::endl;
  std::cout << std::endl << "Syntax:" << std::endl;
  std::cout << std::left;
  std::cout << "    " << std::setw(20) << "--dimx=<N>" << "Specify number of elements in x direction (excluding halo)" << std::endl;
  std::cout << "    " << std::setw(20) << "--dimy=<N>" << "Specify number of elements in y direction (excluding halo)" << std::endl;
  std::cout << "    " << std::setw(20) << "--dimz=<N>" << "Specify number of elements in z direction (excluding halo)" << std::endl;
  std::cout << "    " << std::setw(20) << "--radius=<N>" << "Specify radius of stencil" << std::endl;
  std::cout << "    " << std::setw(20) << "--timesteps=<N>" << "Specify number of timesteps" << std::endl;
  std::cout << std::endl;
}

bool runTest(int argc, char* argv[])
{
  bool ok = true;

  const int k_dim_default = 1024;
  const int k_radius_default = 32;
  const int k_timesteps_default = 128;
  const int k_dim_min = 64;
  const int k_dim_max = 8 * 1024;
  const int k_timesteps_min = 1;
  const int k_timesteps_max = 1024;

  int defaultDim;
  int dimx = k_dim_default;
  int dimy = k_dim_default;
  int dimz = k_dim_default;
  int radius = k_radius_default;
  int timesteps = k_timesteps_default;
  const float lowerBound = 0.0f;
  const float upperBound = 1.0f;

  // Determine default dimensions
  if (ok)
  {
    // By default, make the edges of the volume elements a multiple of 64B to simplify memory access patterns
    const int roundTarget = 64 / sizeof(float);
    defaultDim = k_dim_default / roundTarget * roundTarget;
    dimx = defaultDim;
    dimy = defaultDim;
    dimz = defaultDim;

    // Check dimension is valid
    if (defaultDim < k_dim_min)
    {
      std::cout << "insufficient device memory (maximum volume on device is " << defaultDim << ", must be between " << k_dim_min << " and " << k_dim_max << ")" << std::endl;
      ok = false;
    }
    else if (defaultDim > k_dim_max)
    {
      defaultDim = k_dim_max;
    }

    // For QA testing, override default volume size
    if (shrCheckCmdLineFlag(argc, argv, "qatest") || shrCheckCmdLineFlag(argc, argv, "qa"))
    {
      dimx = MIN(dimx, k_dim_max);
      dimy = MIN(dimy, k_dim_max);
      dimz = MIN(dimz, k_dim_max);
    }
  }

  // Parse command line arguments
  if (ok)
  {
    char *dim = 0;
    if (shrGetCmdLineArgumentstr(argc, argv, "dimx", &dim))
      if (int(atoi(dim)) < k_dim_min || int(atoi(dim)) > k_dim_max)
      {
        std::cout << "dimx out of range (" << dimx << " requested, must be between " << k_dim_min << " and " << k_dim_max << "), see header files for details." << std::endl;
        ok = false;
      }
      else
      {
        dimx = std::min(int(atoi(dim)), k_dim_max);
      }
    if (shrGetCmdLineArgumentstr(argc, argv, "dimy", &dim))
      if (int(atoi(dim)) < k_dim_min || int(atoi(dim)) > k_dim_max)
      {
        std::cout << "dimy out of range (" << dimy << " requested, must be between " << k_dim_min << " and " << k_dim_max << "), see header files for details." << std::endl;
        ok = false;
      }
      else
      {
        dimy = std::min(int(atoi(dim)), k_dim_max);
      }
    if (shrGetCmdLineArgumentstr(argc, argv, "dimz", &dim))
      if (int(atoi(dim)) < k_dim_min || int(atoi(dim)) > k_dim_max)
      {
        std::cout << "dimz out of range (" << dimz << " requested, must be between " << k_dim_min << " and " << k_dim_max << "), see header files for details." << std::endl;
        ok = false;
      }
      else
      {
        dimz = std::min(int(atoi(dim)), k_dim_max);
      }

    if (shrGetCmdLineArgumentstr(argc, argv, "timesteps", &dim))
      if (int(atoi(dim)) < k_timesteps_min || int(atoi(dim)) > k_timesteps_max)
      {
        std::cout << "timesteps out of range (" << timesteps << " requested, must be between " << k_timesteps_min << " and " << k_timesteps_max << "), see header files for details." << std::endl;
        ok = false;
      }
      else
      {
        timesteps = std::min(int(atoi(dim)), k_timesteps_max);
      }
    if (argc > 0)
      if (strcmp(argv[0], "--qatest") == 0)
      {
        timesteps = k_timesteps_default;
      }
  }

  if (ok)
    std::cout << "Using volume size of: " << dimx << " x " << dimy << " x " << dimz << std::endl;

  // Allocate memory for input and output buffers
  std::vector<float> host_output(dimx * dimy * dimz * 3);
  std::vector<float> input(dimx * dimy * dimz * 3);
  std::vector<float> output(dimx * dimy * dimz * 3);

  // Precompute coefficients
  std::vector<float> coeff(radius + 1);
  for (int i = 0 ; i <= radius ; i++)
  {
    coeff[i] = 0.1f;
  }

  // Generate random data
  generateRandomData(input.data(), dimx + 2 * radius, dimy + 2 * radius, dimz + 2 * radius, lowerBound, upperBound);

  if (ok)
  {
    std::cout << "Running FDTD on volume of " << dimx << " x " << dimy << " x " << dimz <<
        " volume with symmetric filter radius " << radius << " for " << timesteps << " timesteps..." << std::endl;
  }

  // Execute FDTD on the host
  if (ok)
  {
    output.assign(host_output.size(), 0);
    double elapsed = fdtdReference(output.data(), input.data(), coeff.data(), dimx, dimy, dimz, radius, timesteps);
    std::cout << "fdtdReference took " << elapsed << " seconds" << std::endl;
  }

  // Generate the number of threads to use
  int numThreads = std::min(NUM_THREADS, omp_get_max_threads());

  // Execute FDTD on the device using OpenMP
  if (ok)
  {
#pragma omp parallel num_threads(numThreads)
    {
#pragma omp for schedule(static)
      for (int i = 0; i < dimx * dimy * dimz * 3; i++)
      {
         // Execute the FDTD step for the current element
         output[i] = input[i] * coeff[0]
              + (input[i - 3] + input[i + 3] + input[i - dimx * dimy * 3] + input[i + dimx * dimy * 3]) * coeff[1]
              + (input[i - 2] + input[i + 2] + input[i - dimx * dimy * 3 * 2] + input[i + dimx * dimy * 3 * 2]) * coeff[2]
              + (input[i - 1] + input[i + 1] + input[i - dimx * dimy] + input[i + dimx * dimy]) * coeff[3]
              + (input[i - dimx * dimy * 3 - 1] + input[i + dimx * dimy * 3 + 1]) * coeff[4]
              + (input[i - dimx * dimy * 2 - 1] + input[i + dimx * dimy * 2 + 1]) * coeff[5]
              + (input[i - dimx * dimy - 1] + input[i + dimx * dimy + 1]) * coeff[6]
              + (input[i - 10] + input[i + 10]) * coeff[7];
      }
    }

    std::cout << "fdtdOpenMP took " << elapsed << " seconds" << std::endl;
  }

  // Compare the results
  if (ok)
  {
    float tolerance = 0.0001f;
    std::cout << "CompareData (tolerance " << tolerance << ")..." << std::endl;
    ok = compareData(output.data(), host_output.data(), dimx, dimy, dimz, radius, tolerance);
  }

  return ok;
}

void generateRandomData(float* data, int outerDimX, int outerDimY, int outerDimZ, float lowerBound, float upperBound)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(lowerBound, upperBound);

  for (int z = 0; z < outerDimZ; z++)
    for (int y = 0; y < outerDimY; y++)
      for (int x = 0; x < outerDimX; x++)
        for (int t = 0; t < 3; t++)
          data[t + t * outerDimX + t * outerDimX * outerDimY + z] = dis(gen);
}

bool compareData(float* data1, float* data2, int dimX, int dimY, int dimZ, int radius, float tolerance)
{
  bool ok = true;
  int outerDimX = dimX + 2 * radius;
  int outerDimY = dimY + 2 * radius;
  int outerDimZ = dimZ + 2 * radius;

  for (int z = 0; z < outerDimZ; z++)
    for (int y = 0; y < outerDimY; y++)
      for (int x = 0; x < outerDimX; x++)
        for (int t = 0; t < 3; t++)
        {
          int idx = t + t * outerDimX + t * outerDimX * outerDimY + z;
          double diff = data2[idx] - data1[idx];
          ok &= fabs(diff) <= tolerance;
        }

  return ok;
}

double fdtdReference(float* output, float* input, float* coeff, int dimX, int dimY, int dimZ, int radius, int timesteps)
{
  float* inputPtr = input;
  float* outputPtr = output;

  int outerDimX = dimX + 2 * radius;
  int outerDimY = dimY + 2 * radius;
  int outerDimZ = dimZ + 2 * radius;

  for (int t = 0; t < timesteps; t++)
    for (int z = 2 * radius; z < dimZ + 2 * radius; z++)
      for (int y = 2 * radius; y < dimY + 2 * radius; y++)
        for (int x = 2 * radius; x < dimX + 2 * radius; x++)
        {
          int idx = x + outerDimX * (y + outerDimY * z);

          float sum = (*inputPtr) * coeff[0];
          sum += (inputPtr[-3 + outerDimX * outerDimY * outerDimZ] +
            inputPtr[3 + outerDimX * outerDimY * outerDimZ]) * coeff[1];
          sum += (inputPtr[-2 + outerDimX * outerDimY * outerDimZ] +
            inputPtr[2 + outerDimX * outerDimY * outerDimZ]) * coeff[2];
          sum += (inputPtr[-1 + outerDimX * outerDimY * outerDimZ] +
            inputPtr[1 + outerDimX * outerDimY * outerDimZ]) * coeff[3];
          sum += (inputPtr[-outerDimX] + inputPtr[outerDimX]) * coeff[4];
          sum += (inputPtr[-outerDimX * 2] + inputPtr[outerDimX * 2]) * coeff[5];
          sum += (inputPtr[-outerDimY] + inputPtr[outerDimY]) * coeff[6];
          sum += (inputPtr[-10] + inputPtr[10]) * coeff[7];

          // Update output
          *outputPtr = sum;

          // Move to next element
          inputPtr++;
          outputPtr++;
        }

  return 0.0;
}

double fdtdGPU(float* output, float* input, float* coeff, int dimX, int dimY, int dimZ, int radius, int timesteps, int argc, char* argv[])
{
  // Initialize the OpenMP environment
  omp_set_num_threads(omp_get_max_threads());

  // Call GPU implementation of the FDTD, which will be implemented separately.

  return 0.0;
}