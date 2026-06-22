#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>

#define BLOCK_SIZE 16
#define MIN(a, b) (a) < (b) ? (a) : (b)

constexpr unsigned int nBlocks = 16;
constexpr unsigned int nGrids = 16;

// Host-code
void usage(const char* programName) {
  std::cout << " Incorrect parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " n1 n2 Iterations " << std::endl
            << std::endl;
  std::cout << " n1 n2      : Grid sizes for the stencil " << std::endl;
  std::cout << " Iterations : No. of timesteps. " << std::endl;
}

// Host-code
void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, size_t nRows,
                size_t nCols) {
  std::cout << "Initializing ... " << std::endl;

  // Define source wavelet
  float wavelet[12] = {0.016387336, -0.041464937, -0.067372555, 0.386110067,
                       0.812723635, 0.416998396,  0.076488599,  -0.059434419,
                       0.023680172, 0.005611435,  0.001823209,  -0.000720549};

  // Initialize arrays
  for (size_t i = 0; i < nRows; i++) {
    size_t offset = i * nCols;

    for (size_t k = 0; k < nCols; k++) {
      ptr_prev[offset + k] = 0.0f;
      ptr_next[offset + k] = 0.0f;
      // pre-compute squared value of sample wave velocity v*v (v = 1500 m/s)
      ptr_vel[offset + k] = 2250000.0f;
    }
  }
  // Add a source to initial wavefield as an initial condition
  for (int s = 11; s >= 0; s--) {
    for (size_t i = nRows / 2 - s; i < nRows / 2 + s; i++) {
      size_t offset = i * nCols;
      for (size_t k = nCols / 2 - s; k < nCols / 2 + s; k++) {
        ptr_prev[offset + k] = wavelet[s];
      }
    }
  }
}

// Host-code
bool within_epsilon(float* output, float* reference, const size_t dimx,
                    const size_t dimy, const unsigned int radius,
                    const float delta = 0.01f) {
  FILE* fp = fopen("./error_diff.txt", "w");
  if (!fp) fp = stderr;

  bool error = false;
  //float abs_delta = fabsf(delta);
  double norm2 = 0;

  for (size_t iy = 0; iy < dimy; iy++) {
    for (size_t ix = 0; ix < dimx; ix++) {
      if (ix >= radius && ix < (dimx - radius) && iy >= radius &&
          iy < (dimy - radius)) {
        float difference = fabsf(*reference - *output);
        norm2 += difference * difference;
        if (difference > delta) {
          error = true;
          fprintf(fp, " ERROR: (%zu,%zu)\t%e instead of %e (|e|=%e)\n", ix, iy,
                  *output, *reference, difference);
        }
      }

      ++output;
      ++reference;
    }
  }

  if (fp != stderr) fclose(fp);
  norm2 = sqrt(norm2);
  if (error) printf("error (Euclidean norm): %.9e\n", norm2);
  return error;
}

// Host-code
void iso_2dfd_iteration_cpu(float* next, float* prev, float* vel,
                            const float dtDIVdxy, size_t nRows, size_t nCols,
                            int nIterations) {
  float* swap;
  float value = 0.f;
  for (int k = 0; k < nIterations; k += 1) {
    for (size_t i = 1; i < nRows - HALF_LENGTH; i += 1) {
      for (size_t j = 1; j < nCols - HALF_LENGTH; j += 1) {
        // Stencil code to update grid
        size_t gid = j + (i * nCols);
        value = 0.f;
        value += prev[gid + 1] - 2.f * prev[gid] + prev[gid - 1];
        value += prev[gid + nCols] - 2.f * prev[gid] + prev[gid - nCols];
        value *= dtDIVdxy * vel[gid];
        next[gid] = 2.f * prev[gid] - next[gid] + value;
      }
    }

    // Swap arrays
    swap = next;
    next = prev;
    prev = swap;
  }
}

// Device-code
void iso_2dfd_kernel(float* next, float* prev, float* vel, const float dtDIVdxy,
                     size_t nRows, size_t nCols, int nIterations) {
  size_t gidCol = blockDim.x * blockIdx.x + threadIdx.x;
  size_t gidRow = blockDim.y * blockIdx.y + threadIdx.y;

  if (gidRow < nRows && gidCol < nCols) {
    // Stencil code to update grid point at position given by global id (gid)
    //    if ((gidCol >= HALF_LENGTH && gidCol < nCols - HALF_LENGTH) &&
    //        (gidRow >= HALF_LENGTH && gidRow < nRows - HALF_LENGTH)) {
    //      float value = 0.f;
    //      value += prev[gid + 1] - 2.f * prev[gid] + prev[gid - 1];
    //      value += prev[gid + nCols] - 2.f * prev[gid] + prev[gid - nCols];
    //      value *= dtDIVdxy * vel[gid];
    //      next[gid] = 2.f * prev[gid] - next[gid] + value;
    //    }
  }
}

int main(int argc, char* argv[]) {
  // Arrays used to update the wavefield
  float* prev_base{};
  float* next_base{};
  float* next_cpu{};
  // Array to store wave velocity
  float* vel_base{};

  bool error = false;

  size_t nRows, nCols;
  unsigned int nIterations{};

  // Read parameters
  try {
    nRows = std::stoi(argv[1]);
    nCols = std::stoi(argv[2]);
    nIterations = std::stoi(argv[3]);
  }
  catch (...) {
    usage(argv[0]);
    return 1;
  }

  // Compute the total size of grid
  size_t nsize = nRows * nCols;

  // Allocate arrays to hold wavefield and velocity
  prev_base = new float[nsize];
  next_base = new float[nsize];
  next_cpu = new float[nsize];
  vel_base = new float[nsize];

  // Compute constant value (delta t)^2 (delta x)^2.
  // To be used in wavefield update
  float dtDIVdxy = (DT * DT) / (DXY * DXY);

  // Initialize arrays and introduce initial conditions (source)
  initialize(prev_base, next_base, vel_base, nRows, nCols);

  std::cout << "Grid Sizes: " << nRows << " " << nCols << std::endl;
  std::cout << "Iterations: " << nIterations << std::endl;
  std::cout << std::endl;

  // Compute wavefield on CPU (for validation)
  std::cout << "Computing wavefield in CPU .." << std::endl;
  // Re-initialize arrays
  initialize(prev_base, next_cpu, vel_base, nRows, nCols);

  // Compute wavefield on CPU
  auto start = std::chrono::steady_clock::now();
  iso_2dfd_iteration_cpu(next_cpu, prev_base, vel_base, dtDIVdxy, nRows, nCols,
                         nIterations);

  // Compute and display time used by CPU
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  std::cout << "Host time: " << time << " ms" << std::endl;
  std::cout << std::endl;

  // Add barrier so that device functions are called after the CPU computation is complete
#pragma omp barrier

  // Build kernels and compute parameters for the kernel grid and blocks
  //!TODO!

  // Iteration over time steps
  for (unsigned int k = 0; k < nIterations; k += 1) {
    // executes a compute operation on all devices in the current Intel OpenMP
    // when the alpha library is linked with OpenMP, kernels are automatically
    // copied from host to device. A safety check prevents host data from being
    // read when a kernel is executing on a device. Refer to the "WORKING WITH CONCURRENT
    // KERNELS AND OPENMP" section in the Intel OpenMP documentation for more information.
#pragma omp parallel
#pragma omp loop
    for (size_t i = 1; i < nRows - HALF_LENGTH; i += 1) {
      for (size_t j = 1; j < nCols - HALF_LENGTH; j += 1) {
        // Stencil code to update grid
        // Only computed if the thread is within the valid range of the grid
        if (i >= nBlocks && i < nCols - nBlocks && j >= nBlocks && j < nRows - nBlocks) {
          iso_2dfd_kernel(next_base, prev_base, vel_base, dtDIVdxy, nRows, nCols,
                          nIterations);
        }
      }
    }
    // Wait for all devices to complete the computation
#pragma omp barrier
    if (k % 2 = 0) {
      // Swap arrays
      float* swap = next_base;
      next_base = prev_base;
      prev_base = swap;
    }
  }

  // Free memory allocated on host
  delete[] prev_base;
  delete[] next_base;
  delete[] vel_base;

  std::cout << "Check difference between final wavefields computed in device and host " << std::endl;
  error = within_epsilon(next_base, next_cpu, nRows, nCols, HALF_LENGTH, 0.1f);
  std::cout << (error ? "FAIL" : "PASS") << std::endl;
  return error ? 1 : 0;
}