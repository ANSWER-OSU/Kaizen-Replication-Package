//==============================================================
// Copyright (c) 2019, Intel Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// =============================================================

// ISO2DFD: the 2D-Finite-Difference-Wave Propagation, 
//
// ISO2DFD is a finite difference stencil kernel for solving the 2D acoustic
// isotropic wave equation. Kernels in this sample are implemented as 2nd order
// in space, 2nd order in time scheme without boundary conditions.
// The sample will explicitly run on the GPU as well as CPU to
// calculate a result.  If successful, the output will include GPU device name.
//
// this code sample can be found at :
// https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd


#include <fstream>
#include <iostream>
#include <omp.h>
#include "iso2dfd.h"

#define BLOCK_SIZE 16
#define MIN(a, b) (a) < (b) ? (a) : (b)

/*
 * Host-Code
 * Utility function to display input arguments
 */
void usage(std::string programName) {
  std::cout << " Incorrect parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " n1 n2 Iterations " << std::endl
            << std::endl;
  std::cout << " n1 n2      : Grid sizes for the stencil " << std::endl;
  std::cout << " Iterations : No. of timesteps. " << std::endl;
}

/*
 * Host-Code
 * Function used for initialization
 */
void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, size_t nRows,
                size_t nCols) {
  std::cout << "Initializing ... " << std::endl;

  // Define source wavelet
  float wavelet[12] = {0.016387336f, -0.041464937f, -0.067372555f, 0.386110067f,
                       0.812723635f, 0.416998396f,  0.076488599f,  -0.059434419f,
                       0.023680172f, 0.005611435f,  0.001823209f,  -0.000720549f};

  // Initialize arrays
  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
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
  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (int s = 11; s >= 0; s--) {
    for (size_t i = nRows / 2 - s; i < nRows / 2 + s; i++) {
      size_t offset = i * nCols;
      for (size_t k = nCols / 2 - s; k < nCols / 2 + s; k++) {
        ptr_prev[offset + k] = wavelet[s];
      }
    }
  }
}

/*
 * Host-Code
 * Utility function to calculate L2-norm between resulting buffer and reference
 * buffer
 */
bool within_epsilon(float* output, float* reference, const size_t dimx,
                    const size_t dimy, const unsigned int radius,
                    const float delta = 0.01f) {
  FILE* fp = fopen("./error_diff.txt", "w");
  if (!fp) fp = stderr;

  bool error = false;
  //float abs_delta = fabsf(delta);
  double norm2 = 0;

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
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
  if (error)
    printf("error (Euclidean norm): %.9e\n", norm2);
  return error;
}

/*
 * Host-Code
 * CPU implementation for wavefield modeling
 * Updates wavefield for the number of iterations given in nIteratons parameter
 */
void iso_2dfd_iteration_cpu(float* next, float* prev, float* vel,
                            const float dtDIVdxy, size_t nRows, size_t nCols,
                            int nIterations) {
  float* swap;
  float value = 0.f;
  #pragma omp parallel for collapse(2) num_teams(8) num_threads(16)
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

int main(int argc, char* argv[]) {
  // Arrays used to update the wavefield
  float* prev_base;
  float* next_base;
  float* next_cpu;
  // Array to store wave velocity
  float* vel_base;

  bool error = false;

  size_t nRows, nCols;
  unsigned int nIterations;

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
  prev_base = reinterpret_cast<float*>(malloc(sizeof(float) * nsize));
  next_base = reinterpret_cast<float*>(malloc(sizeof(float) * nsize));
  next_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * nsize));
  vel_base = reinterpret_cast<float*>(malloc(sizeof(float) * nsize));

  // Compute constant value (delta t)^2 (delta x)^2. To be used in wavefield
  // update
  float dtDIVdxy = (DT * DT) / (DXY * DXY);

  // Initialize arrays and introduce initial conditions (source)
  initialize(prev_base, next_base, vel_base, nRows, nCols);

  std::cout << "Grid Sizes: " << nRows << " " << nCols << std::endl;
  std::cout << "Iterations: " << nIterations << std::endl;
  std::cout << std::endl;

  std::cout << "Computing wavefield in device .." << std::endl;

  // Display info about device
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  std::cout << "Running on:: " << devProp.name << std::endl;
  std::cout << "The Device Max Work Group Size is : " << devProp.maxThreadsPerBlock
            << std::endl;

  float* d_next;
  float* d_prev;
  float* d_vel;
  cudaMalloc((void**)&d_next, sizeof(float)*nsize);
  cudaMemcpy(d_next, next_base, sizeof(float)*nsize, cudaMemcpyHostToDevice); 
  cudaMalloc((void**)&d_prev, sizeof(float)*nsize);
  cudaMemcpy(d_prev, prev_base, sizeof(float)*nsize, cudaMemcpyHostToDevice); 
  cudaMalloc((void**)&d_vel, sizeof(float)*nsize);
  cudaMemcpy(d_vel, vel_base, sizeof(float)*nsize, cudaMemcpyHostToDevice); 

  unsigned int grid_cols = (nCols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (nRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0; col < grid_cols; col++) {
      size_t tid = row * grid_cols + col;
      omp_set_thread_num(tid);
      iso_2dfd_kernel<<<dimGrid, dimBlock>>>((row % 2) ? d_prev : d_next, 
                                             (row % 2) ? d_next : d_prev,
                                             d_vel, dtDIVdxy, nRows, nCols);
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(8) num_threads(16)
  for (size_t row = 0; row < grid_rows; row++) {
    for (size_t col = 0;