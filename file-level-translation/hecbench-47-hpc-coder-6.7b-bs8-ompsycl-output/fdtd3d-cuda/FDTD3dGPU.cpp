#include <iostream>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include "FDTD3dGPU.h"
#include "shrUtils.h"

void finite_difference(
        float*__restrict__ output,
  const float*__restrict__ input,
  const float*__restrict__ coef, 
  const int dimx, const int dimy, const int dimz,
  const int padding)
{
  #pragma omp target teams distribute parallel for collapse(3) num_teams(dimz) num_threads(256)
  for (int iz = 0; iz < dimz; iz++) {
    bool valid = true;
    const int ltidx = omp_get_thread_num() % 16;
    const int ltidy = omp_get_thread_num() / 16;
    const int workx = 16;
    const int worky = 16;
    const int gtidx = omp_get_team_num() * workx + ltidx;
    const int gtidy = omp_get_team_num() * worky + ltidy;

    const int stride_y = dimx + 2 * k_radius_default;
    const int stride_z = stride_y * (dimy + 2 * k_radius_default);

    int inputIndex  = 0;
    int outputIndex = 0;

    // Advance inputIndex to start of inner volume
    inputIndex += k_radius_default * stride_y + k_radius_default + padding;

    // Advance inputIndex to target element
    inputIndex += gtidy * stride_y + gtidx;

    float infront[k_radius_default];
    float behind[k_radius_default];
    float current;

    const int tx = ltidx + k_radius_default;
    const int ty = ltidy + k_radius_default;

    if (gtidx >= dimx) valid = false;
    if (gtidy >= dimy) valid = false;

    // Preload the "infront" and "behind" data
    for (int i = k_radius_default - 2 ; i >= 0 ; i--) {
      behind[i] = input[inputIndex];
      inputIndex += stride_z;
    }

    current = input[inputIndex];
    outputIndex = inputIndex;
    inputIndex += stride_z;

    for (int i = 0 ; i < k_radius_default ; i++) {
      infront[i] = input[inputIndex];
      inputIndex += stride_z;
    }

    // Step through the xy-planes
    for (int iz = 0 ; iz < dimz ; iz++) {
      // Advance the slice (move the thread-front)
      for (int i = k_radius_default - 1 ; i > 0 ; i--)
        behind[i] = behind[i - 1];
      behind[0] = current;
      current = infront[0];
      for (int i = 0 ; i < k_radius_default - 1 ; i++)
        infront[i] = infront[i + 1];
      infront[k_radius_default - 1] = input[inputIndex];

      inputIndex  += stride_z;
      outputIndex += stride_z;

      // Note that for the work items on the boundary of the problem, the
      // supplied index when reading the halo (below) may wrap to the
      // previous/next row or even the previous/next xy-plane. This is
      // acceptable since a) we disable the output write for these work
      // items and b) there is at least one xy-plane before/after the
      // current plane, so the access will be within bounds.

      // Compute the output value
      float value = coef[0] * current;
      for (int i = 1 ; i <= k_radius_default ; i++) {
        value += coef[i] * (infront[i-1] + behind[i-1] + output[outputIndex - i * stride_y] + output[outputIndex + i * stride_y] + output[outputIndex - i] + output[outputIndex + i]);
      }

      // Store the output value
      if (valid) output[outputIndex] = value;
    }
  }
}

bool fdtdGPU(float *output, const float *input, const float *coeff, 
    const int dimx, const int dimy, const int dimz, const int radius, 
    const int timesteps, const int argc, const char **argv)
{
  bool ok = true;
  const int         outerDimx  = dimx + 2 * radius;
  const int         outerDimy  = dimy + 2 * radius;
  const int         outerDimz  = dimz + 2 * radius;
  const size_t      volumeSize = outerDimx * outerDimy * outerDimz;

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;

  // Set the maximum work group size
  size_t maxWorkSize = 256;

  // Copy the input to the device input buffer 
  // offset = padding * 4, bytes = volumeSize * 4
  #pragma omp target data map(to: input[0:volumeSize], coeff[0:radius+1]) map(from: output[0:volumeSize])
  {
    // Execute the FDTD
    shrLog(" GPU FDTD loop\n");

    auto start = std::chrono::steady_clock::now();

    for (int it = 0 ; it < timesteps ; it++) {
      // Launch the kernel
      finite_difference(output, input, coeff, dimx, dimy, dimz, padding);

      // Toggle the buffers
      float* tmp = input;
      input = output;
      output = tmp;
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / timesteps);
  }

  return ok;
}