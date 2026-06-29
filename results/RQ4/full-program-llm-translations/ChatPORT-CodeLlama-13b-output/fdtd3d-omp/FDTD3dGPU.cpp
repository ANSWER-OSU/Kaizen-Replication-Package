void fdtdOpenMP(float *output, const float *input, const float *coeff, 
    const int dimx, const int dimy, const int dimz, const int radius, 
    const int timesteps, const int argc, const char **argv) {
  bool ok = true;
  const int outerDimx = dimx + 2 * radius;
  const int outerDimy = dimy + 2 * radius;
  const int outerDimz = dimz + 2 * radius;
  const size_t volumeSize = outerDimx * outerDimy * outerDimz;
  size_t gridSize [3];
  size_t blockSize [3];

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;

  // Create memory buffer objects
  float *bufferOut;
  posix_memalign((void**) &bufferOut, 128, paddedVolumeSize * sizeof(float));

  float *bufferIn;
  posix_memalign((void**) &bufferIn, 128, paddedVolumeSize * sizeof(float));

  float *bufferCoef;
  posix_memalign((void**) &bufferCoef, 128, (radius+1) * sizeof(float));
  memcpy(bufferCoef, coeff, (radius+1) * sizeof(float));

  // Set the work group size
  int workGroupSize = 256;
  blockSize[0] = 1;
  blockSize[1] = workGroupSize / blockSize[0];
  gridSize[0] = (unsigned int) ceil((float) dimx / blockSize[0]);
  gridSize[1] = (unsigned int) ceil((float) dimy / blockSize[1]);
#ifdef OMP_TARGET_OFFLOAD
  gridSize[2] = (unsigned int) ceil((float) dimz / blockSize[2]);
  shrLog(" set block size to %dx%dx%d\n", blockSize[0], blockSize[1], blockSize[2]);
  shrLog(" set grid size to %dx%dx%d\n", gridSize[0], gridSize[1], gridSize[2]);
#endif

  // Copy the input to the device input buffer 
  // offset = padding * 4, bytes = volumeSize * 4
  int omp_num_teams = gridSize[0] * gridSize[1] * gridSize[2];
  int omp_num_threads = blockSize[0] * blockSize[1] * blockSize[2];
  #pragma omp target teams distribute parallel for num_teams(omp_num_teams) num_threads(omp_num_threads) 
  for (size_t index = 0; index < paddedVolumeSize; index++) {
    bufferIn [index] = input [index];
  }

  // Copy the input to the device output buffer (actually only need the halo)
  #pragma omp target teams distribute parallel for num_teams(omp_num_teams) num_threads(omp_num_threads) 
  for (size_t index = 0; index < paddedVolumeSize; index++) {
    bufferOut [index] = input [index];
  }

  // Execute the FDTD
  shrLog(" OpenMP FDTD loop\n");

  auto start = std::chrono::steady_clock::now();
  for (int it = 0; it < timesteps; it++) {
    #pragma omp target teams distribute parallel for num_teams(omp_num_teams) num_threads(omp_num_threads) collapse(3)
    for (int iz = 0; iz < dimz; iz++) {
      for (int iy = 0; iy < dimy; iy++) {
        for (int ix = 0; ix < dimx; ix++) {
          int inputIndex  = ix + iy * dimx + iz * dimx * dimy;
          int outputIndex = ix + iy * dimx + iz * dimx * dimy;

          float infront[radius];
          float behind[radius];
          float current = input [inputIndex];

          // Preload the "infront" and "behind" data
          for (int i = radius - 2; i >= 0; i--) {
            behind[i] = input [outputIndex];
            outputIndex += outerDimz;
          }

          // Current element
          current = input [inputIndex];
          outputIndex = inputIndex;
          inputIndex += outerDimz;

          // Front elements
          for (int i = 0; i < radius; i++) {
            infront[i] = input [inputIndex];
            inputIndex += outerDimz;
          }

          for (int t = 0; t < radius; t++) {
            // Compute a single element 
            float value = bufferCoef[0] * current;
            for (int i = 1; i <= radius; i++) {
              int inputIndex_t_iy_ix = (t + i) * dimx + iy * dimx + ix;
              int inputIndex_t_iy    = t * dimx + iy * dimx + ix;
              int inputIndex_t       = t * dimx + iy * dimx;
              value += bufferCoef[i] * (infront[i - 1] + behind[i - 1] +
                  input [inputIndex_t_iy_ix] + input [inputIndex_t_iy + i] +
                  input [inputIndex_t + iy + i] + input [inputIndex_t + iy - i]);
            }

            // Stage the output 
            bufferOut [outputIndex] = value;

            // Rotate the "infront" and "behind" buffers
            for (int i = radius - 1; i > 0; i--) {
              infront[i] = infront[i - 1];
              behind[i]  = behind[i - 1];
            }
            infront[0] = input [inputIndex];
            behind[0]  = current;

            inputIndex  += outerDimz;
            outputIndex += outerDimz;
          }
        }
      }
    }

    // Toggle the buffers
    float *tmp = bufferIn;
    bufferIn = bufferOut;
    bufferOut = tmp;
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / timesteps);

  // Read the result back, result is in bufferSrc (after final toggle)
  memcpy(output, bufferIn + padding, volumeSize * sizeof(float));
  free (bufferIn);
  free (bufferOut);
  free (bufferCoef);
}