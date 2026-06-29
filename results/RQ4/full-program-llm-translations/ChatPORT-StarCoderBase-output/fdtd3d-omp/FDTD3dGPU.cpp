void finite_difference(const int numTeams, const int numThreads,
        float*__restrict__ output,
  const float*__restrict__ input,
  const float*__restrict__ coef, 
  const int dimx, const int dimy, const int dimz,
  const int padding) {
  #pragma omp target teams num_teams(numTeams)
  {
    float tile[k_blockDimMaxY + 2 * k_radius_default][k_blockDimMaxX + 2 * k_radius_default];
    #pragma omp parallel num_threads(numThreads)
    {
      bool valid = true;
      const int ltidx = omp_get_thread_num() % 16;
      const int ltidy = omp_get_thread_num() / 16;
      const int workx = omp_get_num_threads() % 16;
      const int worky = omp_get_num_threads() / 16;
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

      // For simplicity we assume that the global size is equal to the actual
      // problem size; since the global size must be a multiple of the local size
      // this means the problem size must be a multiple of the local size (or
      // padded to meet this constraint).
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
        #pragma omp barrier

        // Note that for the work items on the boundary of the problem, the
        // supplied index when reading the halo (below) may wrap to the
        // previous/next row or even the previous/next xy-plane. This is
        // acceptable since a) we disable the output write for these work
        // items and b) there is at least one xy-plane before/after the
        // current plane, so the access will be within bounds.

        // Update the data slice in the local tile
        // Halo above & below
        if (ltidy < k_radius_default) {
          tile[ltidy][tx]                  = input[outputIndex - k_radius_default * stride_y];
          tile[ltidy + worky + k_radius_default][tx] = input[outputIndex + worky * stride_y];
        }
        // Halo left & right
        if (ltidx < k_radius_default) {
          tile[ty][ltidx]                  = input[outputIndex - k_radius_default];
          tile[ty][ltidx + workx + k_radius_default] = input[outputIndex + workx];
        }
        tile[ty][tx] = current;
        #pragma omp barrier

        // Compute the output value
        float value = coef[0] * current;
        for (int i = 1 ; i <= k_radius_default ; i++) {
          value += coef[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] + 
              tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
        }

        // Store the output value
        if (valid) {
          #pragma omp atomic
          output[outputIndex] = value;
        }
      }
    }
  }
}

bool fdtdGPU(float *output, const float *input, const float *coeff, 
    const int dimx, const int dimy, const int dimz, const int radius, 
    const int timesteps, const int argc, const char **argv) {
  bool ok = true;
  const int         outerDimx  = dimx + 2 * radius;
  const int         outerDimy  = dimy + 2 * radius;
  const int         outerDimz  = dimz + 2 * radius;
  const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
  size_t            gridSize[2];
  size_t            blockSize[2];

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;


  // Create memory buffer objects
  float* bufferOut; 
  cudaMalloc((void**)&bufferOut, paddedVolumeSize * sizeof(float));

  float* bufferIn; 
  cudaMalloc((void**)&bufferIn, paddedVolumeSize * sizeof(float));

  float* bufferCoef; 
  cudaMalloc((void**)&bufferCoef, (radius+1) * sizeof(float));
  cudaMemcpy(bufferCoef, coeff, (radius+1) * sizeof(float), cudaMemcpyHostToDevice);

  // Set the maximum work group size
  size_t maxWorkSize = 256;

  // Set the work group size
  blockSize[0] = k_localWorkX;  
  blockSize[1] = maxWorkSize / k_localWorkX;
  gridSize[0] = (unsigned int)ceil((float)dimx / blockSize[0]);
  gridSize[1] = (unsigned int)ceil((float)dimy / blockSize[1]);
  shrLog(" set block size to %dx%d\n", blockSize[0], blockSize[1]);
  shrLog(" set grid size to %dx%d\n", gridSize[0], gridSize[1]);
  #pragma omp target teams num_teams(gridSize[1])
  {
    #pragma omp parallel num_threads(blockSize[1])
    {
      #pragma omp target teams distribute parallel for collapse(2) num_teams(gridSize[1]) num_threads(blockSize[1])
      for (int iz = 0 ; iz < dimz ; iz++) {
        for (int y = 0 ; y < dimy ; y++) {
          for (int x = 0 ; x < dimx ; x++) {
            int globalIndex = ((iz * dimy + y) * dimx + x) * 4 + padding;
            bufferIn[globalIndex] = input[globalIndex];
          }
        }
        for (int y = 0 ; y < dimy ; y++) {
          for (int x = 0 ; x < dimx ; x++) {
            int globalIndex = ((iz * dimy + y) * dimx + x) * 4 + padding;
            bufferOut[globalIndex] = input[globalIndex];
          }
        }
        #pragma omp barrier
        finite_difference(gridSize[1], blockSize[1], bufferOut, bufferIn, bufferCoef, dimx, dimy, dimz, padding);
        #pragma omp barrier
        for (int y = 0 ; y < dimy ; y++) {
          for (int x = 0 ; x < dimx ; x++) {
            int globalIndex = ((iz * dimy + y) * dimx + x) * 4 + padding;
            output[globalIndex] = bufferOut[globalIndex];
          }
        }
      }
    }
  }
  cudaFree(bufferIn);
  cudaFree(bufferOut);
  cudaFree(bufferCoef);
  return ok;
}