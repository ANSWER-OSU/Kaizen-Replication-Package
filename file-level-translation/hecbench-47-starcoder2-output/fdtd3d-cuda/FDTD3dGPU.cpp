void finite_difference(
        float* output,
  const float* input,
  const float* coef,
  const int dimx, const int dimy, const int dimz,
  const int padding)
{
  const int ltidx = omp_get_thread_num() % k_blockDimMaxX;
  const int ltidy = omp_get_thread_num() / k_blockDimMaxX;
  const int workx = k_blockDimMaxX;
  const int worky = k_blockDimMaxY / k_blockDimMaxX;
  const int gtidx = omp_get_thread_num() % k_blockDimMaxX + omp_get_thread_num() / k_blockDimMaxX * workx;
  const int gtidy = omp_get_thread_num() / k_blockDimMaxX + omp_get_thread_num() % k_blockDimMaxX * worky;

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

  bool valid = true;
  if (gtidx >= dimx) valid = false;
  if (gtidy >= dimy) valid = false;

  // For simplicity we assume that the global size is equal to the actual
  // problem size; since the global size must be a multiple of the local size
  // this means the problem size must be a multiple of the local size (or
  // padded to meet this constraint).
  // Preload the "infront" and "behind" data
  #pragma omp parallel for
  for (int i = k_radius_default - 2 ; i >= 0 ; i--)
  {
    behind[i] = input[inputIndex];
    inputIndex += stride_z;
  }

  current = input[inputIndex];
  outputIndex = inputIndex;
  inputIndex += stride_z;

  for (int i = 0 ; i < k_radius_default ; i++)
  {
    infront[i] = input[inputIndex];
    inputIndex += stride_z;
  }

  // Step through the xy-planes
  for (int iz = 0 ; iz < dimz ; iz++)
  {
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
    if (ltidy < k_radius_default)
    {
      tile[ltidy][tx]                  = input[outputIndex - k_radius_default * stride_y];
      tile[ltidy + worky + k_radius_default][tx] = input[outputIndex + worky * stride_y];
    }
    if (ltidx < k_radius_default)
    {
      tile[ty][ltidx]                  = input[outputIndex - k_radius_default];
      tile[ty][ltidx + workx + k_radius_default] = input[outputIndex + workx];
    }
    tile[ty][tx] = current;

    // Compute the output value
    float value = coef[0] * current;
    for (int i = 1 ; i <= k_radius_default ; i++)
    {
      value += coef[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] +
          tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
    }

    if (valid) output[outputIndex] = value;
  }
}

bool fdtdGPU(float* output, const float* input, const float* coeff, 
    const int dimx, const int dimy, const int dimz, const int radius, 
    const int timesteps, const int argc, const char** argv)
{
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
  float* bufferOut = new float[paddedVolumeSize];
  float* bufferIn = new float[paddedVolumeSize];
  float* bufferCoef = new float[radius + 1];
  std::copy(coeff, coeff + (radius + 1), bufferCoef);

  // Set the maximum work group size
  const int maxWorkSize = 256;

  // Set the work group size
  blockSize[0] = k_localWorkX;
  blockSize[1] = maxWorkSize / k_localWorkX;
  gridSize[0] = (unsigned int)ceil((float)dimx / blockSize[0]);
  gridSize[1] = (unsigned int)ceil((float)dimy / blockSize[1]);

  // Copy the input to the device input buffer
  std::copy(input, input + volumeSize, bufferIn + padding);

  // Copy the input to the device output buffer (actually only need the halo)
  std::copy(input, input + volumeSize, bufferOut + padding);

  for (int it = 0; it < timesteps; it++)
  {
    // Toggle the buffers
    float* tmp = bufferIn;
    bufferIn = bufferOut;
    bufferOut = tmp;
  }

  // Read the result back, result is in bufferSrc (after final toggle)
  std::copy(bufferIn + padding, bufferIn + padding + volumeSize, output);

  delete[] bufferIn;
  delete[] bufferOut;
  delete[] bufferCoef;

  return ok;
}