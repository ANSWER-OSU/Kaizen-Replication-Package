template <typename T>
__global__ void rsquared(const int N, const int D, const T* Y, const T* Y_hat, T* result) {
  T sum_x = T(0), sum_y = T(0), sum_xy = T(0), sum_yy = T(0);
  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      T y = Y[row * D + col];
      T y_hat = Y_hat[row * D + col];
      sum_x += y;
      sum_y += y * y_hat;
      sum_xy += y * y_hat;
      sum_yy += y * y;
    }
    __syncthreads();
  }
  T denominator = sqrtf(sum_yy - sum_y * sum_y / N) * sqrtf(sum_xx - sum_x * sum_x / N);
  if (denominator != 0) {
    *result = (sum_xy - sum_x * sum_y / N) / denominator;
  }
}

template <typename T>
int rsquared(const int numTeams, const int numThreads, const int N, const int D, const T* Y, const T* Y_hat, T* result) {
  /*
  Compute the r^2 score for a given dataset.
  */
  int ret = 0;
  if (numTeams > 0) {
    ret = cudaLaunchTeamKernel((void*)rsquared<T>, numTeams, numThreads, N, D, Y, Y_hat, result);
  } else {
    rsquared<T> << <1, 1 >> >(N, D, Y, Y_hat, result);
  }
  return ret;
}