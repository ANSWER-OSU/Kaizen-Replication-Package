// Declare the OpenMP version of the CUDA kernel
__global__
void conv3d_s1_omp(const float * __restrict__ X,
               const float * __restrict__ W,
                     float * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int n = blockIdx.x;
  int m = blockIdx.y;
  int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
  if (h < Hout && w < Wout) {
    float s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}

// Implement the OpenMP version of the CUDA kernel
__global__
void conv3d_s1_omp(const float * __restrict__ X,
               const float * __restrict__ W,
                     float * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int h = blockIdx.x / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.x % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.y;
  int m = blockIdx.z;
  if (h < Hout && w < Wout) {
    float s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}


// Define the OpenMP versions of the CUDA kernels
__global__
void conv3d_s2_omp(const float * __restrict__ X,
               const float * __restrict__ W,
                     float * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int m = blockIdx.x;
  int h = blockIdx.y / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.y % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.z;
  if (h < Hout && w < Wout) {
    float s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}

__global__
void conv3d_s3_omp(const float * __restrict__ X,
               const float * __restrict__ W,
                     float * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int h = blockIdx.x / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.x % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.y;
  int m = blockIdx.z;
  if (h < Hout && w < Wout) {
    float s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}