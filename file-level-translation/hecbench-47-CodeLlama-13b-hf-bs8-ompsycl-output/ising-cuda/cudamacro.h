void cudareduction(const float2 *__restrict__ d_c, float *__restrict__ d_result, const int N) {
  float local_result = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    local_result += d_c[i].x * d_c[i].y;
  }
  __shared__ float shared_result[256];
  shared_result[threadIdx.x] = local_result;
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (threadIdx.x < i) {
      shared_result[threadIdx.x] += shared_result[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) d_result[blockIdx.x] = shared_result[0];
}