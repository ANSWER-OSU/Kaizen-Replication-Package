template <typename T, int kBlockDimX, int kBlockDimY>
__global__
void ChannelSumNCHW(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  T m_val = 0;
  T v_val = 0;
  const int c = blockIdx.x;

  // sum batches from different channels
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    for (int hw = threadIdx.y; hw < HxW; hw += blockDim.y) {
      const int index = (n * C + c) * HxW + hw;
      m_val += X[index];
      v_val += X[index] * X[index];
    }
  }
  BlockReduce<T, kBlockDimX, kBlockDimY>(m_val, v_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
__global__
void ChannelSumNHWC(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  const int inner_size = N * HxW;
  const int c = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int index = i * C + c;
    m_val += X[index];
    v_val += X[index] * X[index];
  }
  BlockReduce<T>(m_val, v_val);
  if (threadIdx.x == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

extern "C" void ComputeChannelSumNCHW (
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* sum,
    float* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE(
      HxW,
      ChannelSumNCHW,
      float,
      C,
      0,
      N,
      C,
      HxW,
      X,
      sum,
      sumsq);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

extern "C" void ComputeChannelSumNHWC (
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* sum,
    float* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelSumNHWC<float> <<<C, NUM_THREADS>>>(N, C, HxW, X, sum, sumsq);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}