void reference(const T* input, const T* dense, T* output, int embedding_dim, int batch_size, const int* offset) {
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int range = offset[batch_idx + 1] - offset[batch_idx];
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[offset[batch_idx] + nested_idx] = input[offset[batch_idx] + nested_idx] + dense_elem;
      }
    }
  }
}

template <typename T>
__global__ void dense_esuhm(const T* __restrict__ input, const T* __restrict__ dense, T* __restrict__ output, int embedding_dim, const int* __restrict__ offset) {
  const int batch_idx  = blockIdx.x; // each batch is handled by a block
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = offset[batch_idx + 1] - offset[batch_idx];
  for (int idx = tid; idx < embedding_dim; idx += grain_size) {
    const T dense_elem = dense[batch_idx * embedding_dim + idx];
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[offset[batch_idx] + nested_idx] = input[offset[batch_idx] + nested_idx] + dense_elem;
    }
  }
}

template <typename T>
__global__ void dense_esuhm2(const T* __restrict__ input, const T* __restrict__ dense, T* __restrict__ output, int embedding_dim, const int* __restrict__ offset) {
  const int batch_idx  = blockIdx.x;
  const int start = offset[batch_idx];
  const int range = offset[batch_idx + 1] - start;
  for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
    const T dense_elem = dense[batch_idx * embedding_dim + idx];
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[start + nested_idx] = input[start + nested_idx] + dense_elem;
    }
  }
}

// modified AI-generated kernel
template <typename T>
__global__ void dense_esuhm3(const T* __restrict__ input, const T* __restrict__ dense, T* __restrict__ output, int embedding_dim, const int* __restrict__ offset) {
  const int batch_idx = blockIdx.x;
  const int start = offset[batch_idx];
  const int range = offset[batch_idx + 1] - start;

  for (int s = 0; s < range; s += blockDim.x) {
    int idx = s + threadIdx.x;
    if (idx < range) {
      T input_elem = input[start + idx];
      T dense_elem = dense[batch_idx * embedding_dim + idx % embedding_dim];
      output[start + idx] = input_elem + dense_elem;
    }
  }
}