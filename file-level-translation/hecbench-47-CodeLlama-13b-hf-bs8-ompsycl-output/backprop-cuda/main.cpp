void kernel_layerforward(float *__restrict__ input_data, float *__restrict__ input_weights, float *__restrict__ partial_sum, const int hid) {
  extern __shared__ float weight_matrix[];
  const int tid = threadIdx.x;
  const int block_idx = blockIdx.x;
  const int row = block_idx * BLOCK_SIZE + tid;
  int local_col = 0;
  float tmp = 0.;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    int col = BLOCK_SIZE * block_idx + i;
    if (col < hid) {
      weight_matrix[local_col] = input_weights[row * hid + col];
      local_col++;
    }
    __syncthreads();
    if (row < hid) {
      for (int j = 0; j < BLOCK_SIZE; j++) {
        int tcol = BLOCK_SIZE * block_idx + j;
        if (tcol < hid) {
          tmp += input_data[tcol + 1] * weight_matrix[j];
        }
      }
      partial_sum[row] = tmp;
    }
    __syncthreads();
  }
}

void kernel_adjust_weights(float *__restrict__ input_data, float *__restrict__ input_weights, float *__restrict__ hidden_delta, float *__restrict__ input_prev_weights, const int hid) {
  extern __shared__ float weight_matrix[];
  const int tid = threadIdx.x;
  const int block_idx = blockIdx.x;
  const int row = block_idx * BLOCK_SIZE + tid;
  int local_col = 0;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    int col = BLOCK_SIZE * block_idx + i;
    if (col < hid) {
      weight_matrix[local_col] = input_weights[row * hid + col];
      local_col++;
    }
    __syncthreads();
    if (row < hid) {
      for (int j = 0; j < BLOCK_SIZE; j++) {
        int tcol = BLOCK_SIZE * block_idx + j;
        if (tcol < hid) {
          input_prev_weights[row * hid + tcol] =
            weight_matrix[j] * hidden_delta[tcol + 1] * input_data[tid + 1];
        }
      }
    }
    __syncthreads();
  }
}