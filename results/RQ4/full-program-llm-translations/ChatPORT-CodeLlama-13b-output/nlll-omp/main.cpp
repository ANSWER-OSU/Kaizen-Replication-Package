template <typename T, typename AccT, typename IndexT, int N>
__global__ void nll_loss_forward_cuda_kernel(T* output, T* total_weight, const T* input, const IndexT* target, const T* weights, int64_t nframe, int64_t kdim, int64_t ignore_index)
{
  __shared__ AccT sm_inputs[N];
  __shared__ AccT acc_weight[N];

  int tid = threadIdx.x;
  sm_inputs[tid] = AccT(0);
  acc_weight[tid] = AccT(0);

  for (int i = tid; i < nframe; i += N)
  {
    IndexT t = target[i];
    if (t != ignore_index)
    {
      T cur_weight = (weights == nullptr) ? T(1) : weights[t];
      sm_inputs[tid] -= input[i * kdim + t] * cur_weight;
      acc_weight[tid] += cur_weight;
    }
  }

  __syncthreads();

  if (tid == 0)
  {
    AccT output_acc = AccT(0);
    AccT total_weight_acc = AccT(0);
    for (int i = 0; i < N; ++i)
    {
      output_acc -= sm_inputs[i];
      total_weight_acc += acc_weight[i];
    }
    *total_weight = T(total_weight_acc);
    *output = T(output_acc);
  }
}

template <typename T, typename IndexT>
void NLLLossForwardCUDAKernelLauncher(T* output, T* total_weight, const T* input, const IndexT* target, const T* weights, int ignore_index, int64_t nframe, int64_t kdim)
{
  int block_size = std::min(int(1024), nframe); // block size is limited to CUDA kernel thread limit (1024 by default)
  if (block_size == nframe)
  {
    // elementwise kernel (single threadblock)
    nll_loss_forward_cuda_kernel<T, T, IndexT, 1><<<1, 1>>>(output, total_weight, input, target, weights, nframe, kdim, ignore_index);
  }
  else
  {
    // reduction kernel (multiple threadblock)
    nll_loss_forward_cuda_kernel<T, T, IndexT, block_size><<<1, block_size>>>(output, total_weight, input, target, weights, nframe, kdim, ignore_index);
  }
}