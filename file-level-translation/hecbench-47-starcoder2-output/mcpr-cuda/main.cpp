// CUDA kernel to translate
__global__ void compute_probs(
  const double *alphas,
  const double *rands,
        double *probs,
  const int n,
  const int K,
  const int M)
{
  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int xIndex = bx * blockDim.x + tx;

  if (xIndex >= n) return;

  double sum = 0;

  for (int k = 0; k < K; k++) {
    double product = 1;
    for (int m = 0; m < M; m++) {
      int index = m * K + k;
      product *= exp(-0.5 * rands[index] * rands[index] - alphas[index]);
    }
    sum += product;
  }
  probs[xIndex] = sum;
}