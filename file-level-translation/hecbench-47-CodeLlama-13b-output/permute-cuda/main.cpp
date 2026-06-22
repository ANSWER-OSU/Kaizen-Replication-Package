#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define B 32 // batch size
#define T 1024 // number of tokens per document
#define HS 32 // head size

using namespace cv::cuda;

__global__ void permute_kernel(
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    const float* inp,
    int B, int H, int NH, int d) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B * H * T * NH) {
    int b = idx / (H*T*NH);
    int rest = idx % (H*T*NH);
    int h_ = rest / (T*NH);
    rest = rest % (T*NH);
    int n = rest / NH;
    int nh_ = rest % NH;

    int inp_idx = (b*T*3*H) + (n*3*H) + (h_*NH) + (nh_*HS);

    Q[idx] = inp[inp_idx];
    K[idx] = inp[inp_idx + HS];
    V[idx] = inp[inp_idx + 2*HS];
  }
}

int main() {

  GpuMat d_inp(B*T*3*HS, 1, CV_32FC1);
  GpuMat d_out_Q(B*T*3*HS, 1, CV_32FC1);
  GpuMat d_out_K(B*T*3*HS, 1, CV_32FC1);
  GpuMat d_out_V(B*T*3*HS, 1, CV_32FC1);

  // Initialize inputs and outputs
  d_inp.setTo(cv::Scalar(1.));
  d_out_Q.setTo(cv::Scalar(0.));
  d_out_K.setTo(cv::Scalar(0.));
  d_out_V.setTo(cv::Scalar(0.));

  int block_size = 256;
  dim3 grid(ceil(B*T*3*HS*H / (block_size* (float)H)), 1, 1);
  dim3 block(block_size, 1, 1);

  // Launch kernel
  permute_kernel<<<grid, block>>>(d_out_Q.ptr_gpu(), d_out_K.ptr_gpu(), d_out_V.ptr_gpu(), d_inp.ptr_gpu(), B, H, NH, HS);

  // Copy results back to host
  GpuMat d_out_permuted_q = d_out_Q.clone();
  GpuMat d_out_permuted_k = d_out_K.clone();
  GpuMat d_out_permuted_v = d_out_V.clone();

  // Validate results
  float* q_ptr = (float*)d_out_permuted_q.ptr_gpu();
  float* k_ptr = (float*)d_out_permuted_k.ptr_gpu();
  float* v_ptr = (float*)d_out_permuted_v.ptr_gpu();

  for (int i=0; i<B*T*3*HS; i++) {
    if (q_ptr[i] != d_inp.ptr_gpu()[i]) std::cout << "Q: " << d_inp.ptr_gpu()[i] << " vs " << q_ptr[i] << std::endl;
    if (k_ptr[i] != d_inp.ptr_gpu()[i]) std::cout << "K: " << d_inp.ptr_gpu()[i] << " vs " << k_ptr[i] << std::endl;
    if (v_ptr[i] != d_inp.ptr_gpu()[i]) std::cout << "V: " << d_inp.ptr_gpu()[i] << " vs " << v_ptr[i] << std::endl;
  }

  return 0;
}