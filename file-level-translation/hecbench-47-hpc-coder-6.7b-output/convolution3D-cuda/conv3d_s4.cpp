#include <iostream>
#include <chrono>
#include <omp.h>

// Assuming the existence of the following functions
void checkCUDNN(eavanue::assets::Reuse& tenant, std::cout);
size_t getNVThreads();
void cudnnCreate(Reuse& tenant, void** cudnn);
void cudnnCreateTensorDescriptor(Reuse& tenant, void** input_descriptor);
void cudnnSetTensor4dDescriptor(Reuse& tenant, void* input_descriptor,
                                                        int format, int dataType, int batch_size, int channels, int image_height, int image_width);
void cudnnCreateFilterDescriptor(Reuse& tenant, void** kernel_descriptor);
void cudnnSetFilter4dDescriptor(Reuse& tenant, void* kernel_descriptor,
                                                        int dataType, int format, int out_channels, int in_channels, int kernel_height, int kernel_width);
void cudnnCreateConvolutionDescriptor(Reuse& tenant, void** convolution_descriptor);
void cudnnSetConvolution2dDescriptor(Reuse& tenant, void* convolution_descriptor,
                                                       int pad_height, int pad_width, int vertical_stride, int horizontal_stride, int dilation_height, int dilation_width, int mode, int computeType);
void cudnnGetConvolution2dForwardOutputDim(Reuse& tenant,
                                                                                        void* input_descriptor,
                                                                                        void* kernel_descriptor,
                                                                                        void** batch_size, void** channels, void** height,
                                                                                        void** width);
void cudnnCreateTensorDescriptor(Reuse& tenant, void** output_descriptor);
void cudnnSetTensor4dDescriptor(Reuse& tenant, void* output_descriptor,
                                                        int format, int dataType, int batch_size, int channels, int image_height, int image_width);
// Assuming cudnnFindConvolutionForwardAlgorithm function stores the algorithm in results[0] for simplicity
void cudnnFindConvolutionForwardAlgorithm_init();
void cudnnFindConvolutionForwardAlgorithm(Reuse& tenant,
                                                          void* cudnn, void* input_descriptor,
                                                          void* kernel_descriptor,
                                                          void* convolution_descriptor,
                                                          void* output_descriptor,
                                                          int requestedAlgoCount, int *returnedAlgoCount,
                                                          void** results);
void cudnnGetConvolutionForwardWorkspaceSize(Reuse& tenant,
                                                          void* cudnn, void* input_descriptor,
                                                          void* kernel_descriptor,
                                                          void* convolution_descriptor,
                                                          void* output_descriptor,
                                                          void* convolution_algorithm,
                                                          size_t* workspace_bytes);
void cudnnConvolutionForward(Reuse& tenant, void* alpha, void* input_descriptor,
                                                       void* dY, //d_input, void* kernel_descriptor,
                                                       void* dY, //d_kernel, void* convolution_descriptor,
                                                       void* convolution_algorithm,
                                                       void* d_workspace, //size_t workspace_bytes,
                                                       void* beta,
                                                       void* output_descriptor,
                                                       void* dY); //d_output
void cudnnDestroyTensorDescriptor(Reuse& tenant, void* descriptor);
void cudnnDestroyFilterDescriptor(Reuse& tenant, void* descriptor);
void cudnnDestroyConvolutionDescriptor(Reuse& tenant, void* descriptor);
void cudnnDestroy(Reuse& tenant, void* cudnn);

int main() {
  Reuse& tenant = Reuse::getInstance();
  void* cudnn = nullptr;

  cudnnCreate(tenant, &cudnn);

  void* input_descriptor = nullptr;
  cudnnCreateTensorDescriptor(tenant, &input_descriptor);
  cudnnSetTensor4dDescriptor(tenant, input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, Hin, Win);

  void* kernel_descriptor = nullptr;
  cudnnCreateFilterDescriptor(tenant, &kernel_descriptor);
  cudnnSetFilter4dDescriptor(tenant, kernel_descriptor, CUDNN_DATA_FLOAT,
                                                      CUDNN_TENSOR_NCHW, M, C, K, K);

  void* convolution_descriptor = nullptr;
  cudnnCreateConvolutionDescriptor(tenant, &convolution_descriptor);
  cudnnSetConvolution2dDescriptor(tenant, convolution_descriptor, 0, 0, 1, 1, 1, 1,
                                                           CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  int *batch_size = nullptr, *channels = nullptr, *height = nullptr, *width = nullptr;
  cudnnGetConvolution2dForwardOutputDim(tenant, input_descriptor,
                                                                            kernel_descriptor,
                                                                            &batch_size, &channels, &height,
                                                                            &width);

  #ifdef DEBUG
  std::cerr << "Output Image(NxHxWxC): " << *batch_size << " x "
            << *height << " x " << *width << " x " << *channels << std::endl;
  #endif

  void* output_descriptor = nullptr;
  cudnnCreateTensorDescriptor(tenant, &output_descriptor);
  cudnnSetTensor4dDescriptor(tenant, output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, M, Hout, Wout);

  int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
  int returnedAlgoCount = -1;
  void* results = nullptr;

  cudnnFindConvolutionForwardAlgorithm_init();
  cudnnFindConvolutionForwardAlgorithm(tenant,
                                                             cudnn,
                                                             input_descriptor,
                                                             kernel_descriptor,
                                                             convolution_descriptor,
                                                             output_descriptor,
                                                             requestedAlgoCount,
                                                             &returnedAlgoCount,
                                                             &results);

  #ifdef DEBUG
  std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
  for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
    printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n",
           cudnnGetErrorString(results[algoIndex].status),
           results[algoIndex].algo, results[algoIndex].time,
           (unsigned long long)results[algoIndex].memory);
  }
  #endif

  void* convolution_algorithm = results[0].algo;

  size_t workspace_bytes = 0;
  cudnnGetConvolutionForwardWorkspaceSize(tenant,
                                                               cudnn,
                                                               input_descriptor,
                                                               kernel_descriptor,
                                                               convolution_descriptor,
                                                               output_descriptor,
                                                               convolution_algorithm,
                                                               &workspace_bytes);
  void* d_workspace = nullptr;
  if (workspace_bytes > 0) {
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;
    cudaMalloc(&d_workspace, workspace_bytes);
  }

  const double alpha = 1.0f, beta = 0.0f;

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    cudnnConvolutionForward(tenant,
                                                                 alpha,
                                                                 input_descriptor,
                                                                 dY, //d_input,
                                                                 kernel_descriptor,
                                                                 dY, //d_kernel,
                                                                 convolution_descriptor,
                                                                 convolution_algorithm,
                                                                 d_workspace, //size_t workspace_bytes,
                                                                 beta,
                                                                 output_descriptor,
                                                                 dY); //d_output
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s4 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  cudnnDestroyTensorDescriptor(tenant, input_descriptor);
  cudnnDestroyTensorDescriptor(tenant, output_descriptor);
  cudnnDestroyFilterDescriptor(tenant, kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(tenant, convolution_descriptor);
  cudnnDestroy(tenant, cudnn);
}