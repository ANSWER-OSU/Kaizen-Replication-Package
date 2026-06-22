void cudnn_conv3d_s4(const int N, const int M, const int C, const int Hin, const int Win, const int K, const int Hout, const int Wout, const float* dX, const float* dW, float* dY) {

  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, Hin, Win));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, M, C, K, K));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  int batch_size{0}, channels{0}, height{0}, width{0};
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, kernel_descriptor, &batch_size, &channels, &height, &width));

  #ifdef DEBUG
  std::cerr << "Output Image(NxHxWxC): " << batch_size << " x " << height << " x " << width << " x " << channels << std::endl;
  #endif

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, M, Hout, Wout));


  int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
  int returnedAlgoCount = -1;
  cudnnConvolutionFwdAlgoPerf_t results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, requestedAlgoCount, &returnedAlgoCount, results));

  #ifdef DEBUG
  std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
  for (int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex) {
    printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, (unsigned long long)results[algoIndex].memory);
  }
  #endif

  cudnnConvolutionFwdAlgo_t convolution_algorithm = results[0].algo;

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_bytes));
  void* d_workspace{nullptr};
  if (workspace_bytes > 0) {
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
    cudaMalloc(&d_workspace, workspace_bytes);
  }

  const float alpha = 1.0f, beta = 0.0f;

  int repeat = 1000;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, dX, kernel_descriptor, dW, convolution_descriptor, convolution_algorithm, d_workspace, workspace_bytes, &beta, output_descriptor, dY));
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  float time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s4 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);
}