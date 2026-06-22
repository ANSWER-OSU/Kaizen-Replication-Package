#include <iostream>
#include <vector>

#define CUDA_KERNEL(idx, t) std::cerr << "CUDA kernel " << idx << " of type " << t << " executed" << std::endl;

#ifdef DEBUG
#define cudnnSafeCall(err) checkCUDNN(err, __FILE__, __LINE__);
#else
#define cudnnSafeCall(err) checkCUDNN(err);
#endif


void checkCUDNN(cudnnStatus_t status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << cudnnGetErrorString(status) << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}


int main(int argc, char *argv[]) {
    // Set batch size (N) and number of channels (C)
    int batch_size = 1, channels = 3;

    // Set input and output side dimensions
    int height_in = 224, width_in = 224;
    int height_out = 56, width_out = 56;

    // Set kernel size (M)
    int kernel_size = 32;

    // Create workspace memory
    cudaMalloc(&d_workspace, workspace_bytes);

    // Define strides and padding for convolution
    cudnnConvolutionDescriptor_t cudnn_convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&cudnn_convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(cudnn_convolution_descriptor,
                                               0,
                                               0,
                                               1,
                                               1,
                                               1,
                                               1,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));

    // Set convolution algorithm
    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // Compute convolution output shape
    int batch_size_out;
    cudnnGetConvolution2dForwardOutputDim(cudnn_convolution_descriptor,
                                       height_in,
                                       width_in,
                                       height_out,
                                       width_out,
                                       batch_size,
                                       channels,
                                       &batch_size_out);


    // Create tensor descriptors
    cudnnTensorDescriptor_t d_X, d_kernel, d_Y;
    checkCUDNN(cudnnCreateTensorDescriptor(&d_X));
    checkCUDNN(cudnnCreateTensorDescriptor(&d_kernel));
    checkCUDNN(cudnnCreateTensorDescriptor(&d_Y));
    checkCUDNN(cudnnSetTensor4dDescriptor(d_X,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        batch_size,
                                        channels,
                                        height_in,
                                        width_in));
    checkCUDNN(cudnnSetTensor4dDescriptor(d_kernel,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        kernel_size,
                                        channels,
                                        height_out,
                                        width_out));
    checkCUDNN(cudnnSetTensor4dDescriptor(d_Y,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        batch_size,
                                        kernel_size,
                                        height_out,
                                        width_out));


    // Compute convolution output shape
    int* d_output_shape;
    cudnnGetConvolution2dForwardOutputShape(cudnn_convolution_descriptor,
                                         d_X,
                                         d_kernel,
                                         &d_output_shape);

    // Create convolution output memory
    cudaMalloc(&d_output, output_d_output.totalBytes);

    // Launch kernel
    CUDA_KERNEL(0, "conv3d_s4") {
        conv3d_s4_kernel<<<1, 1>>>(d_X,
                                   d_kernel,
                                   cudnn_convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   d_output,
                                   output_d_output.totalBytes);
    }

    // Wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Compute average execution time
    double time = 0;
    for (int i = 0; i < repeat; i++) {
        CUDA_KERNEL(0, "conv3d_s4") {
            conv3d_s4_kernel<<<1, 1>>>(d_X,
                                       d_kernel,
                                       cudnn_convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       d_output,
                                       output_d_output.totalBytes);
        }
        cudaDeviceSynchronize();
    }
    time = (get_time() - start_time) / repeat;
    printf("Average kernel execution time of conv3d_s4 kernel: %f (us)\n", time * 1e-3 / repeat);

    // Clean up resources
    cudnnDestroyTensorDescriptor(d_X);
    cudnnDestroyTensorDescriptor(d_kernel);
    cudnnDestroyTensorDescriptor(d_Y);
    cudnnDestroyConvolutionDescriptor(cudnn_convolution_descriptor);
    cudaFree(d_workspace);
    cudaFree(d_output);

    return 0;
}