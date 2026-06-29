#ifndef __OPENMP_MACRO_H__
#define __OPENMP_MACRO_H__

#include <iostream>
#include <cstdlib>
#include <omp.h>

#define CHECK_CUDA(call) {                                                \
    cudaError_t err = call;                                               \
    if( cudaSuccess != err) {                                             \
        std::cerr << "CUDA error in file '" << __FILE__ << "' in line "    \
          << __LINE__ << " : " << cudaGetErrorString( err) << ".\n";     \
        exit(EXIT_FAILURE);                                               \
    }}

#define CHECK_CUBLAS(call) {                                              \
    cublasStatus_t status = call;                                         \
    if( CUBLAS_STATUS_SUCCESS != status) {                                \
        std::cerr << "CUBLAS error: " << #call << " = " << status         \
          << " at (" << __FILE__ << ":" << __LINE__ << ")\n";            \
        exit(EXIT_FAILURE);                                               \
    }}

#define CHECK_CURAND(call) {                                              \
    curandStatus_t status = call;                                         \
    if( CURAND_STATUS_SUCCESS != status) {                                \
        std::cerr << "CURAND error: " << #call << " = " << status         \
          << " at (" << __FILE__ << ":" << __LINE__ << ")\n";            \
        exit(EXIT_FAILURE);                                               \
    }}

#define CHECK_ERROR(errorMessage) {                                       \
    cudaError_t err = cudaGetLastError();                                 \
    if( cudaSuccess != err) {                                             \
        std::cerr << "Cuda error: " << errorMessage << " in file '"       \
          << __FILE__ << "' in line " << __LINE__ << " : "                \
          << cudaGetErrorString(err) << ".\n";                           \
        exit(EXIT_FAILURE);                                               \
    }}
#endif