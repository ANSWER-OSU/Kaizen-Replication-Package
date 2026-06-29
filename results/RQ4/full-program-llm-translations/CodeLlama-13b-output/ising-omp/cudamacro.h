#ifndef __OPENMP_MACRO_H__
#define __OPENMP_MACRO_H__

#include <omp.h>

#define _Pragma(...) __VA_ARGS__

#define __align__(...) __VA_ARGS__

#define __global__(...) __VA_ARGS__

#define __launch_bounds__(...) __VA_ARGS__

#define __shared__ __VA_ARGS__

#define __device__ __VA_ARGS__

#define __host__ __VA_ARGS__

#define __forceinline__ __VA_ARGS__

#define __globalLaunchBounds__(...) __VA_ARGS__

#define __global_atomic__(...) __VA_ARGS__

#define __global_hermit__(...) __VA_ARGS__

#define __global_matrix_trans__(...) __VA_ARGS__

#define __global_bitwise__(...) __VA_ARGS__

#define __global_bus_router__(...) __VA_ARGS__

#define __global_pin_vl__(...) __VA_ARGS__

#define __global_current_vl__(...) __VA_ARGS__

#define __ora_err_check__(...) __VA_ARGS__

#define __ora_err_msg__(...) __VA_ARGS__

#define __ora_err_exit__(...) __VA_ARGS__

#define __ora_err_ret__(...) __VA_ARGS__

#define __cuda_err_chk__(...) __VA_ARGS__

#define __cuda_err_msg__(...) __VA_ARGS__

#define __cuda_err_exit__(...) __VA_ARGS__

#define __cuda_err_ret__(...) __VA_ARGS__

#define __cublas_err_chk__(...) __VA_ARGS__

#define __cublas_err_msg__(...) __VA_ARGS__

#define __cublas_err_exit__(...) __VA_ARGS__

#define __cublas_err_ret__(...) __VA_ARGS__

#define __curand_err_chk__(...) __VA_ARGS__

#define __curand_err_msg__(...) __VA_ARGS__

#define __curand_err_exit__(...) __VA_ARGS__

#define __curand_err_ret__(...) __VA_ARGS__

#define __err_chk__(...) __VA_ARGS__

#define __err_msg__(...) __VA_ARGS__

#define __err_exit__(...) __VA_ARGS__

#define __err_ret__(...) __VA_ARGS__

#endif