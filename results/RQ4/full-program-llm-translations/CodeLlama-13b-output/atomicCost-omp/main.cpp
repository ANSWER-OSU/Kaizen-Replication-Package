#include <iostream>
#include <cassert>
#include <chrono>

#include <omp.h>

static void CheckError( int err, const char *file, int line ){
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    }
}

#define CHECK_ERROR(err) (CheckError(err, __FILE__, __LINE__))

template<typename T>
__global__ void woAtomicOnGlobalMem(T* result, int size){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned int i=tid*size; i<(tid+1)*size; i++){
        result[tid] += i%2;
    }
}

template<typename T>
__global__ void wiAtomicOnGlobalMem(T* result, int size){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned int i=tid*size; i<(tid+1)*size; i++){
        atomicAdd(&result[tid], i%2);
    }
}

template<typename T>
void atomicCost(int length, int size, int repeat){
    std::cout << std::endl;
    std::cout << "Each thread sums up " << size << " elements" << std::endl;

    int num_threads = length / size;
    assert(length % size == 0);
    assert(num_threads % BLOCK_SIZE == 0);

    size_t result_size = sizeof(T) * num_threads;

    T* result_wi = (T*) malloc(result_size);
    T* result_wo = (T*) malloc(result_size);

    T* d_result_wi, *d_result_wo;
    CHECK_ERROR( cudaMalloc((void **)&d_result_wi, result_size) );
    CHECK_ERROR( cudaMemset(d_result_wi, 0, result_size) );
    CHECK_ERROR( cudaMalloc((void **)&d_result_wo, result_size) );
    CHECK_ERROR( cudaMemset(d_result_wo, 0, result_size) );

    dim3 block (BLOCK_SIZE);
    dim3 grid (num_threads / BLOCK_SIZE);

    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for(int i=0; i<repeat; i++){
        wiAtomicOnGlobalMem<T><<<grid, block>>>(d_result_wi, size);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average execution time of WithAtomicOnGlobalMem: " << time * 1e-3f / repeat << " (us)" << std::endl;
    CHECK_ERROR( cudaMemcpy(result_wi, d_result_wi, result_size, cudaMemcpyDeviceToHost) );

    start = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for(int i=0; i<repeat; i++){
        woAtomicOnGlobalMem<T><<<grid, block>>>(d_result_wo, size);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average execution time of WithoutAtomicOnGlobalMem: " << time * 1e-3f / repeat << " (us)" << std::endl;
    CHECK_ERROR( cudaMemcpy(result_wo, d_result_wo, result_size, cudaMemcpyDeviceToHost) );

    int diff = memcmp(result_wi, result_wo, result_size);
    std::cout << (diff ? "FAIL" : "PASS") << std::endl;

    free(result_wi);
    free(result_wo);
    cudaFree(d_result_wi);
    cudaFree(d_result_wo);
}

int main(int argc, char* argv[]) {
    if(argc != 3){
        std::cout << "Usage: " << argv[0] << " <N> <repeat>" << std::endl;
        std::cout << "N: the number of elements to sum per thread (1 - 16)" << std::endl;
        return 1;
    }
    const int nelems = atoi(argv[1]);
    const int repeat = atoi(argv[2]);

    const int length = 922521600;
    assert(length % BLOCK_SIZE == 0);

    std::cout << std::endl << "FP64 atomic add" << std::endl;
    atomicCost<double>(length, nelems, repeat);

    std::cout << std::endl << "INT32 atomic add" << std::endl;
    atomicCost<int>(length, nelems, repeat);

    std::cout << std::endl << "FP32 atomic add" << std::endl;
    atomicCost<float>(length, nelems, repeat);

    return 0;
}