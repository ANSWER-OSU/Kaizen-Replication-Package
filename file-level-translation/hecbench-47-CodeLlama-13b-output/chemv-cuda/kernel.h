#include <omp.h>
#include <cstdint>
#include <complex>

struct ComplexFloat {
    std::float_t Re;
    std::float_t Im;
};

__global__ void chemv_kernel0(ComplexFloat *AT, ComplexFloat *X, ComplexFloat *Y, std::float_t alpha_im, std::float_t alpha_re, std::float_t beta_im, std::float_t beta_re) {
    int tid = threadIdx.x;
    int tidy = blockIdx.x;
    int blockDim = blockDim.x;

    std::complex<float> alpha = std::complex<float>(alpha_re, alpha_im);
    std::complex<float> beta = std::complex<float>(beta_re, beta_im);
    std::complex<float> acc = beta;

    for (int j = 0; j < tidy; j++) {
        std::complex<float> val = 0.0;
        for (int i = 0; i < blockDim; i++) {
            val += AT[j * blockDim + tid] * X[tid * blockDim + i];
        }
        acc += val;
    }
    Y[tidy * blockDim + tid] = acc;
}

__global__ void chemv_kernel1(ComplexFloat *AT, ComplexFloat *X, ComplexFloat *Y, std::float_t alpha_im, std::float_t alpha_re) {
    int tid = threadIdx.x;
    int tidy = blockIdx.x;
    int blockDim = blockDim.x;

    std::complex<float> alpha = std::complex<float>(alpha_re, alpha_im);
    std::complex<float> acc = alpha;

    for (int j = 0; j < tidy; j++) {
        std::complex<float> val = 0.0;
        for (int i = 0; i < blockDim; i++) {
            val += AT[j * blockDim + tid] * X[tid * blockDim + i];
        }
        acc += val;
    }
    Y[tidy * blockDim + tid] = acc;
}

int main(int argc, char* argv[]) {
    uint64_t arraySize = 256 * 256;
    ComplexFloat *AT = (ComplexFloat*) malloc(sizeof(ComplexFloat) * arraySize);
    ComplexFloat *X = (ComplexFloat*) malloc(sizeof(ComplexFloat) * arraySize);
    ComplexFloat *Y = (ComplexFloat*) malloc(sizeof(ComplexFloat) * arraySize);

    std::cout << "Array size: " << arraySize << std::endl;

    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < arraySize; i++) {
        AT[i].Re = static_cast<float>(i);
        AT[i].Im = 0.0;
        X[i].Re = static_cast<float>(i);
        X[i].Im = 0.0;
        Y[i].Re = 0.0;
        Y[i].Im = 0.0;
    }

    std::cout << "Kernel 0: 1 threadblock x " << arraySize << " threads" << std::endl;
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < arraySize; i++) {
        chemv_kernel0(AT, X, Y, 1.0, 1.0, 0.0, 0.0);
    }
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < arraySize; i++) {
        Y[i].Im = 0.0;
    }

    std::cout << "Kernel 1: 1 threadblock x " << arraySize << " threads" << std::endl;
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < arraySize; i++) {
        chemv_kernel1(AT, X, Y, 1.0, 1.0);
    }
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < arraySize; i++) {
        Y[i].Im = 0.0;
    }
}