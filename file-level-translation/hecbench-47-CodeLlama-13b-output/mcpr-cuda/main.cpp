#include <chrono>
#include <cmath>
#include <random>
#include <iostream>
#include <cstdio>
#include <omp.h>
#include "kernels.h"
#include "reference.h"

// transpose
double* t(const double *idata, const int width, const int height) {
    double *odata = (double*) malloc (sizeof(double) * width * height);
    for (int yIndex = 0; yIndex < height; yIndex++) {
        for (int xIndex = 0; xIndex < width; xIndex++) {
            int index_in  = xIndex + width * yIndex;
            int index_out = yIndex + height * xIndex;
            odata[index_out] = idata[index_in];
        }
    }
    return odata;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <path to filename> <repeat>\n", argv[0]);
        return 1;
    }
    char *filename = argv[1];
    const int repeat = atoi(argv[2]);

    // n and K should match the dimension of the dataset in the csv file
    const int n = 26280, K = 21, M = 10000;

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error: failed to open file alphas.csv. Exit\n");
        return 1;
    }

    int alphas_size = n * K; // n rows and K cols
    int alphas_size_byte = n * K * sizeof(double);

    int rands_size = M * K;  // M rows and K cols
    int rands_size_byte = M * K * sizeof(double);

    double *alphas, *rands, *probs, *probs_ref;
    alphas = (double*) malloc (alphas_size_byte);
    rands = (double*) malloc (rands_size_byte);
    probs = (double*) malloc (alphas_size_byte);
    probs_ref = (double*) malloc (alphas_size_byte);

    // load the csv file
    for (int i = 0; i < alphas_size; i++)
        fscanf(fp, "%lf", &alphas[i]);
    fclose(fp);

    // normal distribution (mean: 0 and var: 1)
    std::mt19937 gen(19937);
    std::normal_distribution<double> norm_dist(0.0,1.0);
    for (int i = 0; i < rands_size; i++) rands[i] = norm_dist(gen);

    reference(alphas, rands, probs_ref, n, K, M);

    double *t_rands = t(rands, K, M);
    double *t_alphas = t(alphas, K, n);

    // kernel 1
    int threads_per_block = 192;
    const int nblocks = ceil(1.0 * n / threads_per_block);

    parallel {
        for (int i = 0; i < repeat; i++) {
            compute_probs(alphas, rands, probs, n, K, M);
        }
    }

    std::cout << "Average execution time of compute_probs kernel: " << (time * 1e-9f) / repeat << " (s)" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nblocks; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                probs[i*threads_per_block + j*nblocks*threads_per_block + k] = probs[i*threads_per_block + j*nblocks*threads_per_block + k] + probs_ref[i*threads_per_block + j*nblocks*threads_per_block + k];
            }
        }
    }

    // kernel 2
    const int sm_size = sizeof(double) * K * threads_per_block * 2;

    parallel {
        for (int i = 0; i < repeat; i++) {
            compute_probs_unitStrides_sharedMem(alphas, rands, probs, n, K, M);
        }
    }

    std::cout << "Average execution time of compute_probs_unitStrides_sharedMem kernel: " << (time * 1e-9f) / repeat << " (s)" << std::endl;    

    #pragma omp parallel for
    for (int i = 0; i < nblocks; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                // compute_probs_unitStrides_sharedMem(alphas+i*threads_per_block + j*nblocks*threads_per_block, rands + k*K + j*threads_per_block, probs + k*K + j*threads_per_block, K, threads_per_block);
                probs[i * threads_per_block + j * nblocks * threads_per_block + k] = probs[i * threads_per_block + j * nblocks * threads_per_block + k] + probs_ref[i * threads_per_block + j * nblocks * threads_per_block + k];
            }
        }
    }

    // kernel 3
    double *transposed_rands = t(rands, K, M);
    double *transposed_alphas = t(alphas, K, n);

    parallel {
        for (int i = 0; i < repeat; i++) {
            compute_probs_transposed(transposed_alphas, transposed_rands, probs, n, K, M);
        }
    }

    std::cout << "Average execution time of compute_probs_transposed kernel: " << (time * 1e-9f) / repeat << " (s)" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nblocks; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                // compute_probs_transposed(transposed_alphas + i*threads_per_block + j*nblocks*threads_per_block, transposed_rands + k*K + j*threads_per_block, probs + k*K + j*threads_per_block, K, threads_per_block);
                probs[i * threads_per_block + j * nblocks * threads_per_block + k] = probs[i * threads_per_block + j * nblocks * threads_per_block + k] + probs_ref[i * threads_per_block + j * nblocks * threads_per_block + k];
            }
        }
    }

    // free memory
    free(alphas);
    free(rands);
    free(transposed_rands);
    free(transposed_alphas);
    free(probs);
    free(probs_ref);
    return 0;
}