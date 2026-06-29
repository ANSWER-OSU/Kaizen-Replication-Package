#include <iostream>
#include <cstdlib>

#define K 20
#define N 1000
#define M 10

double* alphas = (double*)malloc(K*N*sizeof(double));
for (int i = 0; i < K*N; ++i) {
    alphas[i] = rand() % 100;
}

double* rands = (double*)malloc(K*M*sizeof(double));
for (int i = 0; i < K*M; ++i) {
    rands[i] = rand() % 100;
}

double* probs = (double*)malloc(K*N*sizeof(double));
#pragma omp parallel
for (int i = 0; i < K*N; ++i) {
    int m, k, maxind = -1;
    double maxval = 0, temp, alpha;

    for (k = 0; k < K; ++k) {
        probs[i] += temp = alphas[i*K + k] + rands[m*K + k];
        if (temp > maxval) {
            maxval = temp;
            maxind = k;
        }
    }

    probs[i] /= M;
}

#pragma omp parallel for
for (int i = 0; i < K*N; ++i) {
    int m, k, maxind = -1;
    double maxval = 0, temp, alpha;

    for (m = 0; m < M; ++m) {
        for (k = 0; k < K; ++k) {
            probs[i] += temp = alphas[k*N + i] + rands[k*M + m];
            if (temp > maxval) {
                maxval = temp;
                maxind = k;
            }
        }
    }

    probs[i] /= M;
}

#pragma omp parallel shared(probs) private(w)
for (int i = 0; i < K*N; ++i) {
    int m, k, maxind = -1;
    double maxval = 0, temp, alpha;
    double shared_probs[K*threadsPerBlock];
    double shared_w[K*threadsPerBlock];
    double* w = &shared_w[K*threadsPerBlock];

    for (k = 0; k < K; ++k) {
        w[k*threadsPerBlock + threadIdx.x] = alphas[k*N + i] + rands[k*M + m];
    }

    for (m = 0; m < M; ++m) {
        maxval = w[K*M + m];
        for (k = 0; k < (K-1); ++k) {
            if (w[k*M + m] > maxval) {
                maxval = w[k*M + m];
                maxind = k;
            }
        }
        shared_probs[maxind*threadsPerBlock + threadIdx.x] += 1.0;
    }

    for (k = 0; k < K; ++k) {
        probs[i] += temp = shared_probs[k*threadsPerBlock + threadIdx.x];
        if (temp > maxval) {
            maxval = temp;
            maxind = k;
        }
    }

    for (k = 0; k < K; ++k) {
        probs[k*n + i] = shared_probs[k*threadsPerBlock + threadIdx.x] / M;
    }
}