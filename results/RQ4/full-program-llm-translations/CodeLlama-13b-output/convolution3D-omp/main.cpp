#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENMP
#define CHUNKS 4

// openmp version of conv3d_s1 kernel
template <typename T>
void conv3d_s1(const T * __restrict__ X, const T * __restrict__ W, T * __restrict__ Y, const int N, const int M, const int C, const int K, const int H, const int W, const int W_grid) {
#pragma omp parallel for collapse(3)
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      for (int h = 0; h < H; h += W_grid) {
        for (int w = 0; w < W; w += W_grid) {
          for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
              for (int q = 0; q < K; q++) {
                Y[OI(n, m, h, w)] = X[II(n, c, h + p, w + q)] * W[WI(m, c, p, q)];
              }
            }
          }
        }
      }
    }
  }
}

// openmp version of conv3d_s2 kernel
template <typename T>
void conv3d_s2(const T * __restrict__ X, const T * __restrict__ W, T * __restrict__ Y, const int N, const int M, const int C, const int K, const int H, const int W, const int W_grid) {
#pragma omp parallel for collapse(3)
  for (int m = 0; m < M; m++) {
    for (int h = 0; h < H; h += W_grid) {
      for (int w = 0; w < W; w += W_grid) {
        for (int n = 0; n < N; n++) {
          for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
              for (int q = 0; q < K; q++) {
                Y[OI(n, m, h, w)] = X[II(n, c, h + p, w + q)] * W[WI(m, c, p, q)];
              }
            }
          }
        }
      }
    }
  }
}

// openmp version of conv3d_s3 kernel
template <typename T>
void conv3d_s3(const T * __restrict__ X, const T * __restrict__ W, T * __restrict__ Y, const int N, const int M, const int C, const int K, const int H, const int W, const int W_grid) {
#pragma omp parallel for collapse(3)
  for (int h = 0; h < H; h += W_grid) {
    for (int w = 0; w < W; w += W_grid) {
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
          for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
              for (int q = 0; q < K; q++) {
                Y[OI(n, m, h, w)] = X[II(n, c, h + p, w + q)] * W[WI(m, c, p, q)];
              }
            }
          }
        }
      }
    }
  }
}

#endif

int main(int argc, char *argv[]) {
  if (argc != 8) {
    printf("Usage: %s batchsize input_channels output_channels input_size kernel_size repeat\n", argv[0]);
    return 1;
  }

  int N = atoi(argv[1]);
  int C = atoi(argv[2]);
  int M = atoi(argv[3]);
  int H = atoi(argv[4]);
  int W = atoi(argv[5]);
  int K = atoi(argv[6]);
  int repeat = atoi(argv[7]);

  float *X, *W, *Y, *Y_ref;
  X = (float *)malloc(N * C * H * W * sizeof(float)); // input
  W = (float *)malloc(M * C * K * K * sizeof(float)); // filter
  Y = (float *)malloc(N * M * H * W * sizeof(float)); // output
  Y_ref = (float *)malloc(N * M * H * W * sizeof(float));

  srand(123);

  for (int i = 0; i < M * C * K * K; i++)
    W[i] = rand() % 31;
  for (int i = 0; i < N * C * H * W; i++)
    X[i] = rand() % 13;

  for (int i = 0; i < N * M * H * W; i++) {
    Y[i] = -1;
    Y_ref[i] = -1;
  }

  reference(X, W, Y_ref, N, M, C, K, H, W, H, W);

#ifdef _OPENMP
  for (int i = 0; i < 100; i++) {
    conv3d_s1<float>(X, W, Y, N, M, C, K, H, W, 256, 256);
    cudaDeviceSynchronize();
    cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
    verify(Y, Y_ref, Y_size);
  }
#endif

  return 0;
}