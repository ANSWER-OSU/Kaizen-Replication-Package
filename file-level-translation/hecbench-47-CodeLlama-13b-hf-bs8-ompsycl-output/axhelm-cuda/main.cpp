// CUDA translation of axhelmReference kernel and axhelm kernel


#include <iostream>
#include <chrono>
#include <cuda.h>
#include <omp.h>

void axhelmReference(const int Nq, const int Nelements, const dfloat lambda1, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  for (int e = 0; e < Nelements; ++e) {
    for (int i = 0; i < Nq; ++i) {
      for (int j = 0; j < Nq; ++j) {
        for (int k = 0; k < Nq; ++k) {
          int id = e * Nq * Nq * Nq + i * Nq * Nq + j * Nq + k;
          dfloat r_G00 = ggeo[id + Nq * Nq * Nq * p_G00ID];
          dfloat r_G01 = ggeo[id + Nq * Nq * Nq * p_G01ID];
          dfloat r_G02 = ggeo[id + Nq * Nq * Nq * p_G02ID];
          dfloat r_G11 = ggeo[id + Nq * Nq * Nq * p_G11ID];
          dfloat r_G12 = ggeo[id + Nq * Nq * Nq * p_G12ID];
          dfloat r_G22 = ggeo[id + Nq * Nq * Nq * p_G22ID];
          dfloat r_D = ggeo[id + Nq * Nq * Nq * p_GWJID];
          dfloat p_r_D = r_D * Dr[i * Nq + j];
          dfloat U00 = q[id + Nq * Nq * Nq * 0] * r_G00 + q[id + Nq * Nq * Nq * 1] * r_G01 + q[id + Nq * Nq * Nq * 2] * r_G02;
          dfloat U10 = q[id + Nq * Nq * Nq * 1] * r_G11 + q[id + Nq * Nq * Nq * 2] * r_G12;
          dfloat U20 = q[id + Nq * Nq * Nq * 2] * r_G22;
          dfloat U01 = U10;
          dfloat U11 = U00;
          dfloat U21 = 0;
          dfloat U02 = U20;
          dfloat U12 = 0;
          dfloat U22 = U00;
          Aq[id + Nq * Nq * Nq * 0] = p_r_D * (U00 * r_G00 + U01 * r_G01 + U02 * r_G02);
          Aq[id + Nq * Nq * Nq * 1] = p_r_D * (U10 * r_G00 + U11 * r_G11 + U12 * r_G12);
          Aq[id + Nq * Nq * Nq * 2] = p_r_D * (U20 * r_G00 + U21 * r_G11 + U22 * r_G22);
        }
      }
    }
  }
}

__global__ void axhelm_n3(const int Nelements, const int offset, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ lambda, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  __shared__ dfloat s_c[8][8];
  __shared__ dfloat s_g[8][8];
  int i = threadIdx.x;
  int j = threadIdx.y;
  int id = j * 8 + i;
  s_c[j][i] = Dr[id];
  s_g[j][i] = 1.f;
  for(int n = 2; n < Nq; n++) {
    int ii = i % 2 * n + (i / 2) * 2 * n + n * n;
    int jj = j % 2 * n + (j / 2) * 2 * n + n * n;
    dfloat r_c = Dr[jj * Nq + ii];
    dfloat r_g = 1.f;
    for(int l = 0; l < Nq; l++) {
      int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
      r_g *= lambda[id];
    }
    s_c[j][i] += r_c;
    s_g[j][i] *= r_g;
  }
  __syncthreads();
  for(int n = 2; n < Nq; n++) {
    int ii = i % 2 * n + (i / 2) * 2 * n + n * n;
    int jj = j % 2 * n + (j / 2) * 2 * n + n * n;
    dfloat r_c = Dr[jj * Nq + ii];
    dfloat r_g = 1.f;
    for(int l = 0; l < Nq; l++) {
      int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
      r_g *= lambda[id];
    }
    s_c[j][i] += r_c;
    s_g[j][i] *= r_g;
  }
  __syncthreads();
  if (i < Nq && j < Nq) {
    int id = e * Nq * Nq * Nq + j * Nq + i;
    dfloat r_c = s_c[j][i];
    dfloat r_g = s_g[j][i];
    for (int k = 0; k < Nq; ++k) {
      int idr = id + k * Nq * Nq * Nq;
      dfloat r_Aq = 0;
      for (int l = 0; l < Nq; ++l) {
        int idl = idr + l * Nq * Nq + k * Nq + l;
        r_Aq += r_c * q[idl] * r_g;
        r_g *= r_lambda[idl];
      }
      Aq[idr] = r_Aq;
    }
  }
}

__global__ void axhelm(const int Nelements, const int offset, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ lambda, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  int i = threadIdx.x;
  int j = threadIdx.y;
  int id = j * Nq + i;
  dfloat r_c = Dr[id];
  dfloat r_g = 1.f;
  for (int l = 0; l < Nq; l++) {
    int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
    r_g *= r_lambda[id];
  }
  for(int n = 2; n < Nq; n++) {
    int ii = i % 2 * n + (i / 2) * 2 * n + n * n;
    int jj = j % 2 * n + (j / 2) * 2 * n + n * n;
    r_c += Dr[jj * Nq + ii];
    for (int l = 0; l < Nq; l++) {
      int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
      r_g *= r_lambda[id];
    }
  }
  if (i < Nq && j < Nq) {
    int id = e * Nq * Nq * Nq + j * Nq + i;
    dfloat r_Aq = 0;
    for (int k = 0; k < Nq; ++k) {
      int idr = id + k * Nq * Nq;
      for (int l = 0; l < Nq; ++l) {
        int idl = idr + l * Nq + k * Nq + l;
        r_Aq += r_c * q[idl] * r_g;
        r_g *= r_lambda[idl];
      }
    }
    Aq[id] = r_Aq;
  }
}

void axhelm_parallel(const int Nq, const int numElements, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ lambda, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numElements) num_threads(Nq * Nq)
  for(int e = 0; e < numElements; e++) {
    for(int i = 0; i < Nq; i++) {
      for(int j = 0; j < Nq; j++) {
        int id = e * Nq * Nq + j * Nq + i;
        dfloat r_c = Dr[id];
        dfloat r_g = 1.f;
        for(int l = 0; l < Nq; l++) {
          int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
          r_g *= lambda[id];
        }
        for(int n = 2; n < Nq; n++) {
          int ii = i % 2 * n + (i / 2) * 2 * n + n * n;
          int jj = j % 2 * n + (j / 2) * 2 * n + n * n;
          r_c += Dr[jj * Nq + ii];
          for(int l = 0; l < Nq; l++) {
            int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
            r_g *= lambda[id];
          }
        }
        if(i < Nq && j < Nq) {
          int id = e * Nq * Nq + j * Nq + i;
          dfloat r_Aq = 0;
          for(int k = 0; k < Nq; ++k) {
            int idr = id + k * Nq * Nq;
            for(int l = 0; l < Nq; ++l) {
              int idl = idr + l * Nq + k * Nq + l;
              r_Aq += r_c * q[idl] * r_g;
              r_g *= lambda[idl];
            }
          }
          Aq[id] = r_Aq;
        }
      }
    }
  }
}

__global__ void axhelm_1(const int Nelements, const int Nq, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ lambda, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  __shared__ dfloat s_c[8][8];
  __shared__ dfloat s_g[8][8];
  int i = threadIdx.x;
  int j = threadIdx.y;
  int id = j * 8 + i;
  s_c[j][i] = Dr[id];
  s_g[j][i] = 1.f;
  for(int n = 2; n < Nq; n++) {
    int ii = i % 2 * n + (i / 2) * 2 * n + n * n;
    int jj = j % 2 * n + (j / 2) * 2 * n + n * n;
    dfloat r_c = Dr[jj * Nq + ii];
    dfloat r_g = 1.f;
    for(int l = 0; l < Nq; l++) {
      int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
      r_g *= r_lambda[id];
    }
    s_c[j][i] += r_c;
    s_g[j][i] *= r_g;
  }
  __syncthreads();
  if (i < Nq && j < Nq) {
    int id = e * Nq * Nq + j * Nq + i;
    dfloat r_c = s_c[j][i];
    dfloat r_g = s_g[j][i];
    dfloat r_Aq = 0;
    for (int l = 0; l < Nq; ++l) {
      int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
      r_Aq += r_c * q[id] * r_g;
      r_g *= r_lambda[id];
    }
    Aq[id] = r_Aq;
  }
}

void axhelm_1_parallel(const int Nq, const int numElements, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ lambda, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  #pragma omp target teams distribute parallel for num_teams(numElements) num_threads(Nq * Nq)
  for(int e = 0; e < numElements; e++) {
    for(int i = 0; i < Nq; i++) {
      for(int j = 0; j < Nq; j++) {
        int id = e * Nq * Nq + j * Nq + i;
        dfloat r_c = Dr[id];
        dfloat r_g = 1.f;
        for(int l = 0; l < Nq; l++) {
          int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
          r_g *= lambda[id];
        }
        for(int n = 2; n < Nq; n++) {
          int ii = i % 2 * n + (i / 2) * 2 * n + n * n;
          int jj = j % 2 * n + (j / 2) * 2 * n + n * n;
          r_c += Dr[jj * Nq + ii];
          for(int l = 0; l < Nq; l++) {
            int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
            r_g *= lambda[id];
          }
        }
        if(i < Nq && j < Nq) {
          int id = e * Nq * Nq + j * Nq + i;
          dfloat r_Aq = 0;
          for (int l = 0; l < Nq; ++l) {
            int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
            r_Aq += r_c * q[id] * r_g;
            r_g *= lambda[id];
          }
          Aq[id] = r_Aq;
        }
      }
    }
  }
}

dfloat axhelmSerial(const int Nq, const int numElements, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ lambda, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  auto start = std::chrono::high_resolution_clock::now();
  for(int e = 0; e < numElements; e++) {
    for(int i = 0; i < Nq; i++) {
      for(int j = 0; j < Nq; j++) {
        int id = e * Nq * Nq + j * Nq + i;
        dfloat r_c = Dr[id];
        dfloat r_g = 1.f;
        for(int l = 0; l < Nq; l++) {
          int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
          r_g *= lambda[id];
        }
        for(int n = 2; n < Nq; n++) {
          int ii = i % 2 * n + (i / 2) * 2 * n + n * n;
          int jj = j % 2 * n + (j / 2) * 2 * n + n * n;
          r_c += Dr[jj * Nq + ii];
          for(int l = 0; l < Nq; l++) {
            int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
            r_g *= lambda[id];
          }
        }
        if(i < Nq && j < Nq) {
          int id = e * Nq * Nq + j * Nq + i;
          dfloat r_Aq = 0;
          for (int l = 0; l < Nq; ++l) {
            int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
            r_Aq += r_c * q[id] * r_g;
            r_g *= lambda[id];
          }
          Aq[id] = r_Aq;
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

dfloat axhelmOpenMP(const int Nq, const int numElements, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ lambda, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp target teams distribute parallel for num_teams(numElements) num_threads(Nq * Nq)
  for(int e = 0; e < numElements; e++) {
    for(int i = 0; i < Nq; i++) {
      for(int j = 0; j < Nq; j++) {
        int id = e * Nq * Nq + j * Nq + i;
        dfloat r_c = Dr[id];
        dfloat r_g = 1.f;
        for(int l = 0; l < Nq; l++) {
          int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
          r_g *= lambda[id];
        }
        for(int n = 2; n < Nq; n++) {
          int ii = i % 2 * n + (i / 2) * 2 * n + n * n;
          int jj = j % 2 * n + (j / 2) * 2 * n + n * n;
          r_c += Dr[jj * Nq + ii];
          for(int l = 0; l < Nq; l++) {
            int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
            r_g *= lambda[id];
          }
        }
        if(i < Nq && j < Nq) {
          int id = e * Nq * Nq + j * Nq + i;
          dfloat r_Aq = 0;
          for (int l = 0; l < Nq; ++l) {
            int id = e * Nq * Nq * Nq + l * Nq * Nq + j * Nq + i;
            r_Aq += r_c * q[id] * r_g;
            r_g *= lambda[id];
          }
          Aq[id] = r_Aq;
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

dfloat axhelmCUDA(const int Nq, const int numElements, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ Dr, const dfloat * __restrict__ lambda, const dfloat * __restrict__ q, dfloat * __restrict__ Aq)
{
  dfloat *o_ggeo, *o_q, *o_Aq, *o_Dr, *o_lambda;
  cudaMalloc((void**) &o_ggeo, Nq * Nq * Nq * p_Nggeo * sizeof(dfloat));
  cudaMalloc((void**) &o_q, Nq * Nq * Nq * sizeof(dfloat));
  cudaMalloc((void**) &o_Aq, Nq * Nq * Nq * sizeof(dfloat));
  cudaMalloc((void**) &o_Dr, Nq * Nq * sizeof(dfloat));
  cudaMalloc((void**) &o_lambda, Nq * Nq * Nq * sizeof(dfloat));
  cudaMemcpy(o_ggeo, ggeo, Nq * Nq * Nq * p_Nggeo * sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_q, q, Nq * Nq * Nq * sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_Dr, Dr, Nq * Nq * sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_lambda, lambda, Nq * Nq * Nq * sizeof(dfloat), cudaMemcpyHostToDevice);
  auto start = std::chrono::high_resolution_clock::now();
  axhelm_1 <<< numElements, Nq * Nq >>> (Nq, numElements, o_ggeo, o_Dr, o_lambda, o_q, o_Aq);
  cudaMemcpy(Aq, o_Aq, Nq * Nq * Nq * sizeof(dfloat), cudaMemcpyDeviceToHost);
  cudaFree(o_ggeo);
  cudaFree(o_q);
  cudaFree(o_Aq);
  cudaFree(o_Dr);
  cudaFree(o_lambda);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char **argv)
{
  if (argc<3) {
    printf("Usage: ./axhelm Ndim numElements\n");
    return 1;
  }
  const int Ndim = atoi(argv[1]);
  const int Nq = atoi(argv[2]);
  const int numElements = 10;
  const int Np = Nq*Nq*Nq;
  dfloat *ggeo = (dfloat*) calloc(Np*p_Nggeo, sizeof(dfloat));
  for(int n = 0; n < Np*p_Nggeo; n++) ggeo[n] = 1;
  dfloat *q = (dfloat*) calloc(Np, sizeof(dfloat));
  for(int n = 0; n < Np; n++) q[n] = 1;
  dfloat *Aq = (dfloat*) calloc(Np, sizeof(dfloat));
  for(int n = 0; n < Np; n++) Aq[n] = 0;
  dfloat *Dr = (dfloat*) calloc(Nq*Nq, sizeof(dfloat));
  dfloat *lambda = (dfloat*) calloc(Np*numElements, sizeof(dfloat));
  for(int n = 0; n < Np*numElements; n++) lambda[n] = 1;
  const dfloat lambda1 = 1.1;
  for(int e = 0; e < numElements; e++) {
    for(int n = 0; n < Np; n++) {
      int id = e * Np + n;
      lambda[id] = lambda1;
    }
  }
  dfloat serial = axhelmSerial(Nq, numElements, ggeo, Dr, lambda, q, Aq);
  dfloat OpenMP = axhelmOpenMP(Nq, numElements, ggeo, Dr, lambda, q, Aq);
  dfloat CUDA = axhelmCUDA(Nq, numElements, ggeo, Dr, lambda, q, Aq);
  std::cout << "serial time: " << serial << " OpenMP time: " << OpenMP << " CUDA time: " << CUDA << "\n";
  free(ggeo);
  free(q);
  free(Aq);
  free(Dr);
  free(lambda);
  return 0;
}