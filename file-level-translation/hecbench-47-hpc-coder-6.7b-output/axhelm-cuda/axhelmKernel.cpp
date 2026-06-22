#include <openmp.h>

void axhelm(const int Nelements,
    const int offset,
    const double * __restrict__ ggeo,
    const double * __restrict__ D,
    const double * __restrict__ lambda,
    const double * __restrict__ Q,
    double * __restrict__ Aq) 
{
  #pragma omp parallel
  {
    #pragma omp for collapse(2)
    for (int e = 0; e < Nelements; ++e) {
      for (int j = 0; j < 8; ++j) {
        int i;
        #pragma omp simd
        for (i = 0; i < 8; ++i) {
          const int id = e * 512 + j * 8 * 8 + i;
          Aq[id] = 0;
          const int base = i + j * 8;
          const int baseQ = e * 8 * 8 + base;
          const int baseG = e * p_Nggeo * 512 + base;
          #pragma omp simd
          for (int k = 0; k < 8; ++k) {
            Q[id + k * 8];
          }
          #pragma omp simd
          for (int k = 0; k < 8; ++k) {
            const int idG = baseG + k * 8 * 8;
            ggeo[idG + p_G00ID * 512];
            ggeo[idG + p_G01ID * 512];
            ggeo[idG + p_G02ID * 512];
            ggeo[idG + p_G11ID * 512];
            ggeo[idG + p_G12ID * 512];
            ggeo[idG + p_G22ID * 512];
            ggeo[idG + p_GWJID * 512];
            lambda[id + 0 * offset];
            lambda[id + 1 * offset];
          }
        }
      }
    }
  }
}

void axhelm_n3(const int Nelements,
    const int offset,
    const double * __restrict__ ggeo,
    const double * __restrict__ D,
    const double * __restrict__ lambda,
    const double * __restrict__ Q,
    double * __restrict__ Aq) 
{
  #pragma omp parallel
  {
    #pragma omp for collapse(2)
    for (int e = 0; e < Nelements; ++e) {
      for (int j = 0; j < 8; ++j) {
        int i;
        #pragma omp simd
        for (i = 0; i < 8; ++i) {
          const int id = e * 512 + j * 8 * 8 + i;
          Aq[id] = 0;
          const int base = i + j * 8;
          const int baseQ = e * 8 * 8 + base;
          const int baseG = e * p_Nggeo * 512 + base;
          #pragma omp simd
          for (int k = 0; k < 8; ++k) {
            Q[id + k * 8 + offset * 0];
            Q[id + k * 8 + offset * 1];
            Q[id + k * 8 + offset * 2];
          }
          #pragma omp simd
          for (int k = 0; k < 8; ++k) {
            const int idG = baseG + k * 8 * 8;
            ggeo[idG + p_G00ID * 512];
            ggeo[idG + p_G01ID * 512];
            ggeo[idG + p_G02ID * 512];
            ggeo[idG + p_G11ID * 512];
            ggeo[idG + p_G12ID * 512];
            ggeo[idG + p_G22ID * 512];
            ggeo[idG + p_GWJID * 512];
            lambda[id + 0 * offset];
            lambda[id + 1 * offset];
          }
        }
      }
    }
  }
}