#include <math.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#define POLYNOMIAL_DEGREE  7
#define p_Nggeo 7
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6
#define p_GWJID 0

#include "meshBasis.hpp"

// cpu reference 
#include "axhelmReference.cpp"

// gpu kernel
#include "axhelmKernel.cpp"

//=========================================================================
// End of kernel
//=========================================================================

float *frandAlloc(int Nelem) {
  float *v = (float*)calloc(Nelem, sizeof(float));

  for (int n = 0; n < Nelem; ++n) {
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: ./axhelm Ndim numElements [nRepetitions]\n");
    return 1;
  }

  const int Ndim = atoi(argv[1]);
  const int Nelements = atoi(argv[2]);
  int Ntests = 1;
  if (argc >= 4)
    Ntests = atoi(argv[3]);

  const int Nq = POLYNOMIAL_DEGREE + 1;
  const int Np = Nq * Nq * Nq;
  const int offset = Nelements * Np;

  // build element nodes and operators
  float *rV, *wV, *DrV;
  meshJacobiGQ(0, 0, POLYNOMIAL_DEGREE, &rV, &wV);
  meshDmatrix1D(POLYNOMIAL_DEGREE, Nq, rV, &DrV);

  std::cout << "word size: " << sizeof(float) << " bytes\n";

  // populate device arrays
  float *ggeo = frandAlloc(Np * Nelements * p_Nggeo);
  float *q = frandAlloc((Ndim * Np) * Nelements);
  float *Aq = frandAlloc((Ndim * Np) * Nelements);

  const float lambda1 = 1.1;
  float *lambda = (float*)calloc(2 * offset, sizeof(float));
  for (int i = 0; i < offset; i++) {
    lambda[i] = 1.0;
    lambda[i + offset] = lambda1;
  }

  // compute the reference result
  for (int n = 0; n < Ndim; ++n) {
    float *x = q + n * offset;
    float *Ax = Aq + n * offset;
    axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, x, Ax);
  }

  #pragma omp parallel for
  for (int element = 0; element < Nelements; element++) {
    for (int dim = 0; dim < Ndim; dim++) {
      if (Ndim > 1)
        axhelm_n3(
            Nelements, offset, ggeo, DrV, lambda, q, Aq,
            element, dim);
      else
        axhelm(
            Nelements, offset, ggeo, DrV, lambda, q, Aq,
            element, dim);
    }
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int test = 0; test < Ntests; ++test) {
  #pragma omp parallel for
    for (int element = 0; element < Nelements; element++) {
      for (int dim = 0; dim < Ndim; dim++) {
        if (Ndim > 1)
          axhelm_n3(
              Nelements, offset, ggeo, DrV, lambda, q, Aq,
              element, dim);
        else
          axhelm(
              Nelements, offset, ggeo, DrV, lambda, q, Aq,
              element, dim);
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

  free(ggeo);
  free(q);
  free(Aq);
  free(lambda);
  free(rV);
  free(wV);
  free(DrV);

  // print statistics
  const float GDOFPerSecond = Ndim * POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE * Nelements / elapsed;
  const long long bytesMoved = (Ndim * 2 * Np + 7 * Np + 2 * Np) * sizeof(float); // x, Mx, opa, lambda
  const double bw = bytesMoved * Nelements / elapsed;
  double flopCount = Ndim * Np * 12 * Nq;
  if (Ndim == 1) flopCount += 22 * Np;
  if (Ndim == 3) flopCount += 69 * Np;
  double gflops = flopCount * Nelements / elapsed;
  std::cout << " NRepetitions=" << Ntests
    << " Ndim=" << Ndim
    << " N=" << POLYNOMIAL_DEGREE
    << " Nelements=" << Nelements
    << " elapsed time=" << elapsed
    << " GDOF/s=" << GDOFPerSecond
    << " GB/s=" << bw
    << " GFLOPS/s=" << gflops
    << "\n";
  return 0;
}