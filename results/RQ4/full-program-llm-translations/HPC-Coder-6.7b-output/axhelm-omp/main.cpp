#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>

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

dfloat *drandAlloc(int Nelem){

  dfloat *v = (dfloat*) calloc(Nelem, sizeof(dfloat));

  for(int n=0;n<Nelem;++n){
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv){

  if (argc<4) {
    printf("Usage: ./axhelm Ndim numElements [nRepetitions]\n");
    return 1;
  }

  const int Ndim = atoi(argv[1]);
  const int Nelements = atoi(argv[2]);
  int Ntests = 1;
  if(argc>=4)
    Ntests = atoi(argv[3]);

  const int Nq = POLYNOMIAL_DEGREE + 1;
  const int Np = Nq*Nq*Nq;
  const int offset = Nelements*Np;


  // build element nodes and operators
  dfloat *rV, *wV, *DrV;
  meshJacobiGQ(0,0,POLYNOMIAL_DEGREE, &rV, &wV);
  meshDmatrix1D(POLYNOMIAL_DEGREE, Nq, rV, &DrV);

  std::cout << "word size: " << sizeof(dfloat) << " bytes\n";

  // populate arrays
  dfloat *ggeo = drandAlloc(Np*Nelements*p_Nggeo);
  dfloat *q    = drandAlloc((Ndim*Np)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*Np)*Nelements);

  const dfloat lambda1 = 1.1;
  dfloat *lambda = (dfloat*) calloc(2*offset, sizeof(dfloat));
  for(int i=0; i<offset; i++) {
    lambda[i]        = 1.0;
    lambda[i+offset] = lambda1;
  }

  // compute the reference result
  for(int n=0;n<Ndim;++n){
    dfloat *x = q + n*offset;
    dfloat *Ax = Aq + n*offset; 
    axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, x, Ax);
  }

  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(2)
{
  dfloat *local_ggeo, *local_q, *local_Aq, *local_DrV, *local_lambda;
  int numThreads = omp_get_num_threads();
  int tid = omp_get_thread_num();
  int ompSize = Nelements * Np;
  int localSize = ompSize / numThreads;
  local_ggeo = (dfloat*)malloc(localSize*p_Nggeo*sizeof(dfloat));
  local_q = (dfloat*)malloc(Ndim*localSize*sizeof(dfloat));
  local_Aq = (dfloat*)malloc(Ndim*localSize*sizeof(dfloat));
  local_DrV = (dfloat*)malloc(Nq*Nq*sizeof(dfloat));
  local_lambda = (dfloat*)malloc(2*localSize*sizeof(dfloat));

  for(int test=0;test<Ntests;++test) {
    if (Ndim > 1)
      axhelm_n3(
        Nelements, offset, ggeo, DrV, lambda, q, Aq);
    else
      axhelm_n1(
        Nelements, offset, ggeo, DrV, lambda, q, Aq);
  }

  omp_set_num_threads(0);
  for(int test=0;test<Ntests;++test) {
    if (Ndim > 1)
      axhelm_n3(
        Nelements, offset, ggeo, DrV, lambda, q, Aq);
    else
      axhelm(
        Nelements, offset, ggeo, DrV, lambda, q, Aq);
  }
}

  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

  // verify
  dfloat maxDiff = 0;
  for(int n=0;n<Ndim*Np*Nelements;++n){
    dfloat diff = fabs(q[n]-Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  std::cout << "Correctness check: maxError= " << maxDiff << "\n";
  free(ggeo);
  free(q);
  free(Aq);
  free(lambda);
  free(rV);
  free(wV);
  free(DrV);

  // print statistics
  const dfloat GDOFPerSecond = Ndim*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*Nelements/elapsed;
  const long long bytesMoved = (Ndim*2*Np+7*Np+2*Np)*sizeof(dfloat); // x, Mx, opa, lambda
  const double bw = bytesMoved*Nelements/elapsed;
  double flopCount = Ndim*Np*12*Nq;
  if(Ndim == 1) flopCount += 22*Np;
  if(Ndim == 3) flopCount += 69*Np;
  double gflops = flopCount*Nelements/elapsed;
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