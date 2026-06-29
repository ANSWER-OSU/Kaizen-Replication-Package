#ifndef KERNEL_H
#define KERNEL_H

#include <cuda.h>

struct ComplexFloat {
    float Re;
    float Im;
};

void chemv_kernel0(const int numTeams, const int numThreads, struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y, float alpha_im, float alpha_re, float beta_im, float beta_re) {
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int c1 = 0; c1 < numTeams; c1++) {
    for (int c3 = 0; c3 < numThreads; c3++) {
      ComplexFloat tmp;
      tmp.Re = ((alpha_re * AT[((c1 * numThreads) + c3)].Re) - (alpha_im * (-AT[((c1 * numThreads) + c3)].Im)));
      tmp.Im = ((alpha_im * AT[((c1 * numThreads) + c3)].Re) + (alpha_re * (-AT[((c1 * numThreads) + c3)].Im)));
      X[((c1 * numThreads) + c3)].Re = ((tmp.Re * X[((c1 * numThreads) + c3)].Re) - (tmp.Im * X[((c1 * numThreads) + c3)].Im));
      X[((c1 * numThreads) + c3)].Im = ((tmp.Im * X[((c1 * numThreads) + c3)].Re) + (tmp.Re * X[((c1 * numThreads) + c3)].Im));
    }
  }
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int c1 = 0; c1 < numTeams; c1++) {
    for (int c3 = 0; c3 < numThreads; c3++) {
      ComplexFloat tmp;
      tmp.Re = ((beta_re * Y[((c1 * numThreads) + c3)].Re) - (beta_im * Y[((c1 * numThreads) + c3)].Im));
      tmp.Im = ((beta_im * Y[((c1 * numThreads) + c3)].Re) + (beta_re * Y[((c1 * numThreads) + c3)].Im));
      Y[((c1 * numThreads) + c3)].Re = ((tmp.Re * X[((c1 * numThreads) + c3)].Re) - (tmp.Im * X[((c1 * numThreads) + c3)].Im));
      Y[((c1 * numThreads) + c3)].Im = ((tmp.Im * X[((c1 * numThreads) + c3)].Re) + (tmp.Re * X[((c1 * numThreads) + c3)].Im));
    }
  }
}

void chemv_kernel1(const int numTeams, const int numThreads, struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y, float alpha_im, float alpha_re) {
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int c1 = 0; c1 < numTeams; c1++) {
    for (int c3 = 0; c3 < numThreads; c3++) {
      ComplexFloat tmp;
      tmp.Re = ((alpha_re * AT[((c1 * numThreads) + c3)].Re) - (alpha_im * (-AT[((c1 * numThreads) + c3)].Im)));
      tmp.Im = ((alpha_im * AT[((c1 * numThreads) + c3)].Re) + (alpha_re * (-AT[((c1 * numThreads) + c3)].Im)));
      X[((c1 * numThreads) + c3)].Re = ((tmp.Re * X[((c1 * numThreads) + c3)].Re) - (tmp.Im * X[((c1 * numThreads) + c3)].Im));
      X[((c1 * numThreads) + c3)].Im = ((tmp.Im * X[((c1 * numThreads) + c3)].Re) + (tmp.Re * X[((c1 * numThreads) + c3)].Im));
    }
  }
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int c1 = 0; c1 < numTeams; c1++) {
    for (int c3 = 0; c3 < numThreads; c3++) {
      Y[((c1 * numThreads) + c3)].Re += X[((c1 * numThreads) + c3)].Re;
      Y[((c1 * numThreads) + c3)].Im += X[((c1 * numThreads) + c3)].Im;
    }
  }
}

#endif