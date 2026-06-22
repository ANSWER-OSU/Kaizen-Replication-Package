#include <iostream>
#include <chrono>
#include <cuda.h>
#include <omp.h>

// polydegree, polydegree, 1, 1)
template <typename T, typename Tf>
void meshJacobiGQ(Tf p, Tf q, Tf r, T *rV, T *wV) {
  // ...
}

// poly degree, num vertices, 1)
template <typename Tf, typename T>
void meshDmatrix1D(Tf p, int q, T *DrV) {
  // ...
}

template <typename Tf, typename T>
T fmax(T a, T b) {
  return (a > b) ? a : b;
}

template <typename Tf, typename T>
T fmin(T a, T b) {
  return (a < b) ? a : b;
}

template <typename Tf, typename T>
void axhelmReference(Tf p, Tf q, Tf r, Tf ggeo[], Tf DrV[], Tf x[], Tf Ax[]) {
  // ...
}

template <typename Tf, typename T>
void axhelm(Tf p, Tf q, Tf r, Tf ggeo[], Tf DrV[], Tf lambda[], Tf x[], Tf Aq[]) {
  // ...
}

template <typename Tf, typename T>
void axhelm_n3(Tf p, Tf q, Tf r, Tf ggeo[], Tf DrV[], Tf lambda[], Tf x[], Tf Aq[]) {
  // ...
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "Usage: ./axhelm Ndim numElements [nRepetitions]" << std::endl;
    return 1;
  }

  int Ndim;
  int Nelements;
  int Ntests = 1;
  sscanf(argv[1], "%d", &Ndim);
  sscanf(argv[2], "%d", &Nelements);
  if (argc >= 4) {
    sscanf(argv[3], "%d", &Ntests);
  }

  const int Nq = POLYNOMIAL_DEGREE + 1;
  const int Np = Nq*Nq*Nq;
  const int offset = Nelements*Np;

  // build element nodes and operators
  Tp *rV, *wV, *DrV;
  meshJacobiGQ(0, 0, POLYNOMIAL_DEGREE, &rV, &wV);
  meshDmatrix1D(POLYNOMIAL_DEGREE, Nq, &DrV);

  // populate device arrays
  T *ggeo, *q, *Aq;

  #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
  for (int n = 0; n < Np * Nelements * p_Nggeo; n++) {
    ggeo[n] = drand48();
  }

  #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
  for (int n = 0; n < (Ndim * Np) * Nelements; n++) {
    q[n] = drand48();
  }

  #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
  for (int n = 0; n < (Ndim * Np) * Nelements; n++) {
    Aq[n] = drand48();
  }

  // compute the reference result
  for (int n = 0; n < Ndim; n++) {
    #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
    for (int i = 0; i < Nelements; i++) {
      axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, q, Aq);
    }
  }

  // compute the OpenMP result
  for (int test = 0; test < Ntests; test++) {
    if (Ndim > 1) {
      #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
      for (int n = 0; n < Nelements; n++) {
        axhelm_n3(Nelements, offset, ggeo, DrV, lambda, q, Aq);
      }
    } else {
      #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
      for (int n = 0; n < Nelements; n++) {
        axhelm(Nelements, offset, ggeo, DrV, lambda, q, Aq);
      }
    }
  }

  // store the device results in the 'q' array
  #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
  for (int n = 0; n < (Ndim * Np) * Nelements; n++) {
    q[n] = Aq[n];
  }

  // verify
  T maxDiff = 0;
  for (int n = 0; n < (Ndim * Np) * Nelements; n++) {
    T diff = abs(q[n] - Aq[n]);
    maxDiff = (maxDiff < diff) ? diff : maxDiff;
  }
  std::cout << "Correctness check: maxError = " << maxDiff << "\n";

  // print statistics
  T elapsed = 0;
  for (int test = 0; test < Ntests; test++) {
    #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
    for (int n = 0; n < Np * Nelements * p_Nggeo; n++) {
      ggeo[n] = drand48();
    }

    #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
    for (int n = 0; n < (Ndim * Np) * Nelements; n++) {
      q[n] = drand48();
    }

    #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
    for (int n = 0; n < (Ndim * Np) * Nelements; n++) {
      Aq[n] = drand48();
    }

    // compute the reference result
    for (int n = 0; n < Ndim; n++) {
      #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
      for (int i = 0; i < Nelements; i++) {
        axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, q, Aq);
      }
    }

    // compute the OpenMP result
    for (int test = 0; test < Ntests; test++) {
      if (Ndim > 1) {
        #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
        for (int n = 0; n < Nelements; n++) {
          axhelm_n3(Nelements, offset, ggeo, DrV, lambda, q, Aq);
        }
      } else {
        #pragma omp target teams distribute parallel for private(n) collapse(2) num_teams(NELEMENTS)
        for (int n = 0; n < Nelements; n++) {
          axhelm(Nelements, offset, ggeo, DrV, lambda, q, Aq);
        }
      }
    }

    start = std::chrono::high_resolution_clock::now();
    end = std::chrono::high_resolution_clock::now();
    elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  const T GDOFPerSecond = Ndim * POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE * Nelements * Ntests / elapsed;
  const T bytesMoved = (Ndim * 2 * Np + 7 * Np + 2 * Np) * sizeof(dfloat);
  const T bw = bytesMoved * Nelements * Ntests / elapsed;
  T flopCount = Ndim * Np * 12 * Nq;
  if (Ndim == 1) {
    flopCount += 22 * Np;
  }
  if (Ndim == 3) {
    flopCount += 69 * Np;
  }
  const T gflops = flopCount * Nelements * Ntests / elapsed;
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