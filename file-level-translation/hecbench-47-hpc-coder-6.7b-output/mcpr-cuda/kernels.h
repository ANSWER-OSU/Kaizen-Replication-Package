#include <omp.h>

void compute_probs_unitStrides_omp(
  const double* alphas,
  const double* rands,
  double* probs,
  int n, int K, int M) 
{
  double maxval;
  int m, k;
  int maxind;
  double M_d = M;
  double w[K];

  // unit strides in innermost loop
  #pragma omp parallel for num_threads(omp_get_max_threads()) shared(n, K, M_d, probs, alphas, rands) private(w, maxval, maxind)
  for(int i = 0; i < n; ++i){ // assign overall id/index of the thread = id of row
    for(k = 0; k < K; ++k){  // initialize probs (though already done on CPU)
      probs[k*n + i] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){    // loop over Monte Carlo iterations
      #pragma omp simd simdlen(K) private(w)
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        w[k] = alphas[k*n + i] + rands[k*M + m];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        } 
      }
      probs[maxind*n + i] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      probs[k*n + i] /= M_d;
    }
  }
}