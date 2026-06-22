#pragma omp parallel for
for (int i = 0; i < n; ++i) {
  double maxval;
  int m, k;
  int maxind;
  double M_d = (double) M;
  double w[21];

  for (k = 0; k < K; ++k) {
    probs[i*K + k] = 0.0;
  }

  for (m = 0; m < M; ++m) {
    for (k = 0; k < K; ++k) {
      w[k] = alphas[i*K + k] + rands[m*K + k];
    }

    maxind = K-1;
    maxval = w[K-1];
    for (k = 0; k < (K-1); ++k) {
      if (w[k] > maxval) {
        maxind = k;
        maxval = w[k];
      }
    }
    probs[i*K + maxind] += 1.0;
  }

  for (k = 0; k < K; ++k) {
    probs[i*K + k] /= M_d;
  }
}

#pragma omp parallel for
for (int i = 0; i < n; ++i) {
  double maxval;
  int m, k;
  int maxind;
  double M_d = (double) M;
  double w[21];

  for (k = 0; k < K; ++k) {
    probs[k*n + i] = 0.0;
  }

  for (m = 0; m < M; ++m) {
    for (k = 0; k < K; ++k) {
      w[k] = alphas[k*n + i] + rands[k*M + m];
    }

    maxind = K-1;
    maxval = w[K-1];
    for (k = 0; k < (K-1); ++k) {
      if (w[k] > maxval) {
        maxind = k;
        maxval = w[k];
      }
    }
    probs[maxind*n + i] += 1.0;
  }

  for (k = 0; k < K; ++k) {
    probs[k*n + i] /= M_d;
  }
}

#pragma omp parallel for
for (int i = 0; i < n; ++i) {
  double maxval;
  int m, k;
  int maxind;
  double M_d = (double) M;

  double probs_shared[21];
  double w[21];

  for (k = 0; k < K; ++k) {
    probs_shared[k] = 0.0;
  }

  for (m = 0; m < M; ++m) {
    for (k = 0; k < K; ++k) {
      w[k] = alphas[k*n + i] + rands[k*M + m];
    }

    maxind = K-1;
    maxval = w[K-1];
    for (k = 0; k < (K-1); ++k) {
      if (w[k] > maxval) {
        maxind = k;
        maxval = w[k];
      }
    }
    probs_shared[maxind] += 1.0;
  }

  for (k = 0; k < K; ++k) {
    probs_shared[k] /= M_d;
  }

  for (k = 0; k < K; ++k) {
    probs[k*n + i] = probs_shared[k];
  }
}