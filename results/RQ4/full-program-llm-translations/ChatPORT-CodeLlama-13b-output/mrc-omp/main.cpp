void MRCGradient (
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float*__restrict__ dX1, float*__restrict__ dX2)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float y = Y[i];
    float o = dOutput[i];
    float dist = y * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) {
      dX1[i] = 0.f;
      dX2[i] = 0.f;
    } else {
      dX1[i] = -y * o;
      dX2[i] = y * o;
    }
  }
}

void MRCGradient2(
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float*__restrict__ dX1, float*__restrict__ dX2)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float y = Y[i];
    float o = dOutput[i];
    float dist = y * (X1[i] - X2[i]) + margin;
    dX1[i] = dist < 0.f ? 0.f : -y * o;
    dX2[i] = dist < 0.f ? 0.f : y * o;
  }
}