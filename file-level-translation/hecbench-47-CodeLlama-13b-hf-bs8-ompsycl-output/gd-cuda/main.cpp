void update(omp_ptr<float> x,
            const omp_ptr<const float> grad,
            int m, int n, float lambda, float alpha)
{
  #pragma omp target teams distribute parallel for
  for(int i = 0; i < n; i++) {
    float g = grad[i] / (float)m + lambda * x[i];
    x[i] = x[i] - alpha * g;
  }
}