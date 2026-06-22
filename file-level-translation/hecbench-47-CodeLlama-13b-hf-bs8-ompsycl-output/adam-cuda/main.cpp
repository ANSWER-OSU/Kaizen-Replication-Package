void adam (
        int numTeams,
        int numThreads,
        float* __restrict__ p,
        float* __restrict__ m,
        float* __restrict__ v,
  const float* __restrict__ g,
  const float b1,
  const float b2,
  const float eps,
  const float grad_scale,
  const float step_size,
  const int time_step,
  const size_t vector_size,
  adamMode_t mode,
  const float decay)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads)
  for (size_t i = 0; i < vector_size; i++) {
    for (int t = 1; t <= time_step; t++) {
      float scaled_grad = g[i]/grad_scale;
      m[i] = b1*m[i] + (1.f-b1)*scaled_grad;
      v[i] = b2*v[i] + (1.f-b2)*scaled_grad*scaled_grad;
      float m_corrected = m[i] / (1.f-powf(b1, t));
      float v_corrected = v[i] / (1.f-powf(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v_corrected + eps);
      else // Mode 1
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected/denom) + (decay*p[i]);
      p[i] -= (step_size*update);
    }
  }
}