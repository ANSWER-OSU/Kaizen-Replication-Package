// OpenMP program to solve the adam algorithm:

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <random>

// adamMode_t and reference functions are skipped for brevity

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <vector size> <number of time steps> <repeat>\n", argv[0]);
    return 1;
  }

  const int vector_size = atoi(argv[1]);
  const int time_step = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  size_t size_bytes = vector_size * sizeof(float);

  float *m = (float*) malloc (size_bytes);
  float *v = (float*) malloc (size_bytes);
  float *g = (float*) malloc (size_bytes);
  float *p = (float*) malloc (size_bytes);
  float *r = (float*) malloc (size_bytes);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(0, 1);
  for (int i = 0; i < vector_size; i++) {
    m[i] = dist(gen);
    v[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = dist(gen);
  }

  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;
  adamMode_t mode = ADAM_MODE_0;

  omp_set_num_threads(1);
  double start = omp_get_wtime();

  for (int i = 0; i < repeat; i++) {
    adam<float, float>(
      p, m, v, g,
      beta1, beta2,
      eps,
      grad_scale,
      step_size,
      time_step,
      vector_size,
      mode,
      decay);
  }

  double end = omp_get_wtime();
  printf("Average kernel execution time %lf (ms)\n", 1e3 * (end - start) / repeat);

  // verify
  reference<float, float>(
    repeat,
    r, m, v, g,
    beta1, beta2,
    eps,
    grad_scale,
    step_size,
    time_step,
    vector_size,
    mode,
    decay);

  bool ok = true; 
  double cr = 0, cp = 0;
  for (int i = 0; i < vector_size; i++) {
    if (fabsf(r[i] - p[i]) > 1e-3f) {
      ok = false;
      break;
    }
    cr += r[i]; cp += p[i];
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  printf("Checksum: %lf %lf\n", cr / vector_size, cp / vector_size);

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  return 0;
}

template <typename T, typename G>
void adam (
        T* __restrict__ p,
        T* __restrict__ m,
        T* __restrict__ v,
  const G* __restrict__ g,
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
  #pragma omp parallel for
  for (size_t j = 0; j < vector_size; j++) {
    for (int t = 1; t <= time_step; t++) {
      T scaled_grad = g[j]/grad_scale;
      m[j] = b1*m[j] + (1.f-b1)*scaled_grad;
      v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;
      float m_corrected = m[j] / (1.f-powf(b1, t));
      float v_corrected = v[j] / (1.f-powf(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v_corrected + eps);
      else // Mode 1
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected/denom) + (decay*p[j]);
      p[j] -= (step_size*update);
    }
  }
}