#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include "reference.h"

void adam(float* p, float* m, float* v, const float* g, const float b1, const float b2, const float eps, const float grad_scale, const float step_size, const int time_step, const size_t vector_size, adamMode_t mode, const float decay)
{
#pragma omp parallel shared(p, m, v, g, b1, b2, eps, grad_scale, step_size, time_step, vector_size, mode, decay) num_threads(omp_get_max_threads())
{
    #pragma omp for
    for (size_t j = 0; j < vector_size; j++) {
        float m_private = m[j];
        float v_private = v[j];
        for(int t = 1; t <= time_step; t++) {
            float scaled_grad = g[j] / grad_scale;
            m_private = b1 * m_private + (1.f - b1) * scaled_grad;
            v_private = b2 * v_private + (1.f - b2) * scaled_grad * scaled_grad;
            float m_corrected = m_private / (1.f - powf(b1, t));
            float v_corrected = v_private / (1.f - powf(b2, t));
            float denom;
            if (mode == ADAM_MODE_0)
                denom = sqrtf(v_corrected + eps);
            else // Mode 1
                denom = sqrtf(v_corrected) + eps;
            float update = (m_corrected / denom) + (decay * p[j]);
            p[j] -= (step_size * update);
        }
    }
}
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <vector size> <number of time steps> <repeat>\n", argv[0]);
    return 1;
  }

  const int vector_size = atoi(argv[1]);
  const int time_step = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  float *m = (float*)malloc(vector_size * sizeof(float));
  float *v = (float*)malloc(vector_size * sizeof(float));
  float *g = (float*)malloc(vector_size * sizeof(float));
  float *p = (float*)malloc(vector_size * sizeof(float));
  float *r = (float*)malloc(vector_size * sizeof(float));

  std::uniform_real_distribution<float> dist(0, 1);
  std::mt19937 gen(19937);
  for (int i = 0; i < vector_size; i++) {
    m[i] = dist(gen);
    v[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = dist(gen);
  }

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;

  adamMode_t mode = ADAM_MODE_0;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    adam(p, m, v, g, beta1, beta2, eps, grad_scale, step_size, time_step, vector_size, mode, decay);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

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