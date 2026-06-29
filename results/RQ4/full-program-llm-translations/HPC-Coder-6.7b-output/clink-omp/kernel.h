#include <omp.h>

#define N 8192
#define WGS 256  // I'm assuming WGS is used only for blockIdx.x calculation
#define SAMPLE_TEST_LEN 20000

int main() {
  #pragma omp parallel for schedule(static)
  for (int gid = 0; gid < N; gid++) {
    if (gid >= N) return;

    float h_state[5] = {0,0,0,0,0};
    float c_state[5] = {0,0,0,0,0};
    float i_state[5] = {0,0,0,0,0};
    float f_state[5] = {0,0,0,0,0};
    float o_state[5] = {0,0,0,0,0};
    float g_state[5] = {0,0,0,0,0};

    for (int t = 0; t < SAMPLE_TEST_LEN; ++t) {
      #pragma omp atomic read
      float x = d_x[gid * SAMPLE_TEST_LEN + t];

      for (int j = 0; j < 5; ++j) {
        float sum = x * d_inW[j];
        for (int i = 0; i < 5; ++i)
          sum += h_state[i] * d_intW[j * 5 + i];
        sum += d_intB[j];
        i_state[j] = 1.f / (1.f + expf(-sum));
      }

      for (int j = 0; j < 5; ++j) {
        float sum = x * d_inW[5 + j];
        for (int i = 0; i < 5; ++i)
          sum += h_state[i] * d_intW[25 + j * 5 + i];
        sum += d_intB[5 + j];
        f_state[j] = 1.f / (1.f + expf(-sum));
      }

      for (int j = 0; j < 5; ++j) {
        float sum = x * d_inW[10 + j];
        for (int i = 0; i < 5; ++i)
          sum += h_state[i] * d_intW[50 + j * 5 + i];
        sum += d_intB[10 + j];
        o_state[j] = 1.f / (1.f + expf(-sum));
      }

      for (int j = 0; j < 5; ++j) {
        float sum = x * d_inW[15 + j];
        for (int i = 0; i < 5; ++i)
          sum += h_state[i] * d_intW[75 + j * 5 + i];
        sum += d_intB[15 + j];
        g_state[j] = tanhf(sum);
      }

      for (int j = 0; j < 5; ++j) {
        c_state[j] = c_state[j] * f_state[j] + g_state[j] * i_state[j];
        h_state[j] = tanhf(c_state[j]) * o_state[j];
      }

      float sum = d_outB[0];
      for (int j = 0; j < 5; ++j)
        sum += h_state[j] * d_outW[j];
      d_y[gid * SAMPLE_TEST_LEN + t] = sum;
    }
  }
  return 0;
}