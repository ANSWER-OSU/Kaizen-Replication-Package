#include <omp.h>
void permute(float* out, const float* inp, int B, int T, int C, int NH, const int block_size) {
  int HS = C / NH;
  float* Q = out + 0 * B * T * C;
  float* K = out + 1 * B * T * C;
  float* V = out + 2 * B * T * C;
  #pragma omp parallel for
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < NH; n++) {
      for (int t = 0; t < T; t++) {
        for (int c = n * HS; c < (n+1) * HS; c++) {
          Q[b * T * C + t * C + c] = inp[b * T * 3 * C + t * 3 * C + c];
          K[b * T * C + t * C + c] = inp[b * T * 3 * C + t * 3 * C + C + c];
          V[b * T * C + t * C + c] = inp[b * T * 3 * C + t * 3 * C + 2 * C + c];
        }
      }
    }
  }
}