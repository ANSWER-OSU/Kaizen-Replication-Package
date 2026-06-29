#include <iostream>
#include <omp.h>

// OpenMP version of the CUDA kernel
void concat_gpu(float *inp1, float *inp2, float *output, int sz0, int sz2, int sz1_1, int sz1_2) {
  #pragma omp parallel for collapse(3) default(shared)
  for (int idx0 = 0; idx0 < sz0; idx0++) {
    for (int idx1 = 0; idx1 < (sz1_1 + sz1_2); idx1++) {
      for (int idx2 = 0; idx2 < sz2; idx2++) {
        float *dst_ptr = output + idx0 * sz1_1 * sz2 + idx1 * sz2 + idx2;
        float *src_ptr;
        int sz1;
        int idx1_t = idx1;
        if (idx1_t < sz1_1) {
          sz1 = sz1_1;
          src_ptr = inp1;
        } else {
          idx1_t -= sz1_1;
          sz1 = sz1_2;
          src_ptr = inp2;
        }
        src_ptr += (idx0 * sz1 * sz2 + idx1_t * sz2 + idx2);
        *dst_ptr = *src_ptr;
      }
    }
  }
}