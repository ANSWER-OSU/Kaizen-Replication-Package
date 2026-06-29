#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "backprop.h"

// OpenMP code to translate CUDA code to OpenMP offload kernel

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {
  setup(argc, argv);
#ifdef OMP_OFFLOAD
  #pragma omp target teams num_teams(num_blocks)
  {
    float* d_input;
    float *d_input_weights;
    float *d_hidden_partial_sum;
    float *d_hidden_delta;
    float *d_input_prev_weights;
    #pragma omp parallel num_threads(num_threads)
    {
      int by = omp_get_team_num();
      int tx = omp_get_thread_num() % BLOCK_SIZE;
      int ty = omp_get_thread_num() / BLOCK_SIZE;
      int index = ( ( hid + 1 ) * BLOCK_SIZE * by ) + ( ( hid + 1 ) * tx ) + ty + 1;
      int index_in = ( ( in + 1 ) * BLOCK_SIZE * by ) + ( ( in + 1 ) * tx ) + ty + 1;
      float tmp;
      float sum = 0.f;
      for ( int k = 0; k < BLOCK_SIZE; ++k ) {
        int index_in_prev = index_in - 1;
        if ( index_in_prev < 1 ) {
          index_in_prev = in;
        }
        int index_prev = index - 1;
        if ( index_prev < 1 ) {
          index_prev = hid;
        }
        tmp = d_input_weights[index_in] * d_hidden_partial_sum[index_prev];
        sum += tmp;
      }
      d_input_weights[index] = sum + d_input_weights[index_in];
    }
  }
#endif
  double offload_end = get_time();
  printf("OpenMP offloading time = %lf(s)\n", offload_end - offload_start);

  bool ok = true;
#ifdef OMP_OFFLOAD
  #pragma omp target teams distribute parallel for reduction(| : ok) num_teams(num_blocks) num_threads(num_threads)
  for (int i = 0; i < (in+1)*(hid+1); i++) {
    if (fabsf(input_weights_one_dim[i] - input_weights_one_dim_r[i]) >= 1e-3f) {
      ok = false;
      break;
    }
  }
#endif
  printf("%s\n", ok ? "PASS" : "FAIL");

#ifdef OUTPUT
  for (int i = 0; i < (in+1); i++) 
    printf("i=%d input_units=%f\n", i,net->input_units[i]);
  for (int i = 0; i < (in+1)*(hid+1); i++) 
    printf("i=%d input_weights=%f\n", i,input_weights_one_dim[i]);
#endif
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_one_dim_r);

  return 0;
}