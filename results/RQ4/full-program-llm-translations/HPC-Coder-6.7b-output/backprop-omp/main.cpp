#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"

#include "bpnn_layerforward.h"
#include "bpnn_adjust_weights.h"

#include "reference.h"

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  setup(argc, argv);
  return 0;
}


int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  float *input_weights_one_dim;
  float *input_weights_one_dim_r;
  float *input_weights_prev_one_dim;
  float * partial_sum;
  float sum;

  num_threads = omp_get_max_threads();

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_one_dim_r = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_threads * WIDTH * sizeof(float));

  unsigned int num_blocks = in / num_threads;

  // this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
  // todo: fix mem allocation
  int m = 0;
  #pragma omp parallel shared(net)
  {
    for (int k = 0; k <= in; k++) {
      for (int j = 0; j <= hid; j++) {
        net->input_weights[k][j] = input_weights_one_dim[m];
        net->input_weights[k][j] = input_weights_one_dim_r[m];
        net->input_prev_weights[k][j] = input_weights_prev_one_dim[m];
        m++;
      }
    }
  }

  printf("Performing openMP offload\n");

  double offload_start = get_time();

  #pragma omp parallel num_threads(num_threads)
  {
    bpnn_layerforward(net->input_units, net->hidden_units, net->hidden_weights, hid, out);
  }

  #pragma omp parallel for reduction(+:sum)
  for (int j = 1; j <= hid; j++) {
    for (unsigned int k = 0; k < num_blocks; k++) {  
      partial_sum[omp_get_thread_num() * hid + j-1] += net->input_weights[0][j];
      sum += partial_sum[omp_get_thread_num() * hid + j-1];
    }
  }

  #pragma omp single
  {
    for (int j = 1; j <= hid; j++) {
      net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }

    #pragma omp master
    {
      bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
      bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
      bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
      bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
    }
  }

  // input_weights has been written in the first kernel, so it needs to be restored.
  #pragma omp parallel for
  for (int i = 0; i < (in+1)*(hid+1); i++) {
    net->input_weights[i] = input_weights_one_dim[i];
    net->input_prev_weights[i] = input_weights_prev_one_dim[i];
  }

  double offload_end = get_time();
  printf("OpenMP offloading time = %lf(s)\n", offload_end - offload_start);

  reference (in, hid, out, net, 
             input_weights_one_dim_r,
             input_weights_prev_one_dim,
             partial_sum); 

  bool ok = true;
  for (int i = 0; i < (in+1)*(hid+1); i++) {
    if (fabsf(net->input_weights[i] - input_weights_one_dim_r[i]) >= 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

#ifdef OUTPUT
  for (int i = 0; i < (in+1); i++) 
    printf("i=%d input_units=%f\n", i,net->input_units[i]);
  for (int i = 0; i < (in+1)*(hid+1); i++) 
    printf("i=%d input_weights=%f\n", i,input_weights_one_dim[i]);
#endif

  free(input_weights_one_dim);
  free(input_weights_one_dim_r);
  free(input_weights_prev_one_dim);
  free(partial_sum);

  return 0;
}