#include <omp.h>

int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out, num_threads;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  float *input_weights_one_dim;
  float *input_weights_one_dim_r;
  float *input_weights_prev_one_dim;
  float * partial_sum;
  float sum;

  num_threads = in / BLOCK_SIZE;

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_one_dim_r = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_threads * WIDTH * sizeof(float));

  int m = 0;
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_one_dim_r[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
      m++;
    }
  }

  printf("Performing device offload\n");

  double offload_start = get_time();

#pragma omp parallel num_threads(num_threads)
  {
    float *d_input;
    float *d_input_weights;
    float *d_hidden_partial_sum;
    float *d_hidden_delta;
    float *d_input_prev_weights;
    
    omp_set_num_threads(num_threads);
    
    cudaMalloc((void**)&d_input, sizeof(float)*(in+1));
    cudaMalloc((void**)&d_input_weights, sizeof(float)*(in+1)*(hid+1));
    cudaMalloc((void**)&d_hidden_partial_sum, sizeof(float)*num_threads*WIDTH);
    
    cudaMemcpy(d_input, net->input_units, sizeof(float)*(in+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_weights, input_weights_one_dim, sizeof(float)*(in+1)*(hid+1), cudaMemcpyHostToDevice);
    
    kernel_layerforward<<<1, BLOCK_SIZE>>>(d_input, d_input_weights, d_hidden_partial_sum, hid);
    cudaMemcpy(partial_sum, d_hidden_partial_sum, sizeof(float)*num_threads*WIDTH, cudaMemcpyDeviceToHost);
    
    for (int j = 1; j <= hid; j++) {
      sum = 0.f;
      for (int k = 0; k < num_threads; k++) {  
        sum += partial_sum[k * hid + j-1] ;
      }
      sum += net->input_weights[0][j];
      net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }
    
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
    
    cudaMemcpy(d_input_weights, input_weights_one_dim, sizeof(float)*(in+1)*(hid+1), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_hidden_delta, sizeof(float)*(hid+1));
    cudaMalloc((void**)&d_input_prev_weights, sizeof(float)*(in+1)*(hid+1));
    
    cudaMemcpy(d_hidden_delta, net->hidden_delta, sizeof(float)*(hid+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_prev_weights, input_weights_prev_one_dim, sizeof(float)*(in+1)*(hid+1), cudaMemcpyHostToDevice);
    
    kernel_adjust_weights<<<1, BLOCK_SIZE>>>(d_input, d_input_weights, d_hidden_delta, d_input_prev_weights, hid);
    cudaMemcpy(input_weights_one_dim, d_input_weights, sizeof(float)*(in+1)*(hid+1), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_input_weights);
    cudaFree(d_hidden_partial_sum);
    cudaFree(d_hidden_delta);
    cudaFree(d_input_prev_weights);
  }

  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

  reference (in, hid, out, net, 
             input_weights_one_dim_r,
             input_weights_prev_one_dim,
             partial_sum); 

  bool ok = true;
  for (int i = 0; i < (in+1)*(hid+1); i++) {
    if (fabsf(input_weights_one_dim[i] - input_weights_one_dim_r[i]) >= 1e-3f) {
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
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_one_dim_r);

  return 0;
}