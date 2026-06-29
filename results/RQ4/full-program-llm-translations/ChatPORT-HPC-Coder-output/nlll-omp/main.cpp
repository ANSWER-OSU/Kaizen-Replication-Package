#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"

template <typename scalar_t, typename accscalar_t, 
          typename index_t, int NLL_LOSS_THREADS>
void nll_loss_forward_reduce2d_kernel(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ total_weight,
    const scalar_t* __restrict__ input,
    const index_t*  __restrict__ target,
    const scalar_t* __restrict__ weights,
    bool size_average,
    int64_t nframe,
    int64_t kdim,
    int64_t ignore_index)
{
  #pragma omp target teams distribute reduction(+ : accscalar_t (sm_inputs[NLL_LOSS_THREADS])) num_teams(1)
  for (int i = 0; i < nframe; i++) {
    index_t t = target[i];
    if (t != ignore_index) {
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      sm_inputs[omp_get_thread_num()] -= static_cast<accscalar_t>(input[i * kdim + t] * cur_weight);
      acc_weight[omp_get_thread_num()] += static_cast<accscalar_t>(cur_weight);
    }
  }

  #pragma omp target update from(sm_inputs[:NLL_LOSS_THREADS], acc_weight[:NLL_LOSS_THREADS])

  *total_weight = static_cast<scalar_t>(acc_weight[0]);
  accscalar_t output_acc = 0;
  for (int i = 1; i < NLL_LOSS_THREADS; ++i) {
    *total_weight += static_cast<scalar_t>(acc_weight[i]);
    output_acc += sm_inputs[i];
  }
  output_acc += sm_inputs[0];
  if (size_average) {
    *output = static_cast<scalar_t>(output_acc / *total_weight);
  } else {
    *output = static_cast<scalar_t>(output_acc);
  }
}

template <typename scalar_t, typename index_t, int GPU_THREADS>
void eval(const int64_t nframe,
          const int64_t kdim,
          const int64_t n_classes,
          const bool size_average,
          const int64_t ignore_index,
          const scalar_t r_output,
          const scalar_t r_total_weight,
          scalar_t *h_input,
          scalar_t *h_weights,
           index_t *h_target,
          const int repeat)
{
  int64_t input_size = nframe * kdim;
  int64_t input_size_bytes = input_size * sizeof(scalar_t);

  int64_t weights_size = nframe;
  int64_t weights_size_bytes = weights_size * sizeof(scalar_t);

  int64_t target_size = nframe;
  int64_t target_size_bytes = target_size * sizeof(index_t);

  int output_size_bytes = sizeof(scalar_t);

  scalar_t h_output;
  scalar_t h_total_weight;

  for (int i = 0; i < repeat; i++) {
    nll_loss_forward_reduce2d_kernel
      <scalar_t, scalar_t, index_t, GPU_THREADS>
      (&h_output, &h_total_weight,
       h_input, h_target, h_weights,
       size_average, nframe, kdim, ignore_index);
  }

  bool ok = true;
  float ho, ro, hw, rw;
  ho = (float)h_output;
  ro = (float)r_output; 
  hw = (float)h_total_weight;
  rw = (float)r_total_weight;
  // relax the error bound for FP16
  if (fabs(ho - ro) > 1.f || fabs(hw - rw) > 1.f) {
    printf("%f %f %f %f\n", ho, ro, hw, rw);
    ok = false;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

template <typename scalar_t, typename index_t>
void driver(char** argv) {
  const int64_t nframe = atol(argv[1]);
  const int64_t n_classes = atol(argv[2]);
  const int repeat = atoi(argv[3]);

  const int64_t input_size = nframe * n_classes;
  const int64_t input_size_bytes = input_size * sizeof(scalar_t);

  const int64_t weights_size = nframe;
  const int64_t weights_size_bytes = weights_size * sizeof(scalar_t);

  const int64_t target_size = nframe;
  const int64_t target_size_bytes = target_size * sizeof(index_t);

  scalar_t *h_input = (scalar_t*) malloc (input_size_bytes);
  scalar_t *h_weights = (scalar_t*) malloc (weights_size_bytes);
  index_t *h_target = (index_t*) malloc (target_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> d1 (-1.f, 1.f);
  std::uniform_int_distribution<index_t> d2 (0, n_classes-1);

  for (int64_t i = 0; i < input_size; i++)
    h_input[i] = d1(g);

  for (int64_t i = 0; i < weights_size; i++)
    h_weights[i] = d1(g);

  for (int64_t i = 0; i < target_size; i++)
    h_target[i] = d2(g);

  const bool size_average = true;

  // the index may not necessarily be in the class range
  const int64_t ignore_index = n_classes / 2;
  
  // verify the loss function
  scalar_t r_output;
  scalar_t r_total_weight;

  reference<scalar_t, scalar_t, index_t>(
    &r_output, &r_total_weight,
    h_input, h_target, h_weights,
    size_average, nframe, n_classes, ignore_index);

  #define EVAL(nThreads) \
  eval<scalar_t, index_t, nThreads>(nframe, n_classes, n_classes, \
                                    size_average, ignore_index, \
                                    r_output, r_total_weight, \
                                    h_input, h_weights, h_target, repeat)
  EVAL(64);
  EVAL(128);
  EVAL(256);
  EVAL(512);
  EVAL(1024);

  free(h_input);
  free(h_target);
  free(h_weights);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <minibatch size> <number of classes> <repeat>\n", argv[0]);
    return 1;
  }

  printf("=========== Data type is FP32 ==========\n");
  driver<float, int>(argv);

  printf("=========== Data type is FP16 ==========\n");
  driver<half, int>(argv);

  return 0;
}