#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>

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
  // Shared memory to store the input and weight predictions
  accscalar_t sm_inputs[NLL_LOSS_THREADS],
                         acc_weight[NLL_LOSS_THREADS];

  #pragma omp parallel shared(sm_inputs, acc_weight)
  {
    int tid = omp_get_thread_num();
    sm_inputs[tid] = static_cast<accscalar_t>(0);
    acc_weight[tid] = static_cast<accscalar_t>(0);

    // Loop over each index in the batch
    #pragma omp for
    for (int i = 0; i < nframe; i++) {
      index_t t = target[i];
      if (t != ignore_index) {
        scalar_t cur_weight =
            weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
        
        // Accumulate the prediction and weight for the current index
        sm_inputs[tid] -= static_cast<accscalar_t>(input[i * kdim + t] * cur_weight);
        acc_weight[tid] += static_cast<accscalar_t>(cur_weight);
      }
    }

    #pragma omp for
    for (int j = 0; j < NLL_LOSS_THREADS; j++) {
      // Accumulate the final output and total weight from all threads
      *output += sm_inputs[j];
      *total_weight += acc_weight[j];
    }
  }
  
  // Scaling the output using the size_average flag
  if (size_average) {
    *output /= *total_weight;
  }
}

template <typename scalar_t, typename index_t>
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

  scalar_t h_output = 0, h_total_weight = 0;

  scalar_t *d_output;
  scalar_t *d_total_weight;
  scalar_t *d_input;
   index_t *d_target;
  scalar_t *d_weights;

  // Allocate memory for the input and target data
  d_input = h_input;
  d_weights = h_weights;
  d_target = h_target;

  // Assume the d_output and d_total_weight already have allocated memory
  // for the nll loss forward reduce2d kernel
  d_output = &h_output;
  d_total_weight = &h_total_weight;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    nll_loss_forward_reduce2d_kernel<scalar_t, scalar_t, index_t, omp_get_max_threads()>(
      d_output,
      d_total_weight,
      d_input,
      d_target,
      d_weights,
      size_average,
      nframe,
      kdim,
      ignore_index);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nOpenMP thread count: %d\n", omp_get_max_threads());
  printf("Average execution time of nll loss forward kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  bool ok = true;
  float ho, ro, hw, rw;
  ho = (float)h_output;
  ro = (float)r_output; 
  hw = (float)h_total_weight;
  rw = (float)r_total_weight;
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

  // Assume a reference function exists for evaluating the loss function
  reference<scalar_t, scalar_t, index_t>(
    &r_output, &r_total_weight,
    h_input, h_target, h_weights,
    size_average, nframe, n_classes, ignore_index);

  eval<scalar_t, index_t>(nframe, n_classes, n_classes,
                          size_average, ignore_index,
                          r_output, r_total_weight,
                          h_input, h_weights, h_target, repeat);

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