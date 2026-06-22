#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <omp.h>

// Example
// https://pytorch.org/docs/stable/generated/torch.flip.html

template <typename scalar_t>
void flip (const int64_t num_dims, const int64_t num_flip_dims,
           const int32_t dim_size, const int32_t repeat) {
  std::vector<int64_t> flip;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;

  for (int64_t i = 0; i < num_dims; i++) {
#ifdef EXAMPLE
    shape.push_back(2);
#else
    shape.push_back(dim_size);
#endif
  }

  int64_t n = 1;
  for (int64_t i = 0; i < num_dims; i++) {
    n = n * shape[i];
  }

  for (int64_t i = 0; i < num_flip_dims; i++) {
    flip.push_back(i);
  }

  stride.push_back(shape[1] * shape[2]);
  stride.push_back(shape[2]);
  stride.push_back(1);

  property("shape", shape);
  property("flip_dims", flip);
  property("stride", stride);

  int64_t dims_bytes = num_dims * sizeof(int64_t);
  int64_t flip_dims_bytes = num_flip_dims * sizeof(int64_t);
  int64_t input_size_bytes = n * sizeof(scalar_t);
  int64_t output_size_bytes = input_size_bytes;

  scalar_t *input = (scalar_t*) malloc (input_size_bytes);

  for (int i = 0; i < n; i++) {
    input[i] = (scalar_t) i;
  }

  scalar_t *output = (scalar_t*) malloc(output_size_bytes);
  scalar_t *output_ref = (scalar_t*) malloc(output_size_bytes);

  #pragma omp target teams distribute parallel for collapse(2) num_teams(1) num_threads(256)
  for (int64_t c1 = 0; c1 < shape[0]; c1++) {
    for (int64_t c2 = 0; c2 < shape[1]; c2++) {
      int64_t offset = c1 * stride[0] + c2 * stride[1];
      output[offset] = input[offset];
    }
  }

  int error = memcmp(output, output_ref, output_size_bytes);
  printf("%s\n", error ? "FAIL" : "PASS");

#ifdef EXAMPLE
  for (int i = 0; i < n; i++) {
    printf("%f ", output[i]);
  }
  printf("\n");
#endif

  free(input);
  free(output);
  free(output_ref);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t num_dims = atoi(argv[1]);
  const int64_t dim_size = atoi(argv[2]);
  const int32_t repeat = atoi(argv[3]);

#ifdef EXAMPLE
  const int64_t num_flip_dims = 2;
#else
  const int64_t num_flip_dims = num_dims;
#endif

  printf("=========== Data type is FP32 ==========\n");
  flip<float>(num_dims, num_flip_dims, dim_size, repeat);

  printf("=========== Data type is FP64 ==========\n");
  flip<double>(num_dims, num_flip_dims, dim_size, repeat);

  return 0;
}