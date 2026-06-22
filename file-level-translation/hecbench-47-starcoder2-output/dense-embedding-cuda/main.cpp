#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <random>

template <typename T>
void reference(
    const T* input,
    const T* dense,
    T* output,
    int embedding_dim,
    int batch_size,
    const int* offset)
{
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int range = offset[batch_idx + 1] - offset[batch_idx];
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[offset[batch_idx] + nested_idx] =
          input[offset[batch_idx] + nested_idx] + dense_elem;
      }
    }
  }
}

template <typename T>
void dense_embedding(
    const T* input,
    const T* dense,
    T* output,
    int embedding_dim,
    const int* offset,
    int batch_size)
{
  #pragma omp parallel for
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int range = offset[batch_idx + 1] - offset[batch_idx];
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[offset[batch_idx] + nested_idx] =
          input[offset[batch_idx] + nested_idx] + dense_elem;
      }
    }
  }
}


int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int batch_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  assert(nrows > batch_size * batch_size);

  printf("Number of rows in the embedding table: %d\n", nrows);
  printf("Batch size: %d\n", batch_size);

  const int embed_dims[] = {768, 2048, 12288};

  for (size_t n = 0; n < sizeof(embed_dims)/sizeof(int); n++) {
    int ncols = embed_dims[n];
    printf("\nEmbedding dimension: %d\n", ncols);

    int input_size = nrows * ncols;  // same as output size
    size_t input_size_bytes = input_size * sizeof(float);

    int dense_size = batch_size * ncols ;
    int dense_size_bytes = dense_size * sizeof(float);

    int batch_size_bytes = (batch_size + 1) * sizeof(float);

    float *input, *dense, *output_ref, *output_omp;
    input = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    dense = (float*) malloc (dense_size_bytes); // [batch_size x embedding_dim]
    output_ref = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    output_omp = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    int *input_offset = (int*) malloc (batch_size_bytes);  // [batch_size]

    // meet a few conditions:
    // range > ncols
    // range + offset[i] < input_size
    // offset[i] > offset[i-1]
    srand(123);
    input_offset[0] = 0;
    for (int i = 1; i <= batch_size; i++)
      input_offset[i] = input_offset[i-1] + (rand() % batch_size + 1) * ncols;

    std::default_random_engine g (123);
    std::uniform_real_distribution<float> distr (-1.f, 1.f);
    for (int i = 0; i < dense_size; i++) {
      dense[i] = distr(g);
    }

    for (int i = 0; i < input_size; i++) {
      input[i] = distr(g);
      output_ref[i] = 0;
      output_omp[i] = 0;
    }

    reference(input, dense, output_ref, ncols, batch_size, input_offset);

    double start = 0;
    double end = 0;
    double time = 0;

    start = omp_get_wtime();

    for (int i = 0; i < repeat; i++)
      dense_embedding(input, dense, output_omp, ncols, input_offset, batch_size);

    end = omp_get_wtime();
    time = (end - start) * 1000;
    printf("Average execution time of dense embedding kernel (omp): %lf (ms)\n", (time) / repeat);

    bool ok = true;
    for (int i = 0; i < input_size; i++) {
      if (fabsf(output_omp[i] - output_ref[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    free(input);
    free(dense);
    free(output_ref);
    free(output_omp);
    free(input_offset);
  }

  return 0;
}