#include <omp.h>

void eval_omp(const int nrows, const int repeat) {
  // ...
  T *a, *b, *o;
  a = (T*) malloc (size_bytes);
  b = (T*) malloc (size_bytes);
  o = (T*) malloc (size_bytes);

#pragma omp parallel
{
  T *d_a, *d_b, *d_o;
  d_a = (T*) omp_alloc (size_bytes);
  d_b = (T*) omp_alloc (size_bytes);
  d_o = (T*) omp_alloc (size_bytes);

  int n_threads = omp_get_num_threads();
  int thread_id = omp_get_thread_num();

  // ...
  std::default_random_engine g (123);
  std::uniform_real_distribution<T> distr (-2.f, 2.f);
  for (int i = thread_id; i < num_elems; i += n_threads) {
    a[i] = distr(g);
    b[i] = distr(g);
  }

  #pragma omp barrier

  // ...
  #pragma omp for
  for (int i = 0; i < num_elems; i += n_threads) {
    auto* out_row = d_o + 3*i;
    const auto* x1_row = d_a + 3*i;
    const auto* x2_row = d_b + 3*i;

    const T val0 = (x1_row[1] * x2_row[2] - x1_row[2] * x2_row[1]);

    const T val1 = (x1_row[2] * x2_row[0] - x1_row[0] * x2_row[2]);

    const T val2 = (x1_row[0] * x2_row[1] - x1_row[1] * x2_row[0]);

    out_row[0] = val0;
    out_row[1] = val1;
    out_row[2] = val2;
  }

  omp_flush(0, num_elems);

  // ...
  free(a);
  free(b);
  free(o);
  omp_free(d_a);
  omp_free(d_b);
  omp_free(d_o);
}
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of rows in a 2D tensor> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  printf("=========== Data type is FP32 ==========\n");
  eval_omp<float>(nrows, repeat);

  printf("=========== Data type is FP64 ==========\n");
  eval_omp<double>(nrows, repeat);

  return 0;
}