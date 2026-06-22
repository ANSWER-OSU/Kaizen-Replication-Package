void cross_kernel(
    const int numel,
          float* out,
    const float* x1,
    const float* x2,
    const int ostride,
    const int x1stride,
    const int x2stride)
{
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < numel; i++) {
    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const float val0 = (x1_row[1 * x1stride] * x2_row[2 * x2stride] -
                        x1_row[2 * x1stride] * x2_row[1 * x2stride]);

    const float val1 = (x1_row[2 * x1stride] * x2_row[0 * x2stride] -
                        x1_row[0 * x1stride] * x2_row[2 * x2stride]);

    const float val2 = (x1_row[0 * x1stride] * x2_row[1 * x2stride] -
                        x1_row[1 * x1stride] * x2_row[0 * x2stride]);

    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

void cross2_kernel(
    const int numel,
          float* out,
    const float* x1,
    const float* x2,
    const int ostride,
    const int x1stride,
    const int x2stride)
{
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < numel; i++) {
    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const float x1_c0 = x1_row[0 * x1stride];
    const float x1_c1 = x1_row[1 * x1stride];
    const float x1_c2 = x1_row[2 * x1stride];
    const float x2_c0 = x2_row[0 * x2stride];
    const float x2_c1 = x2_row[1 * x2stride];
    const float x2_c2 = x2_row[2 * x2stride];

    const float val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1 ;

    const float val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2 ;

    const float val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0 ;

    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

void cross3_kernel(
    const int numel,
          float* out,
    const float* x1,
    const float* x2)
{
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < numel; i++) {
    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const float x1_c0 = x1_row[0];
    const float x1_c1 = x1_row[1];
    const float x1_c2 = x1_row[2];
    const float x2_c0 = x2_row[0];
    const float x2_c1 = x2_row[1];
    const float x2_c2 = x2_row[2];

    const float val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1 ;

    const float val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2 ;

    const float val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0 ;

    out_row[0] = val0;
    out_row[1] = val1;
    out_row[2] = val2;
  }
}

void eval(const int nrows, const int repeat) {
  const int num_elems = nrows * 3;
  const int size_bytes = num_elems * sizeof(float); 

  float *a, *b, *o, *o2, *o3;
  a = (float*) malloc (size_bytes);
  b = (float*) malloc (size_bytes);
  o = (float*) malloc (size_bytes);
  o2 = (float*) malloc (size_bytes);
  o3 = (float*) malloc (size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);
  for (int i = 0; i < num_elems; i++) {
    a[i] = distr(g);
    b[i] = distr(g);
  }

  #pragma omp target data use_device_ptr(a, b, o)
  {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) 
      cross_kernel(nrows, o, a, b, 1, 1, 1);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of cross1 kernel: %f (us)\n", (time * 1e-3f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) 
      cross2_kernel(nrows, o, a, b, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of cross2 kernel: %f (us)\n", (time * 1e-3f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) 
      cross3_kernel(nrows, o, a, b);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of cross3 kernel: %f (us)\n", (time * 1e-3f) / repeat);
  }

  bool ok = true;
  for (int i = 0; i < num_elems; i++) {
    if (fabs(o[i] - o2[i]) > 1e-3 || fabs(o[i] - o3[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(a);
  free(b);
  free(o);
  free(o2);
  free(o3);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of rows in a 2D tensor> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  eval(nrows, repeat);

  return 0;
}