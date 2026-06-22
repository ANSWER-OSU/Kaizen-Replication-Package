void compute_glu(
   const int numTeams,
   const int numThreads,
   const int M,
   const int split_dim_size,
   const int N,
   const float* Xdata,
         float* Ydata)
{
  #pragma omp target teams distribute parallel for collapse(3) num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < split_dim_size; j++) {
      for (int k = 0; k < N; k++) {
        const int x1_index = i * split_dim_size * N + j * N + k;
        const int x2_index = i * split_dim_size * N + (j + split_dim_size) * N + k;
        Ydata[x1_index] = Xdata[x1_index] * (1.f / (1.f + expf(-Xdata[x2_index])));
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }

  const int ndims = atoi(argv[1]);
  const int dim_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  std::vector<int> Xshape;

  for (int i = 0; i < ndims; i++) {
    Xshape.push_back(dim_size);
  }

  std::vector<int> Yshape;
  Yshape.insert(Yshape.end(), Xshape.begin(), Xshape.end());

  uint64_t nelems = size_from_dim(0, Xshape);
  uint64_t nelems_bytes = nelems * sizeof(float);

  float *X = (float*) malloc (nelems_bytes);
  float *Y = (float*) malloc (nelems_bytes);
  float *Y_ref = (float*) malloc (nelems_bytes);

  std::default_random_engine rng(123);
  std::uniform_real_distribution<float> distribution(-6.f,6.f);

  for (uint64_t i = 0; i < nelems; i++) {
    X[i] = distribution(rng);
  }

  #pragma omp target data map(to: X[0:nelems]) map(from: Y[0:nelems])
  {
    const int numTeams = (size_to_dim(0, Xshape) * size_to_dim(1, Xshape) * size_from_dim(2, Xshape) + 256 - 1) / 256;
    const int numThreads = 256;

    for (int input_dim = -1; input_dim < 3 * (ndims-1); input_dim++) {
      const int split_index = (input_dim == -1) ? ndims - 1 : (input_dim % ndims);

      if (Yshape[split_index] % 2 != 0) {
        printf("Split dimension %d should be divided by two. Skip\n", Yshape[split_index]);
        continue;
      }
      const int split_dim_size = Yshape[split_index] / 2;
      const int m = size_to_dim(split_index, Xshape);
      const int n = size_from_dim(split_index + 1, Xshape);

      compute_glu(numTeams, numThreads, m, split_dim_size, n, X, Y_ref);
      for (int i = 0; i < repeat; i++) {
        compute_glu(numTeams, numThreads, m, split_dim_size, n, X, Y);
      }
      bool ok = true;
      for (uint64_t i = 0; i < nelems/2; i++) {
        if (fabsf(Y[i] - Y_ref[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }
  }

  free(X);
  free(Y);
  free(Y_ref);

  return 0;
}