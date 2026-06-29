void SwishKernel(const int numTeams, const int numThreads, const int N, const float* X, float* Y)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads) 
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] / (1.f + exp(-X[i]));
  }
}

void SwishGradientKernel(const int numTeams, const int numThreads, const int N, const float* X, const float* Y, const float* dY, float* dX)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads) 
  for (int i = 0; i < N; i++) {
    dX[i] = dY[i] * (Y[i] + (1.f - Y[i]) / (1.f + exp(-X[i])));
  }
}

void eval_swish (const int numTeams, const int numThreads, const int N, const int repeat) {

  float *h_X  = (float*) malloc (N * sizeof(float));
  float *h_Y  = (float*) malloc (N * sizeof(float));
  float *h_dY = (float*) malloc (N * sizeof(float));
  float *h_dX = (float*) malloc (N * sizeof(float));
  float *r_Y  = (float*) malloc (N * sizeof(float));
  float *r_dX = (float*) malloc (N * sizeof(float));

  std::default_random_engine gen (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);
  for (int i = 0; i < N; i++) {
    h_X[i] = distr(gen);
    h_dY[i] = distr(gen);
  }

  #pragma omp target data map(to: h_X[0:N], h_dY[0:N]) map(from: h_Y[0:N], h_dX[0:N])
  {
    SwishKernel(numTeams, numThreads, N, h_X, h_Y);
    SwishGradientKernel(numTeams, numThreads, N, h_X, h_Y, h_dY, h_dX);
  }

  reference (N, h_X, r_Y, r_dX, h_dY);

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_dX[i] - r_dX[i]) > 1e-3 || fabs(h_Y[i] - r_Y[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_X);
  free(h_Y);
  free(h_dX);
  free(h_dY);
  free(r_dX);
  free(r_Y);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of elements> <number of teams> <number of threads> <repeat>\n", argv[0]);
    return 1;
  }

  const int N = atoi(argv[1]);
  const int numTeams = atoi(argv[2]);
  const int numThreads = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  eval_swish (numTeams, numThreads, N, repeat);