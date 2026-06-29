#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <vector>

// transpose
std::vector<double> t(const std::vector<double> &idata, const int width, const int height)
{
  std::vector<double> odata(width * height);
  for (int yIndex = 0; yIndex < height; yIndex++) {
    for (int xIndex = 0; xIndex < width; xIndex++) {
      int index_in  = xIndex + width * yIndex;
      int index_out = yIndex + height * xIndex;
      odata[index_out] = idata[index_in];
    }
  }
  return odata;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to filename> <repeat>\n";
    return 1;
  }
  char *filename = argv[1];
  const int repeat = atoi(argv[2]);

  // n and K should match the dimension of the dataset in the csv file
  const int n = 26280, K = 21, M = 10000;

  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Error: failed to open file alphas.csv. Exit\n";
    return 1;
  }

  int alphas_size = n * K; // n rows and K cols
  std::vector<double> alphas(alphas_size), rands(M * K), probs(alphas_size), probs_ref(alphas_size);

  // load the csv file
  for (int i = 0; i < alphas_size; i++) f >> alphas[i];
  f.close();

  // normal distribution (mean: 0 and var: 1)
  std::mt19937 gen(19937);
  std::normal_distribution<double> norm_dist(0.0,1.0);
  for (int i = 0; i < rands_size; i++) rands[i] = norm_dist(gen);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < alphas_size; i++) reference(alphas, rands, probs_ref, n, K, M, i);

  std::vector<double> t_rands = t(rands, K, M);
  std::vector<double> t_alphas = t(alphas, K, n);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < alphas_size; i++) reference_unitStrides(t_alphas, t_rands, probs_ref, n, K, M, i);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) compute_probs(alphas_size, K, M, j);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of compute_probs: " << (time * 1e-9f) / repeat << " (s)\n";

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) compute_probs_unitStrides(alphas_size, K, M, j);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of compute_probs_unitStrides: " << (time * 1e-9f) / repeat << " (s)\n";

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) compute_probs_unitStrides_sharedMem(alphas_size, K, M, j);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of compute_probs_unitStrides_splitMem: " << (time * 1e-9f) / repeat << " (s)\n";

  return 0;
}