#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>

void MRCGradient(
    const int N, const std::vector<int>& Y, const std::vector<float>& X1, const std::vector<float>& X2, const std::vector<float>& dOutput,
    const float margin, std::vector<float>& dX1, std::vector<float>& dX2)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) {
      dX1[i] = dX2[i] = 0.f;
    } else {
      dX1[i] = -Y[i] * dOutput[i];
      dX2[i] = Y[i] * dOutput[i];
    }
  }
}

void MRCGradient2(
    const int N, const std::vector<int>& Y, const std::vector<float>& X1, const std::vector<float>& X2, const std::vector<float>& dOutput,
    const float margin, std::vector<float>& dX1, std::vector<float>& dX2)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float y = Y[i];
    float o = dOutput[i];
    float dist = -y * (X1[i] - X2[i]) + margin;
    dX1[i] = dist < 0.f ? 0.f : -y * o;
    dX2[i] = dist < 0.f ? 0.f : y * o;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <number of elements> <repeat>" << std::endl;
    return 1;
  }
  const int length = std:: atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  const float m = 0.01;  // margin

  std::vector<float> h_X1 (length);
  std::vector<float> h_X2 (length);
  std::vector<float> h_O (length);
  std::vector<int> h_Y (length);
  std::vector<float> h_dX1 (length);
  std::vector<float> h_dX2 (length);
  std::vector<float> r_dX1 (length);
  std::vector<float> r_dX2 (length);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);
  for (int i = 0; i < length; i++) {
    h_X1[i] = distr(g);
    h_X2[i] = distr(g);
    h_O[i] = distr(g);
    h_Y[i] = (distr(g) < 0) ? -1 : 1;
  }

  const int num_threads = omp_get_max_threads();
  std::vector<float> d_X1 (length);
  std::vector<float> d_X2 (length);
  std::vector<float> d_O (length);
  std::vector<int> d_Y (length);
  std::vector<float> d_dX1 (length);
  std::vector<float> d_dX2 (length);

  // warmup
  for (int i = 0; i < repeat; i++) {
    MRCGradient(length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
    MRCGradient2(length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) 
    MRCGradient(length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of MRCGradient kernel: " << (time * 1e-3f) / repeat << " (us)" << std::endl;

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) 
    MRCGradient2(length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of MRCGradient2 kernel: " << (time * 1e-3f) / repeat << " (us)" << std::endl;

  // verify
  r_dX1 = d_dX1;
  r_dX2 = d_dX2;

  bool ok = true;
  for (int i = 0; i < length; i++) {
    if (std::fabs(h_dX1[i] - r_dX1[i]) > 1e-3 || std::fabs(h_dX2[i] - r_dX2[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  return 0;
}