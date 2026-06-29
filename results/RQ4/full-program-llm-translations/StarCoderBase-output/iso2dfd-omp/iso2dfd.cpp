#include <iostream>
#include <chrono>
#include <omp.h>

void iso_2dfd_iteration_cpu(float* next, float* prev, float* vel,
                            const float dtDIVdxy, size_t nRows, size_t nCols,
                            int nIterations) {
  float* swap;
  float value = 0.f;
  for (int k = 0; k < nIterations; k += 1) {
    for (size_t i = 1; i < nRows - 1; i += 1) {
      for (size_t j = 1; j < nCols - 1; j += 1) {
        size_t gid = j + (i * nCols);
        value = 0.f;
        value += prev[gid + 1] - 2.f * prev[gid] + prev[gid - 1];
        value += prev[gid + nCols] - 2.f * prev[gid] + prev[gid - nCols];
        value *= dtDIVdxy * vel[gid];
        next[gid] = 2.f * prev[gid] - next[gid] + value;
      }
    }
    swap = next;
    next = prev;
    prev = swap;
  }
}

int main() {
  constexpr size_t nRows = 1000;
  constexpr size_t nCols = 1000;
  constexpr size_t nIterations = 100;
  float* prev = new float[nRows * nCols];
  float* next = new float[nRows * nCols];
  float* vel = new float[nRows * nCols];
  float dtDIVdxy = 0.1f;

  auto start = std::chrono::steady_clock::now();
  iso_2dfd_iteration_cpu(next, prev, vel, dtDIVdxy, nRows, nCols, nIterations);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Host time: " << time << " ms" << std::endl;

  delete[] prev;
  delete[] next;
  delete[] vel;
  return 0;
}