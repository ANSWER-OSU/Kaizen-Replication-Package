#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

// Reference
// https://pytorch.org/docs/stable/generated/torch.linalg.cross.html#torch.linalg.cross

template <typename T>
void cross1(int numel, T* out, const T* x1, const T* x2) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < numel; i++) {
    int base_index = i * 3;

    const T x1_c0 = x1[base_index + 0];
    const T x1_c1 = x1[base_index + 1];
    const T x1_c2 = x1[base_index + 2];
    const T x2_c0 = x2[base_index + 0];
    const T x2_c1 = x2[base_index + 1];
    const T x2_c2 = x2[base_index + 2];

    T val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1;
    T val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2;
    T val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0;

    out[base_index + 0] = val0;
    out[base_index + 1] = val1;
    out[base_index + 2] = val2;
  }
}

template <typename T>
void eval(const int nrows, const int repeat) {
  const int num_elems = nrows * 3;
  const int size_bytes = num_elems * sizeof(T);

  std::vector<T> a(num_elems);
  std::vector<T> b(num_elems);
  std::vector<T> o(num_elems);
  std::vector<T> o2(num_elems);
  std::vector<T> o3(num_elems);

  std::default_random_engine g(123);
  std::uniform_real_distribution<T> distr(-2.f, 2.f);
  for (int i = 0; i < num_elems; i++) {
    a[i] = distr(g);
    b[i] = distr(g);
  }

  std::vector<T> ref1(num_elems);
  std::vector<T> ref2(num_elems);
  std::vector<T> ref3(num_elems);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cross1(nrows, o.data(), a.data(), b.data());
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("Average execution time of cross1 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  bool ok = true;
  for (int i = 0; i < num_elems; i++) {
    if (fabs(o[i] - o2[i]) > 1e-3 || fabs(o[i] - o3[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of rows in a 2D tensor> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  printf("=========== Data type is FP32 ==========\n");
  eval<float>(nrows, repeat);

  return 0;
}