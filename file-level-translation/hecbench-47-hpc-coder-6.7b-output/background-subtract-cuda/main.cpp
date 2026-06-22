#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <omp.h>

#define BLOCK_SIZE 256

void merge(
  const int imgSize,
  const std::vector<unsigned char> &Img,
  const std::vector<unsigned char> &Img1,
  const std::vector<unsigned char> &Img2,
  std::vector<unsigned char> &Bn,
  std::vector<unsigned char> &Tn,
  const std::vector<unsigned char> &Tn_ref)
{
  std::vector<unsigned char> Mp(imgSize);

  #pragma omp parallel num_threads(2) shared(imgSize, Img, Img1, Img2, Mp, Bn, Tn)
  {
    #pragma omp sections
    {
      #pragma omp section
      {
        size_t i = 0;
        #pragma omp parallel for schedule(static)
        for (; i < imgSize; i++) {
          if ( abs(Img[i] - Img1[i]) > Tn_ref[i] || abs(Img[i] - Img2[i]) > Tn_ref[i] )
            Mp[i] = 255;
          else Mp[i] = 0;
        }
      }

      #pragma omp section
      {
        size_t i = 0;
        #pragma omp parallel for schedule(static)
        for (; i < imgSize; i++) {
          if ( Mp[i] == 0 ) Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];
        }
      }

      #pragma omp section
      {
        size_t i = 0;
        #pragma omp parallel for schedule(static)
        for (; i < imgSize; i++) {
          if (Mp[i] == 0) {
            float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
            Tn[i] = fmaxf(th, 20.f);
          }
        }
      }
    }
  }
}

void updateBackground(
  const int imgSize,
  const std::vector<unsigned char> &Img,
  const std::vector<unsigned char> &Mp,
        std::vector<unsigned char> &Bn)
{
  #pragma omp parallel num_threads(2) shared(imgSize, Img, Mp, Bn)
  {
    #pragma omp sections
    {
      #pragma omp section
      {
        size_t i = 0;
        #pragma omp parallel for schedule(static)
        for (; i < imgSize; i++) {
          if ( Mp[i] == 0 ) Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cout << "Usage: " << argv[0] << " <image width> <image height> <merge> <repeat>" << std::endl;
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int merged = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int imgSize = width * height;
  std::vector<unsigned char> Img(imgSize);
  std::vector<unsigned char> Img1(imgSize);
  std::vector<unsigned char> Img2(imgSize);
  std::vector<unsigned char> Bn(imgSize);
  std::vector<unsigned char> Tn(imgSize, 128);
  std::vector<unsigned char> Tn_ref(imgSize, 128);

  std::mt19937 generator( 123 );
  std::uniform_int_distribution<int> distribute( 0, 255 );

  for (int j = 0; j < imgSize; j++) {
    Bn[j] = distribute(generator);
    Tn_ref[j] = 128;
  }

  long time = 0;

  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < imgSize; j++)
      Img[j] = distribute(generator);

    if (i >= 2) {
      {
        auto start = std::chrono::steady_clock::now();
        if (merged) merge(imgSize, Img, Img1, Img2, Bn, Tn, Tn_ref);
        else updateBackground(imgSize, Img, Mp, Bn);
        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
      merge_ref(imgSize, Img, Img1, Img2, Tn_ref, Bn);
    }
    std::swap(Img, Img2);
    std::swap(Img1, Img);
  }

  float kernel_time = (repeat <= 2) ? 0 : (time * 1e-3f) / (repeat - 2);
  std::cout << "Average kernel execution time: " << kernel_time << " (us)" << std::endl;

  // verification
  int max_error = 0;
  for (int i = 0; i < imgSize; i++) {
    if (std::abs(Tn[i] - Tn_ref[i]) > max_error)
      max_error = std::abs(Tn[i] - Tn_ref[i]);
  }
  for (int i = 0; i < imgSize; i++) {
    if (std::abs(Bn[i] - Bn[i]) > max_error)
      max_error = std::abs(Bn[i] - Bn[i]);
  }
  std::cout << "Max error is " << max_error << std::endl;

  std::cout << (max_error ? "FAIL" : "PASS") << std::endl;

  return 0;
}