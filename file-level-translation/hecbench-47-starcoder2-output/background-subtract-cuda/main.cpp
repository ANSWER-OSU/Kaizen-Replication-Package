#include <iostream>
#include <chrono>

const int BLOCK_SIZE = 256;

void findMovingPixels(
  const size_t imgSize,
  const unsigned char *Img,
  const unsigned char *Img1,
  const unsigned char *Img2,
  const unsigned char *Tn,
        unsigned char *Mp) // moving pixel map
{
  #pragma omp parallel for
  for (size_t i = 0; i < imgSize; i++) {
    if ( abs(Img[i] - Img1[i]) > Tn[i] || abs(Img[i] - Img2[i]) > Tn[i] )
      Mp[i] = 255;
    else {
      Mp[i] = 0;
    }
  }
}

// alpha = 0.92
void updateBackground(
  const size_t imgSize,
  const unsigned char *Img,
  const unsigned char *Mp,
        unsigned char *Bn)
{
  #pragma omp parallel for
  for (size_t i = 0; i < imgSize; i++) {
    if ( Mp[i] == 0 ) Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];
  }
}

// alpha = 0.92, c = 3
void updateThreshold(
  const size_t imgSize,
  const unsigned char *Img,
  const unsigned char *Mp,
  const unsigned char *Bn,
        unsigned char *Tn)
{
  #pragma omp parallel for
  for (size_t i = 0; i < imgSize; i++) {
    if (Mp[i] == 0) {
      float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
      Tn[i] = fmaxf(th, 20.f);
    }
  }
}

void merge(
  const size_t imgSize,
  const unsigned char *Img,
  const unsigned char *Img1,
  const unsigned char *Img2,
        unsigned char *Tn,
        unsigned char *Bn)
{
  #pragma omp parallel for
  for (size_t i = 0; i < imgSize; i++) {
    if ( abs(Img[i] - Img1[i]) <= Tn[i] && abs(Img[i] - Img2[i]) <= Tn[i] ) {
      // update background
      Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];

      // update threshold
      float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
      Tn[i] = fmaxf(th, 20.f);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cout << "Usage: " << argv[0] << " <image width> <image height> <merge> <repeat>" << std::endl;
    return 1;
  }

  const int width = std::stoi(argv[1]);
  const int height = std::stoi(argv[2]);
  const int merged = std::stoi(argv[3]);
  const int repeat = std::stoi(argv[4]);

  const int imgSize = width * height;
  unsigned char *Img = new unsigned char[imgSize];
  unsigned char *Img1 = new unsigned char[imgSize];
  unsigned char *Img2 = new unsigned char[imgSize];
  unsigned char *Bn = new unsigned char[imgSize];
  unsigned char *Bn_ref = new unsigned char[imgSize];
  unsigned char *Tn = new unsigned char[imgSize];
  unsigned char *Tn_ref = new unsigned char[imgSize];

  for (int j = 0; j < imgSize; j++) {
    Bn_ref[j] = Bn[j] = rand() % 256;
    Tn_ref[j] = Tn[j] = 128;
  }

  long time = 0;

  for (int i = 0; i < repeat; i++) {

    for (int j = 0; j < imgSize; j++) {
      Img[j] = rand() % 256;
    }

    if (i >= 2) {
      if (merged) {
        auto start = std::chrono::steady_clock::now();
        merge(imgSize, Img, Img1, Img2, Tn, Bn);
        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
      else {
        auto start = std::chrono::steady_clock::now();
        findMovingPixels(imgSize, Img, Img1, Img2, Tn, Tn);
        updateBackground(imgSize, Img, Tn, Bn);
        updateThreshold(imgSize, Img, Tn, Bn, Tn);
        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
      merge_ref(imgSize, Img, Img1, Img2, Tn_ref, Bn_ref);
    }

    unsigned char *t = Img2;
    Img2 = Img1;
    Img1 = Img;
    Img = t;
  }

  float kernel_time = (repeat <= 2) ? 0 : (time * 1e-3f) / (repeat - 2);
  std::cout << "Average kernel execution time: " << kernel_time << " (us)" << std::endl;

  // verification
  int max_error = 0;
  for (int i = 0; i < imgSize; i++) {
    if (abs(Tn[i] - Tn_ref[i]) > max_error)
      max_error = abs(Tn[i] - Tn_ref[i]);
  }
  for (int i = 0; i < imgSize; i++) {
    if (abs(Bn[i] - Bn_ref[i]) > max_error)
      max_error = abs(Bn[i] - Bn_ref[i]);
  }
  std::cout << "Max error is " << max_error << std::endl;

  std::cout << (max_error ? "FAIL" : "PASS") << std::endl;

  delete[] Img;
  delete[] Img1;
  delete[] Img2;
  delete[] Tn;
  delete[] Bn;
  delete[] Tn_ref;
  delete[] Bn_ref;

  return 0;
}