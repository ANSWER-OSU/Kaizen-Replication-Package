void findMovingPixels(
  const size_t imgSize,
  const unsigned char *__restrict__ Img,
  const unsigned char *__restrict__ Img1,
  const unsigned char *__restrict__ Img2,
  const unsigned char *__restrict__ Tn,
        unsigned char *__restrict__ Mp) // moving pixel map
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
  const unsigned char *__restrict__ Img,
  const unsigned char *__restrict__ Mp,
        unsigned char *__restrict__ Bn)
{
  #pragma omp parallel for
  for (size_t i = 0; i < imgSize; i++) {
    if ( Mp[i] == 0 ) Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];
  }
}

// alpha = 0.92, c = 3
void updateThreshold(
  const size_t imgSize,
  const unsigned char *__restrict__ Img,
  const unsigned char *__restrict__ Mp,
  const unsigned char *__restrict__ Bn,
        unsigned char *__restrict__ Tn)
{
  #pragma omp parallel for
  for (size_t i = 0; i < imgSize; i++) {
    if (Mp[i] == 0) {
      float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
      Tn[i] = fmaxf(th, 20.f);
    }
  }
}

//
// merge three kernels into a single kernel
//
void merge(
  const size_t imgSize,
  const unsigned char *__restrict__ Img,
  const unsigned char *__restrict__ Img1,
  const unsigned char *__restrict__ Img2,
        unsigned char *__restrict__ Tn,
        unsigned char *__restrict__ Bn)
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