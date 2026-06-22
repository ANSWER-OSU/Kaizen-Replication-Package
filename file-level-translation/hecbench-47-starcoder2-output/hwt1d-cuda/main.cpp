#include <omp.h>

template<typename T>
T roundToPowerOf2(T val)
{
  int bytes = sizeof(T);
  val--;
  for(int i = 0; i < bytes; i++)
    val |= val >> (1<<i);
  val++;
  return val;
}

void runKernel(float* inData, float* hOutData, float* dOutData, float* dPartialOutData,
               unsigned int signalLength, float* inDataBuf, float* partialOutDataBuf, float* outDataBuf)
{
  cudaMemcpy(inDataBuf, inData, sizeof(float) * signalLength, cudaMemcpyHostToDevice);
  cudaMemcpy(outDataBuf, dOutData, sizeof(float) * signalLength, cudaMemcpyHostToDevice);
  cudaMemcpy(partialOutDataBuf, dPartialOutData, sizeof(float) * signalLength, cudaMemcpyHostToDevice);

  unsigned int levels = 0;
  HostTree tree;
  getTree(signalLength, &tree, &levels);
  unsigned int blocksPerGrid = tree.gridSize;
  unsigned int threadsPerBlock = tree.blockSize;
  unsigned int sharedMemSize = tree.sharedMemSize;

  HaarTransformKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(inDataBuf, partialOutDataBuf, outDataBuf, signalLength, levels);

  cudaMemcpy(dOutData, outDataBuf, sizeof(float) * signalLength, cudaMemcpyDeviceToHost);
  cudaMemcpy(dPartialOutData, partialOutDataBuf, sizeof(float) * signalLength, cudaMemcpyDeviceToHost);
}

void calApproxFinalOnHost(float* inData, float* hOutData, unsigned int signalLength)
{
  float* approxOutData = (float*)malloc(signalLength * sizeof(float));
  float* diffOutData = (float*)malloc(signalLength * sizeof(float));
  float* approxPartialOutData = (float*)malloc(signalLength * sizeof(float));
  float* diffPartialOutData = (float*)malloc(signalLength * sizeof(float));
  float* spec = (float*)fftwf_malloc(sizeof(float) * signalLength);
  fftwf_plan plan = fftwf_plan_dft_1d(signalLength, spec, spec, FFTW_FORWARD, FFTW_ESTIMATE);

  for(size_t i = 0; i < signalLength; i++)
  {
    spec[i] = inData[i];
  }

  fftwf_execute(plan);

  float firstElReal = spec[0].re;
  float firstElImag = spec[0].im;

  float mean = firstElReal / signalLength;

  float* amp = (float*)malloc(sizeof(float) * signalLength);

  for(size_t i = 0; i < signalLength; i++)
  {
    float real = spec[i].re - mean;
    float imag = spec[i].im;
    float mag = sqrtf(real * real + imag * imag);
    amp[i] = mag;
  }

  // Sort the amplitudes
  msosrt(amp, 0, signalLength - 1);

  // Find the 95%-ile
  float ninetyfiveTile = amp[(int)(signalLength * .95f)];

  float totalSum = 0.f;
  for(size_t i = 0; i < signalLength; i++)
  {
    totalSum += amp[i];
  }

  float cumulativeSum = 0.f;
  for(size_t i = 0; i < signalLength; i++)
  {
    cumulativeSum += amp[i];
    if(cumulativeSum / totalSum >= 0.95f)
    {
      ninetyfiveTile = amp[i];
      break;
    }
  }

  // Find elbow
  size_t k = 0;
  for(size_t i = 0; i < signalLength; i++)
  {
    if(amp[i] == ninetyfiveTile)
    {
      k = i;
      break;
    }
  }

  float* r = (float*)malloc(sizeof(float) * k);
  float* ell = (float*)malloc(sizeof(float) * k);

  float* expo = (float*)malloc(sizeof(float) * k);

  float* divident = (float*)malloc(sizeof(float) * k);
  float* divisor = (float*)malloc(sizeof(float) * k);

  float* rD = (float*)malloc(sizeof(float) * k);
  float* ellD = (float*)malloc(sizeof(float) * k);

  float* expoD = (float*)malloc(sizeof(float) * k);

  float* dividentD = (float*)malloc(sizeof(float) * k);
  float* divisorD = (float*)malloc(sizeof(float) * k);

  fftwf_plan plan2 = fftwf_plan_dft_1d(k, spec, spec, FFTW_BACKWARD, FFTW_ESTIMATE);

  r[0] = firstElReal;
  ell[0] = firstElImag;
  for(size_t i = 1; i < k; i++)
  {
    r[i] = spec[i].re;
    ell[i] = spec[i].im;
  }

  float meanR = 0.f;
  float meanEll = 0.f;
  float meanDivident = 0.f;
  float meanDivisor = 0.f;
  for(size_t i = 0; i < k; i++)
  {
    meanR += r[i];
    meanEll += ell[i];
  }
  meanR /= k;
  meanEll /= k;
  for(size_t i = 0; i < k; i++)
  {
    r[i] -= meanR;
    ell[i] -= meanEll;
  }
  for(size_t i = 0; i < k; i++)
  {
    expo[i] = expf(-fabsf(r[i]));
  }
  for(size_t i = 0; i < k; i++)
  {
    divident[i] = r[i] * r[i] + ell[i] * ell[i];
    divisor[i] = r[i] * r[i] + ell[i] * ell[i] + expo[i] * expo[i];
  }
  for(size_t i = 0; i < k; i++)
  {
    rD[i] = r[i] * divident[i] / divisor[i];
    ellD[i] = ell[i] * divident[i] / divisor[i];
  }
  for(size_t i = 0; i < k; i++)
  {
    expoD[i] = expf(-fabsf(rD[i]));
  }
  for(size_t i = 0; i < k; i++)
  {
    dividentD[i] = rD[i] * rD[i] + ellD[i] * ellD[i];
    divisorD[i] = rD[i] * rD[i] + ellD[i] * ellD[i] + expoD[i] * expoD[i];
  }
  for(size_t i = 0; i < k; i++)
  {
    spec[i].re = rD[i] * dividentD[i] / divisorD[i];
    spec[i].im = ellD[i] * dividentD[i] / divisorD[i];
  }

  memset(spec + k, 0, sizeof(float) * (signalLength - k));

  fftwf_execute(plan2);

  for(size_t i = 0; i < signalLength; i++)
  {
    approxOutData[i] = spec[i];
  }

  memset(approxPartialOutData, 0, signalLength * sizeof(float));
  for(size_t i = 0; i < signalLength / 2; i++)
  {
    approxPartialOutData[i] = approxOutData[i];
  }

  for(size_t i = 0; i < signalLength; i++)
  {
    diffOutData[i] = dOutData[i] - approxOutData[i];
  }

  for(size_t i = 0; i < signalLength; i++)
  {
    diffPartialOutData[i] = dPartialOutData[i] - approxPartialOutData[i];
  }

  memset(hOutData, 0, signalLength * sizeof(float));

  if(signalLength % 32)
  {
    for(size_t i = 0; i < signalLength / 2; i++)
    {
      hOutData[i] += diffPartialOutData[i] + diffOutData[i];
      hOutData[signalLength - i - 1] += diffPartialOutData[i + signalLength / 2] + diffOutData[i + signalLength / 2];
    }
  }
  else
  {
    for(size_t i = 0; i < signalLength / 2; i++)
    {
      hOutData[i] += diffPartialOutData[i] + diffOutData[i];
      hOutData[signalLength - i - 1] += diffPartialOutData[i] + diffOutData[i];
    }
  }

  free(amp);
  free(r);
  free(ell);
  free(expo);
  free(divident);
  free(divisor);
  free(rD);
  free(ellD);
  free(expoD);
  free(dividentD);
  free(divisorD);
  free(approxOutData);
  free(diffOutData);
  free(approxPartialOutData);
  free(diffPartialOutData);
  fftwf_free(spec);
  fftwf_destroy_plan(plan);
  fftwf_destroy_plan(plan2);
}

int main(int argc, char * argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <signal length> <repeat>\n";
    return 1;
  }
  unsigned int signalLength = atoi(argv[1]);
  const int iterations = atoi(argv[2]);

  signalLength = roundToPowerOf2<unsigned int>(signalLength);

  unsigned int levels = 0;
  if (getLevels(signalLength, &levels) == 1) {
    std::cerr << "signalLength > 2 ^ 23 not supported\n";
    return 1;
  }

  float *inData = (float*)malloc(signalLength * sizeof(float));

  srand(2);
  for(unsigned int i = 0; i < signalLength; i++)
  {
    inData[i] = (float)(rand() % 10);
  }

  float *dOutData = (float*) malloc(signalLength * sizeof(float));

  memset(dOutData, 0, signalLength * sizeof(float));

  float *dPartialOutData = (float*) malloc(signalLength * sizeof(float));

  memset(dPartialOutData, 0, signalLength * sizeof(float));

  float *hOutData = (float*)malloc(signalLength * sizeof(float));

  memset(hOutData, 0, signalLength * sizeof(float));

  float *inDataBuf;
  cudaMalloc((void**)&inDataBuf, sizeof(float) * signalLength);

  float *partialOutDataBuf;
  cudaMalloc((void**)&partialOutDataBuf, sizeof(float) * signalLength);

  float *outDataBuf;
  cudaMalloc((void**)&outDataBuf, sizeof(float) * signalLength);

  std::cout << "Executing kernel for " 
            << iterations << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

#pragma omp parallel
  {
    runKernel(inData, hOutData, dOutData, dPartialOutData,
              signalLength, inDataBuf, partialOutDataBuf, outDataBuf);
  }

  calApproxFinalOnHost(inData, hOutData, signalLength);

  bool ok = true;
  for(unsigned int i = 0; i < signalLength; ++i)
  {
    if(fabs(dOutData[i] - hOutData[i]) > 0.1f)
    {
      ok = false;
      break;
    }
  }

  cudaFree(inDataBuf);
  cudaFree(outDataBuf);
  cudaFree(partialOutDataBuf);

  free(inData);
  free(dOutData);
  free(dPartialOutData);
  free(hOutData);

  if(ok)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  return 0;
}