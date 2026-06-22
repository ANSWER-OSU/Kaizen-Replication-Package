#include <iostream>
#include <chrono>
#include <limits>
#include <omp.h>

// rounds to a power of 2
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

int main(int argc, char * argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <signal length> <repeat>\n";
    return 1;
  }
  unsigned int signalLength = atoi(argv[1]);
  const int iterations = atoi(argv[2]);

  // signal length must be power of 2
  signalLength = roundToPowerOf2<unsigned int>(signalLength);

  unsigned int levels = 0;
  // omitted the check for levels

  // Allocate and init memory used by host
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

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < iterations; i++) {
    runKernel(inData, hOutData, dOutData, dPartialOutData,
              signalLength, inData, partialOutData, dOutData);
  } 

  auto start = std::chrono::steady_clock::now();

  calApproxFinalOnHost(inData, hOutData, signalLength);

  bool ok = true;
  #pragma omp parallel for reduction(||: ok)
  for(unsigned int i = 0; i < signalLength; ++i)
  {
    if(fabs(dOutData[i] - hOutData[i]) > 0.1f){
      ok = false;
    }
  }

  free(inData);
  free(dOutData);
  free(dPartialOutData);
  free(hOutData);

  if (ok)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  return 0;
}