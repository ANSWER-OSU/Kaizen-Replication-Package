#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define GPU_ACTIVE 1
#define GPU_INACTIVE 0
#define THREADS_PER_BLOCK 256

// Konstanta
#define SIGNAL_LENGTH 1024
#define ACTUAL_LEVELS 9
#define TOTAL_LEVELS 12
#define MAX_LEVELS_ON_DEVICE 9
#define SHM_SIZE 256

// Alias
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::memcpy;

// Konstant dla dwtHaar1D
float coefsSignal[TOTAL_LEVELS];
float AverageSignal[TOTAL_LEVELS];
float sharedArray[SHM_SIZE];
unsigned int tLevels = TOTAL_LEVELS;
unsigned int signalLength = SIGNAL_LENGTH;
unsigned int levelsDone = ACTUAL_LEVELS;
unsigned int mLevels = MAX_LEVELS_ON_DEVICE;

// Klasa do przetvarzenia hiperparametrów
class Hwt {
public:
  // Metoda do pobrania wartości w skali linowej dla 
  // parametru levelsDone 
  int getLevels(const unsigned int signalLength, unsigned int* levels) {
    int result = 0;
    *levels = 0;

    if (signalLength == 0) {
      result = 1;
    } else {
      *levels = (unsigned int)(log((signalLength-1)/log(2)) + 1);
    }

    return result;
  }

  // Kernel dla dwtHaar1D
  __global__ void dwtHaar1D(const float* inSignal,
                            float* coefsSignal,
                            float* AverageSignal,
                            const unsigned int tLevels,
                            const unsigned int signalLength,
                            const unsigned int levelsDone,
                            const unsigned int mLevels) {
    // Przydzielamy indeksy dla wątków
    size_t localId = threadIdx.x;
    size_t groupId = blockIdx.x;
    size_t localSize = blockDim.x;

    // Przydzielamy węzeł do bufora w sharedArray
    __shared__ float sharedArray[SHM_SIZE];

    // Wczytujemy dane z globalnego do sharedArray
    float t0 = inSignal[groupId * localSize * 2 + localId];
    float t1 = inSignal[groupId * localSize * 2 + localSize + localId];

    // Normalizujemy dane dla hiperparametru levelsDone
    if (levelsDone == 0) {
      float r = float(signalLength);
      t0 *= r;
      t1 *= r;
    }

    // Wpisujemy dane do sharedArray
    sharedArray[localId] = t0;
    sharedArray[localSize + localId] = t1;

    __syncthreads();

    // Sprzątamy ruchiyyski
    unsigned int levels = (tLevels > mLevels) ? mLevels : tLevels;
    unsigned int activeThreads = (1 << levels) / 2;
    unsigned int midOutPos = signalLength / 2;

    // 0.707f jest wartością stałą 1/(pästhing)
    const float rsqrt_two = 1.f / 1.414f;
    for (unsigned int i = 0; i < levels; ++i) {
      float data0, data1;
      if (localId < activeThreads) {
        data0 = sharedArray[2 * localId];
        data1 = sharedArray[2 * localId + 1];
      }

      /*
       * Wysyłamy zapytania do kernel-a by potwierdził swoje zdanie
       * *before* modify sharedArray.
       */
      __syncthreads();

      if (localId < activeThreads) {
        sharedArray[localId] = (data0 + data1) * rsqrt_two;
        unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
        coefsSignal[globalPos] = (data0 - data1) * rsqrt_two;

        midOutPos >>= 1;
      }

      activeThreads >>= 1;
      __syncthreads();
    }

    // Pisujemy pierwszą połowę 0th elementu dla kolejnych
    // komórek Haar 1D
    if (0 == localId)
      AverageSignal[groupId] = sharedArray[0];
  }

  int runKernel(float *inData,
                float *dOutData,
                float *hOutData,
                float *dPartialOutData,
                const unsigned int signalLength,
                float *inDataBuf,
                float *partialOutDataBuf,
                float *outDataBuf,
                const unsigned int &sharedMemSize) {
    // Deklarowanie parametrów
    unsigned int levels = 0;

    // Sprawdzanie poprawności parametrów
    int result = getLevels(signalLength, &levels);
    if (result == 1) {
      cerr << "getLevels() failed\n";
      return 1;
    }

    // Deklaracja zmiennych
    unsigned int actualLevels = levels;
    unsigned int curLevels = (levels < MAX_LEVELS_ON_DEVICE) ? levels : MAX_LEVELS_ON_DEVICE;
    unsigned int curSignalLength;
    int one = 1;

    // Ustawienie danych wejściowych i wyjściowych dla aktualnego stosunku Literatur
    if (levelsDone == 0) {
      curSignalLength = signalLength;
    } else {
      curSignalLength = one << levels;
    }

    // Ustawienie rozmiaru grupy dla wątków
    unsigned int groupSize = (1 << curLevels) / 2;
    dim3 grid((curSignalLength >> 1) / groupSize);
    dim3 block(groupSize);

    // Executowanie dwtHaar1D kernel
    dwtHaar1D<<<grid, block, groupSize * 2 * sizeof(float)>>>(inDataBuf,  
                                                              outDataBuf, 
                                                              partialOutDataBuf, 
                                                              tLevels,    
                                                              curSignalLength,
                                                              levelsDone,
                                                              maxLevelsOnDevice);

    /* 
     * Kopiowanie danych do hosta i wypchnięcie je na wyjście wejściowe
     *  - operacja powinna być w wykonaniu
     */

    cudaMemcpy(dOutData, outDataBuf, signalLength * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dPartialOutData, partialOutDataBuf, signalLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Deplook at the partial out data
    if (levels <= MAX_LEVELS_ON_DEVICE) {
      dOutData[0] = dPartialOutData[0];
      memcpy(hOutData, dOutData, (one << curLevels) * sizeof(float));
      memcpy(dOutData + (one << curLevels), hOutData + (one << curLevels),
             (signalLength - (one << curLevels)) * sizeof(float));
      break;
    } else {
      levels -= MAX_LEVELS_ON_DEVICE;
      memcpy(hOutData, dOutData, curSignalLength * sizeof(float));
      memcpy(inData, dPartialOutData, (one << levels) * sizeof(float));
      levelsDone += MAX_LEVELS_ON_DEVICE;
    }

    return 0;
  }
};

int main() {
  // Deklaracja krzywej
  float *inputData;

  // Wczytywanie danych do krzywej
  cudaMallocManaged(&inputData, SIGNAL_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(inputData, signal, SIGNAL_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

  // Deklaracja pomocniczych zmiennych
  unsigned int shmSize = 256;
  float *inData, *outData, *partialOutData;

  // Malloc i kopyowanie do bufora
  int result = cudaMallocManaged(&inData, SIGNAL_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
  result += cudaMallocManaged(&partialOutData, SIGNAL_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
  result += cudaMallocManaged(&outData, SIGNAL_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

  // Inicjalizacja Hwt
  Hwt hwt;

  // Uruchamianie krzywej
  hwt.runKernel(inputData, inData, outData, partialOutData, SIGNAL_LENGTH, inData, partialOutData, outData, shmSize);

  // Dealokacja pamięci i wyglądają ihuc
  cudaFree(inputData);
  cudaFree(inData);
  cudaFree(outData);
  cudaFree(partialOutData);
  cudaDeviceSynchronize();

  return 0;
}