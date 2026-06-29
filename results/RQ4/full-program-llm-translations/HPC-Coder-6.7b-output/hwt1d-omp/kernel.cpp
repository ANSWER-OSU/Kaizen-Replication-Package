#include <iostream>
#include <cmath>
#include <omp.h>

void dwtHaar1D(const float *inSignal, float *coefsSignal, float *AverageSignal,
               const unsigned int tLevels, const unsigned int signalLength,
               const unsigned int levelsDone, const unsigned int mLevels) {
  size_t localId = omp_get_thread_num();
  size_t groupId = omp_get_thread_num();
  size_t localSize = omp_get_num_threads();

  float sharedArray[localSize * 2]; // Assuming sharedArray is available

  // Read input signal data from global memory to shared memory
  float t0 = inSignal[groupId * localSize * 2 + localId];
  float t1 = inSignal[groupId * localSize * 2 + localSize + localId];
  if (levelsDone == 0) {
    float r = sqrtf(1.0f / (float)signalLength);
    t0 *= r;
    t1 *= r;
  }
  sharedArray[localId] = t0;
  sharedArray[localSize + localId] = t1;

  #pragma omp barrier;  // sync threads

  unsigned int levels = tLevels > mLevels ? mLevels : tLevels;
  unsigned int activeThreads = (1 << levels) / 2;
  unsigned int midOutPos = signalLength / 2;

  const float rsqrt_two = 0.7071f;
  for (unsigned int i = 0; i < levels; ++i) {
    float data0, data1;
    if (localId < activeThreads) {
      data0 = sharedArray[2 * localId];
      data1 = sharedArray[2 * localId + 1];
    }

    #pragma omp barrier; // sync threads

    if (localId < activeThreads) {
      sharedArray[localId] = (data0 + data1) * rsqrt_two;
      unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
      coefsSignal[globalPos] = (data0 - data1) * rsqrt_two;

      midOutPos >>= 1;
    }
    activeThreads >>= 1;
    #pragma omp barrier; // sync threads
  }

  // Write 0th element for the next decomposition steps
  if (localId == 0)
    AverageSignal[groupId] = sharedArray[0];
}

void getLevels(const unsigned int signalLength, unsigned int *levels) {
  *levels = (unsigned int)(std::log2(signalLength) - 1.0f) + 1;
}

int runKernel(float *inData, float *dOutData, float *hOutData,
              float *dPartialOutData, const unsigned int signalLength,
              float *inDataBuf, float *partialOutDataBuf, float *outDataBuf) {
  unsigned int levels = 0;

  int result = getLevels(signalLength, &levels);
  if (result == 1) {
    std::cerr << "getLevels() failed\n";
    return 1;
  }

  unsigned int actualLevels = levels;

  const int maxLevelsOnDevice = 9;

  const int signalLengthByte = signalLength * sizeof(float);
  float *temp = (float *)malloc(signalLengthByte);
  memcpy(temp, inData, signalLengthByte);

  int levelsDone = 0;
  int one = 1;
  unsigned int curLevels = 0;
  unsigned int curSignalLength;
  while ((unsigned int)levelsDone < actualLevels) {
    curLevels = (levels < maxLevelsOnDevice) ? levels : maxLevelsOnDevice;

    if (levelsDone == 0)
      curSignalLength = signalLength;
    else
      curSignalLength = (one << levels);

    unsigned int groupSize = (1 << curLevels) / 2;

    unsigned int totalLevels = levels;

    #pragma omp parallel num_threads(groupSize)
    dwtHaar1D(inDataBuf, outDataBuf, partialOutDataBuf, totalLevels,
              curSignalLength, levelsDone, maxLevelsOnDevice);

    memcpy(dOutData, outDataBuf, signalLengthByte);

    memcpy(dPartialOutData, partialOutDataBuf, signalLengthByte);

    if (levels <= maxLevelsOnDevice) {
      dOutData[0] = dPartialOutData[0];
      memcpy(hOutData, dOutData, (one << curLevels) * sizeof(float));
      memcpy(dOutData + (one << curLevels), hOutData + (one << curLevels),
             (signalLength - (one << curLevels)) * sizeof(float));
      break;
    } else {
      levels -= maxLevelsOnDevice;
      memcpy(hOutData, dOutData, curSignalLength * sizeof(float));
      memcpy(inData, dPartialOutData, (one << levels) * sizeof(float));
      levelsDone += (int)maxLevelsOnDevice;
    }
  }

  memcpy(inData, temp, signalLengthByte);
  free(temp);
  return 0;
}