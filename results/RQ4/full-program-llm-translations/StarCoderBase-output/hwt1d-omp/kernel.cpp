/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 */

#include "hwt.h"

void dwtHaar1D(const float *__restrict__ inSignal,
               float *__restrict__ coefsSignal,
               float *__restrict__ AverageSignal,
               const unsigned int tLevels,
               const unsigned int signalLength,
               const unsigned int levelsDone,
               const unsigned int mLevels)
{
  const float rsqrt_two = 0.7071f;
  const unsigned int localId = omp_get_thread_num();
  const unsigned int groupId = omp_get_thread_num();
  const unsigned int localSize = omp_get_num_threads();
  float *sharedArray = new float[localSize * 2];

  float t0 = inSignal[groupId * localSize * 2 + localId];
  float t1 = inSignal[groupId * localSize * 2 + localSize + localId];
  if (levelsDone == 0)
  {
    const float r = rsqrtf((float)signalLength);
    t0 *= r;
    t1 *= r;
  }
  sharedArray[localId] = t0;
  sharedArray[localSize + localId] = t1;

  #pragma omp barrier

  unsigned int levels = tLevels > mLevels ? mLevels : tLevels;
  unsigned int activeThreads = (1 << levels) / 2;
  unsigned int midOutPos = signalLength / 2;
  for (unsigned int i = 0; i < levels; ++i)
  {
    float data0, data1;
    if (localId < activeThreads)
    {
      data0 = sharedArray[2 * localId];
      data1 = sharedArray[2 * localId + 1];
    }

    #pragma omp barrier

    if (localId < activeThreads)
    {
      sharedArray[localId] = (data0 + data1) * rsqrt_two;
      const unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
      coefsSignal[globalPos] = (data0 - data1) * rsqrt_two;
      midOutPos >>= 1;
    }
    activeThreads >>= 1;

    #pragma omp barrier
  }

  if (localId == 0)
  {
    AverageSignal[groupId] = sharedArray[0];
  }
  delete[] sharedArray;
}

int runKernel(float *inData,
              float *dOutData,
              float *hOutData,
              float *dPartialOutData,
              const unsigned int signalLength,
              float *inDataBuf,
              float *partialOutDataBuf,
              float *outDataBuf)
{
  unsigned int levels = 0;
  int result = getLevels(signalLength, &levels);
  if (result == 1)
  {
    std::cerr << "getLevels() failed\n";
    return 1;
  }
  const unsigned int actualLevels = levels;
  const unsigned int maxLevelsOnDevice = 9;
  const unsigned int signalLengthByte = signalLength * sizeof(float);
  float *temp = new float[signalLength];
  std::copy(inData, inData + signalLength, temp);

  unsigned int levelsDone = 0;
  const unsigned int one = 1;
  unsigned int curLevels = 0;
  unsigned int curSignalLength;
  while (levelsDone < actualLevels)
  {
    curLevels = (levels < maxLevelsOnDevice) ? levels : maxLevelsOnDevice;
    if (levelsDone == 0)
    {
      curSignalLength = signalLength;
    }
    else
    {
      curSignalLength = one << levels;
    }
    const unsigned int groupSize = (one << curLevels) / 2;
    const unsigned int totalLevels = levels;
    std::copy(inData, inData + signalLength, inDataBuf);

    const unsigned int grid = (curSignalLength >> 1) / groupSize;
    const unsigned int block = groupSize;
  #pragma omp parallel num_threads(grid * block)
    {
      dwtHaar1D(inDataBuf,
                outDataBuf,
                partialOutDataBuf,
                totalLevels,
                curSignalLength,
                levelsDone,
                maxLevelsOnDevice);
    }
    std::copy(outDataBuf, outDataBuf + signalLength, dOutData);
    std::copy(partialOutDataBuf, partialOutDataBuf + signalLength, dPartialOutData);

    if (levels <= maxLevelsOnDevice)
    {
      dOutData[0] = dPartialOutData[0];
      std::copy(dOutData, dOutData + (one << curLevels), hOutData);
      std::copy(dOutData + (one << curLevels),
                dOutData + (signalLength - (one << curLevels)),
                hOutData + (one << curLevels));
      break;
    }
    else
    {
      levels -= maxLevelsOnDevice;
      std::copy(dOutData, dOutData + curSignalLength, hOutData);
      std::copy(dPartialOutData, dPartialOutData + (one << levels), inData);
      levelsDone += maxLevelsOnDevice;
    }
  }

  std::copy(temp, temp + signalLength, inData);
  delete[] temp;
  return 0;
}