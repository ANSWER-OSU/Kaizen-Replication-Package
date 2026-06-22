void dwtHaar1D( const float *__restrict__ inSignal, float *__restrict__ coefsSignal, float *__restrict__ AverageSignal, const unsigned int tLevels, const unsigned int signalLength, const unsigned int levelsDone, const unsigned int mLevels)
{
    size_t localId = omp_get_thread_num();
    size_t groupId = omp_get_team_num();
    size_t localSize = omp_get_num_threads();

    float sharedArray[2*localSize];
    
    /**
     * Read input signal data from global memory
     * to shared memory
     */
    float t0 = inSignal[groupId * localSize * 2 + localId];
    float t1 = inSignal[groupId * localSize * 2 + localSize + localId];
    // Divide with signal length for normalized decomposition
    if(0 == levelsDone)
    {
       float r = rsqrtf(signalLength);
       t0 *= r;
       t1 *= r;
    }
    sharedArray[localId] = t0;
    sharedArray[localSize + localId] = t1;
     
    #pragma omp barrier
    
    unsigned int levels = tLevels > mLevels ? mLevels: tLevels;
    unsigned int activeThreads = (1 << levels) / 2;
    unsigned int midOutPos = signalLength / 2;
    
    const float rsqrt_two = 0.7071f;
    for(unsigned int i = 0; i < levels; ++i)
    {

        float data0, data1;
        if(localId < activeThreads)
        {
            data0 = sharedArray[2 * localId];
            data1 = sharedArray[2 * localId + 1];
        }

        /* make sure all work items have read from sharedArray before modifying it */
        #pragma omp barrier

        if(localId < activeThreads)
        {
            sharedArray[localId] = (data0 + data1) * rsqrt_two;
            unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
            coefsSignal[globalPos] = (data0 - data1) * rsqrt_two;
       
            midOutPos >>= 1;
        }
        activeThreads >>= 1;
        #pragma omp barrier
    }
    
    /**
     * Write 0th element for the next decomposition
     * steps which are performed on host 
     */
    
     if(0 == localId)
        AverageSignal[groupId] = sharedArray[0];
}

int runKernel(
    float *inData, 
    float *dOutData, 
    float *hOutData, 
    float *dPartialOutData,
    const unsigned int signalLength)
{
  unsigned int levels = 0;

  int result = getLevels(signalLength, &levels);
  if (result == 1) {
    std::cerr << "getLevels() failed\n";
    return 1;
  }

  unsigned int actualLevels = levels;

  //max levels on device should be decided by kernelWorkGroupSize
  //int tempVar = (int)(log((float)kernelInfo.kernelWorkGroupSize) / log((float)2));
  //int maxLevelsOnDevice = tempVar + 1;
  // Assume work group size is 256
  const int maxLevelsOnDevice = 9;

  const int signalLengthByte = signalLength * sizeof(float);
  float* temp = (float*)malloc(signalLengthByte);
  memcpy(temp, inData, signalLengthByte);

  int levelsDone = 0;
  int one = 1;
  unsigned int curLevels = 0;
  unsigned int curSignalLength;
  while((unsigned int)levelsDone < actualLevels)
  {
    curLevels = (levels < maxLevelsOnDevice) ? levels : maxLevelsOnDevice;

    // Set the signal length for current iteration
    if(levelsDone == 0)
    {
      curSignalLength = signalLength;
    }
    else
    {
      curSignalLength = (one << levels);
    }

    // Set group size
    unsigned int groupSize = (1 << curLevels) / 2;

    unsigned int totalLevels = levels;

    #pragma omp target teams distribute parallel for num_teams(curSignalLength / groupSize) num_threads(groupSize / 2)
      dwtHaar1D(inData, dOutData, dPartialOutData, totalLevels, curSignalLength, levelsDone, maxLevelsOnDevice);

    if(levels <= maxLevelsOnDevice)
    {
      dOutData[0] = dPartialOutData[0];
      memcpy(hOutData, dOutData, (one << curLevels) * sizeof(float));
      memcpy(dOutData + (one << curLevels), hOutData + (one << curLevels), 
          (signalLength  - (one << curLevels)) * sizeof(float));
      break;
    }
    else
    {
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