void runKernel(const float *inData, float *hOutData, float *dOutData, float *dPartialOutData, 
               unsigned int signalLength, float *inDataBuf, float *partialOutDataBuf, float *outDataBuf)
{
  cudaMemcpy(inDataBuf, inData, sizeof(float) * signalLength, cudaMemcpyHostToDevice);
  
  #pragma omp target data map(to: inDataBuf[0:signalLength]) \
                        map(tofrom: outDataBuf[0:signalLength]) 
  {
    #pragma omp target teams distribute num_teams(NUM_BLOCKS) 
    for(int level = 0; level < levels; level++) {
      #pragma omp parallel for
      for(unsigned int t = 0; t < NUM_BLOCKS; t++) {
        unsigned int i = t * BLOCK_SIZE + omp_get_thread_num();
        if (i < signalLength) {
          partialOutDataBuf[i] = calcApproxSimple(inDataBuf, i, level);
        }
      }
      #pragma omp target update from(partialOutDataBuf[0:signalLength])

      #pragma omp parallel for 
      for(unsigned int t = 0; t < NUM_BLOCKS; t++) {
        unsigned int i = t * BLOCK_SIZE + omp_get_thread_num();
        if (i < signalLength) {
          outDataBuf[i] += partialOutDataBuf[i];
        }
      }
    }
  }

  cudaMemcpy(dOutData, outDataBuf, sizeof(float) * signalLength, cudaMemcpyDeviceToHost);
}