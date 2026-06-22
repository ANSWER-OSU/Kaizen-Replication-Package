#include <omp.h>

int getLevels(unsigned int length, unsigned int* levels);
void calApproxFinalOnHost(float *inData, float *hOutData, unsigned int signalLength);

int runKernel(
    float *inData, 
    float *dOutData, 
    float *hOutData, 
    float *dPartialOutData,
    const unsigned int signalLength,
    float *inDataBuf,
    float *partialOutDataBuf,
    float *outDataBuf)
{
    int threads_per_block;
    int num_blocks;
    int num_devices;
    float *d_inData;
    float *d_partialOutData;
    int i;
    int num_levels = getLevels(signalLength, levels);

    #pragma omp parallel
    {
        num_devices = omp_get_num_devices();

        for (i = 0; i < num_devices; i++)
        {
            omp_set_default_device(i);

            cudaMalloc((void **)&d_inData, signalLength * sizeof(float));
            cudaMalloc((void **)&d_partialOutData, signalLength * sizeof(float));

            cudaMemcpy(d_inData, inData, signalLength * sizeof(float), cudaMemcpyHostToDevice);

            threads_per_block = 256;
            num_blocks = (signalLength + 256 - 1) / 256;
            for (i = 0; i < num_levels; i++)
            {
                if (levels[i] & 1) // If the current DWT level is odd
                {
                    dwthaar1d_kernel_odd<<<num_blocks, threads_per_block, 0>>>(d_inData, d_partialOutData, signalLength);
                    cudaMemcpy(d_inData, d_partialOutData, signalLength * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                else // If the current DWT level is even
                {
                    dwthaar1d_kernel_even<<<num_blocks, threads_per_block, 0>>>(d_inData, d_partialOutData, signalLength);
                    cudaMemcpy(d_inData, d_partialOutData, signalLength * sizeof(float), cudaMemcpyDeviceToDevice);
                }
            }
            calApproxFinalOnHost(inData, hOutData, signalLength);

            cudaFree(d_inData);
            cudaFree(d_partialOutData);
        }

        omp_set_default_device(0);
    }

    return 0;
}