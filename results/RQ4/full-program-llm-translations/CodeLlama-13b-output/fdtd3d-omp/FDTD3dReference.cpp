#include <iostream>
#include <fstream>
#include <cmath>

// OpenMP header file
#include <omp.h>

// Function to generate random data
void generateRandomData(float* data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound)
{
    srand(0);

    for (int iz = 0; iz < dimz; iz++) {
        for (int iy = 0; iy < dimy; iy++) {
            for (int ix = 0; ix < dimx; ix++) {
                *data = (float)(lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound));
                ++data;
            }
        }
    }
}

// Function to generate pattern data
void generatePatternData(float* data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound)
{
    for (int iz = 0; iz < dimz; iz++) {
        for (int iy = 0; iy < dimy; iy++) {
            for (int ix = 0; ix < dimx; ix++) {
                *data = (float)(lowerBound + ((float)iz / (float)dimz) * (upperBound - lowerBound));
                ++data;
            }
        }
    }
}

// Function to implement FDTD (naive method)
bool fdtdReference(float* output, const float* input, const float* coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps)
{
    bool ok = true;
    const int     outerDimx    = dimx + 2 * radius;
    const int     outerDimy    = dimy + 2 * radius;
    const int     outerDimz    = dimz + 2 * radius;
    const size_t  volumeSize   = outerDimx * outerDimy * outerDimz;
    const int     stride_y     = outerDimx;
    const int     stride_z     = stride_y * outerDimy;
    float        *intermediate = 0;
    const float  *bufsrc       = 0;
    float        *bufdst       = 0;
    float        *bufdstnext   = 0;

    // Allocate temporary buffer
    if (ok)
    {
        intermediate = (float*) calloc(volumeSize, sizeof(float));
    }

    // Decide which buffer to use first (result should end up in output)
    if (ok)
    {
        if ((timesteps % 2) == 0)
        {
            bufsrc     = input;
            bufdst     = intermediate;
            bufdstnext = output;
        }
        else
        {
            bufsrc     = input;
            bufdst     = output;
            bufdstnext = intermediate;
        }
    }

    // Run the FDTD (naive method) using OpenMP
    omp_set_num_threads(4);
    #pragma omp parallel for default(none) shared(output, input, coeff, dimx, dimy, dimz, radius, timesteps) private(intermediate, bufsrc, bufdst, bufdstnext)
    for (int it = 0; it < timesteps; ++it)
    {
        // Determine variables for this timestep
        const float *src = bufsrc;
        float *dst = bufdst;

        // Update the values in the output array
        #pragma omp parallel for default(none) shared(output, input, coeff, dimx, dimy, dimz, radius, timesteps) private(bufsrc, bufdst, bufdstnext)
        for (int iz = -radius ;; iz < dimz + radius; ++iz)
        {
            for (int iy = -radius ;; iy < dimy + radius; ++iy)
            {
                for (int ix = -radius ;; ix < dimx + radius; ++ix)
                {
                    if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
                    {
                        float value = (*src) * coeff[0];

                        #pragma omp parallel for default(none) shared(output, input, coeff, dimx, dimy, dimz, radius, timesteps) private(bufsrc, bufdst, bufdstnext, value)
                        for(int ir = 1 ;; ir <= radius; ++ir)
                        {
                            value += coeff[ir] * (*(src + ir) + *(src - ir));                       // horizontal
                            value += coeff[ir] * (*(src + ir * stride_y) + *(src - ir * stride_y)); // vertical
                            value += coeff[ir] * (*(src + ir * stride_z) + *(src - ir * stride_z)); // in front & behind
                        }
                        *dst = value;
                    }
                    else
                    {
                        *dst = *src;
                    }
                    ++dst;
                    ++src;
                }
            }
        }
        // Rotate buffers
        float* tmp = bufdst;
        bufdst = bufdstnext;
        bufdstnext = tmp;
        bufsrc = (const float*) tmp;
    }

    if (intermediate)
        free(intermediate);

    return ok;
}

// Function to compare data
bool compareData(const float* output, const float* reference, const int dimx, const int dimy, const int dimz, const int radius, const float tolerance)
{
    bool ok = true;

    for (int iz = -radius; iz < dimz + radius; iz++)
    {
        for (int iy = -radius; iy < dimy + radius; iy++)
        {
            for (int ix = -radius; ix < dimx + radius; ix++)
            {
                if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
                {
                    // Determine the absolute difference
                    float difference = abs(*reference - *output);
                    float error;

                    // Determine the relative error
                    if (*reference != 0)
                        error = difference / *reference;
                    else
                        error = difference;

                    // Check the error is within the tolerance
                    if (error > tolerance)
                    {
                        ok = false;
                        return ok;
                    }
                }
                ++output;
                ++reference;
            }
        }
    }

    return ok;
}

int main()
{
    // Set problem parameters
    const int dimx = 1000, dimy = 1000, dimz = 1000;
    const int radius = 1;
    const int timesteps = 1000;
    const float tolerance = 1e-5;
    
    // Allocate memory for input, reference and output arrays
    float* input = (float*) calloc(dimx * dimy * dimz, sizeof(float));
    float* reference = (float*) calloc(dimx * dimy * dimz, sizeof(float));
    float* output = (float*) calloc(dimx * dimy * dimz, sizeof(float));

    // Generate random data and compute reference solution
    generateRandomData(input, dimx, dimy, dimz, 0, 1);
    fdtdReference(reference, input, NULL, dimx, dimy, dimz, radius, timesteps);

    // Compute FDTD solution using OpenMP
    fdtdReference(output, input, NULL, dimx, dimy, dimz, radius, timesteps);

    // Compare results
    bool ok = compareData(output, reference, dimx, dimy, dimz, radius, tolerance);

    if (ok)
        std::cout << "Results match!" << std::endl;
    else
        std::cout << "Results do not match!" << std::endl;

    // Free memory
    free(input);
    free(reference);
    free(output);

    return 0;
}