#include <iostream>
#include <iomanip>
#include <omp.h>

const int k_dim_min = 10, k_dim_max = 100, k_timesteps_min = 10, k_timesteps_max = 1000, k_radius_default = 1;
const int k_dim_qa = 100, k_timesteps_default = 100;
const float k_lowerBound = 0.0f, k_upperBound = 1.0f;

void generateRandomData(float *data, int dimx, int dimy, int dimz, float lowerBound, float upperBound) {
    for (int i = 0; i < dimx * dimy * dimz; ++i) {
        data[i] = k_lowerBound + (k_upperBound - k_lowerBound) * random() / RAND_MAX;
    }
}

bool fdtdReference(float *output, float *input, float *coeff, int dimx, int dimy, int dimz, int radius, int timesteps) {
    for (int t = 0; t < timesteps; ++t) {
        // Compute the output at this timestep
        #pragma omp parallel for
        for (int z = 0; z < dimz; ++z) {
            for (int y = 0; y < dimy; ++y) {
                for (int x = 0; x < dimx; ++x) {
                    int index = z * dimy * dimz + y * dimz + x;
                    output[index] = 0.0f;
                    for (int i = 0; i <= radius; ++i) {
                        int k = index - i * dimy * dimz;
                        if (k >= 0) {
                            output[index] += coeff[i] * input[k];
                        }
                        k = index + i * dimy * dimz;
                        if (k < volumeSize) {
                            output[index] += coeff[i] * input[k];
                        }
                    }
                }
            }
        }
        // Store the new input for the next timestep
        #pragma omp parallel for
        for (int z = 0; z < dimz; ++z) {
            for (int y = 0; y < dimy; ++y) {
                for (int x = 0; x < dimx; ++x) {
                    int index = z * dimy * dimz + y * dimz + x;
                    input[index] = output[index];
                }
            }
        }
    }
    return true;
}

bool fdtdGPU(float *output, float *input, float *coeff, int dimx, int dimy, int dimz, int radius, int timesteps, int argc, char **argv, const bool noprompt = false) {
    #pragma omp parallel num_threads(4)
    for (int t = 0; t < timesteps; ++t) {
        // Compute the output at this timestep
        #pragma omp for
        for (int z = 0; z < dimz; ++z) {
            for (int y = 0; y < dimy; ++y) {
                for (int x = 0; x < dimx; ++x) {
                    int index = z * dimy * dimz + y * dimz + x;
                    output[index] = 0.0f;
                    for (int i = 0; i <= radius; ++i) {
                        int k = index - i * dimy * dimz;
                        if (k >= 0) {
                            output[index] += coeff[i] * input[k];
                        }
                        k = index + i * dimy * dimz;
                        if (k < volumeSize) {
                            output[index] += coeff[i] * input[k];
                        }
                    }
                }
            }
        }
        // Store the new input for the next timestep
        #pragma omp for
        for (int z = 0; z < dimz; ++z) {
            for (int y = 0; y < dimy; ++y) {
                for (int x = 0; x < dimx; ++x) {
                    int index = z * dimy * dimz + y * dimz + x;
                    input[index] = output[index];
                }
            }
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int defaultDim;
    int dimx;
    int dimy;
    int dimz;
    int outerDimx;
    int outerDimy;
    int outerDimz;
    int radius;
    int timesteps;
    size_t volumeSize;
    const float lowerBound = 0.0f;
    const float upperBound = 1.0f;

    const float tolerance = 0.0001f;

    // Determine default dimensions
    defaultDim = (int)floor(pow((memsize / (2.0 * sizeof(float))), 1.0/3.0));
    int roundTarget = 128 / sizeof(float);
    defaultDim = defaultDim / roundTarget * roundTarget;
    defaultDim -= k_radius_default * 2;

    // Check dimension is valid
    if (defaultDim < k_dim_min) {
        shrLogEx(LOGBOTH | ERRORMSG, -1000, STDERROR);
        shrLog("\tinsufficient device memory (maximum volume on device is %d, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
        return false;
    }
    else if (defaultDim > k_dim_max) {
        defaultDim = k_dim_max;
    }

    // For QA testing, override default volume size
    if (shrCheckCmdLineFlag(argc, argv, "qatest")) {
        defaultDim = MIN(defaultDim, k_dim_qa);
    }

    // Parse command line arguments
    char *dim = 0;
    if (shrGetCmdLineArgumentstr(argc, argv, "dimx", &dim)) {
        dimx = (int)atoi(dim);
        if (dimx < k_dim_min || dimx > k_dim_max) {
            shrLogEx(LOGBOTH | ERRORMSG, -1001, STDERROR);
            shrLog("\tdimx out of range (%d requested, must be between %d and %d), see header files for details.\n", dimx, k_dim_min, k_dim_max);
            return false;
        }
    }
    else {
        dimx = defaultDim;
    }
    if (shrGetCmdLineArgumentstr(argc, argv, "dimy", &dim)) {
        dimy = (int)atoi(dim);
        if (dimy < k_dim_min || dimy > k_dim_max) {
            shrLogEx(LOGBOTH | ERRORMSG, -1002, STDERROR);
            shrLog("\tdimy out of range (%d requested, must be between %d and %d), see header files for details.\n", dimy, k_dim_min, k_dim_max);
            return false;
        }
    }
    else {
        dimy = defaultDim;
    }
    if (shrGetCmdLineArgumentstr(argc, argv, "dimz", &dim)) {
        dimz = (int)atoi(dim);
        if (dimz < k_dim_min || dimz > k_dim_max) {
            shrLogEx(LOGBOTH | ERRORMSG, -1003, STDERROR);
            shrLog("\tdimz out of range (%d requested, must be between %d and %d), see header files for details.\n", dimz, k_dim_min, k_dim_max);
            return false;
        }
    }
    else {
        dimz = defaultDim;
    }

    radius = k_radius_default;

    if (shrGetCmdLineArgumentstr(argc, argv, "timesteps", &dim)) {
        timesteps = (int)atoi(dim);
        if (timesteps < k_timesteps_min || radius >= k_timesteps_max) {
            shrLogEx(LOGBOTH | ERRORMSG, -1005, STDERROR);
            shrLog("\ttimesteps out of range (%d requested, must be between %d and %d), see header files for details.\n", timesteps, k_timesteps_min, k_timesteps_max);
            return false;
        }
    }
    else {
        timesteps = k_timesteps_default;
    }
    if (dim) free(dim);

    // Determine volume size
    outerDimx = dimx + 2 * radius;
    outerDimy = dimy + 2 * radius;
    outerDimz = dimz + 2 * radius;
    volumeSize = outerDimx * outerDimy * outerDimz;

    // Allocate memory
    float *host_output = (float *)calloc(volumeSize, sizeof(float));
    float *input = (float *)malloc(volumeSize * sizeof(float));
    float *coeff = (float *)malloc((radius + 1) * sizeof(float));

    // Create coefficients
    for (int i = 0; i <= radius; ++i) {
        coeff[i] = 0.1f;
    }

    // Generate data
    generateRandomData(input, outerDimx, outerDimy, outerDimz, lowerBound, upperBound);

    shrLog("\nFDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);

    // Execute on the host
    shrLog("fdtdReference...\n");
    fdtdReference(host_output, input, coeff, dimx, dimy, dimz, radius, timesteps);
    shrLog("fdtdReference complete\n");

    // Allocate memory
    float *device_output = (float *)calloc(volumeSize, sizeof(float));

    // Execute on the device
    shrLog("fdtdGPU...\n");
    fdtdGPU(device_output, input, coeff, dimx, dimy, dimz, radius, timesteps);
    shrLog("fdtdGPU complete\n");

    // Compare the results
    shrLog("\nCompareData (tolerance %f)...\n", tolerance);
    bool ok = compareData(device_output, host_output, dimx, dimy, dimz, radius, tolerance);

    if (ok) {
        if (input) free(input);
        if (coeff) free(coeff);
        if (host_output) free(host_output);
        if (device_output) free(device_output);
    }

    return ok;
}

bool runTest(int argc, const char **argv) {
    return main(argc, (char **)argv);
}

void showHelp(const int argc, const char **argv) {
    if (argc > 0)
        std::cout << std::endl << argv[0] << std::endl;
    std::cout << std::endl
              << "Syntax:" << std::endl
              << "    "
              << std::setw(20) << "--dimx=<N>"
              << "Specify number of elements in x direction (excluding halo)"
              << std::endl
              << "    " << std::setw(20) << "--dimy=<N>"
              << "Specify number of elements in y direction (excluding halo)"
              << std::endl
              << "    " << std::setw(20) << "--dimz=<N>"
              << "Specify number of elements in z direction (excluding halo)"
              << std::endl
              << "    " << std::setw(20) << "--radius=<N>"
              << "Specify radius of stencil" << std::endl
              << "    " << std::setw(20) << "--timesteps=<N>"
              << "Specify number of timesteps" << std::endl
              << "    " << std::setw(20) << "--noprompt"
              << "Skip prompt before exit";
    std::cout << std::endl;
}