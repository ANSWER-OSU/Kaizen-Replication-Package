#include <cstddef>
#include <algorithm>

// OpenMP headers
#include <omp.h>

// Problem size along one side; total number of cells is this squared
#define NUM 1024

// Block size
#define BLOCK_SIZE 128

// Define Real and zero as floats
#define Real float
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f

// SOR relaxation parameter
const Real omega = 1.85f;

// Function to update temperature for red cells
void red_kernel (const Real *__restrict__ aP,
                 const Real *__restrict__ aW,
                 const Real *__restrict__ aE,
                 const Real *__restrict__ aS,
                 const Real *__restrict__ aN,
                 const Real *__restrict__ b,
                 const Real *__restrict__ temp_black,
                       Real *__restrict__ temp_red,
                       Real *__restrict__ norm_L2)
{
    int row = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int col = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (row > NUM / 2 || col > NUM) return;

    int ind_red = col * ((NUM >> 1) + 2) + row; // local (red) index
    int ind = 2 * row - (col & 1) - 1 + NUM * (col - 1); // global index

    Real temp_old = temp_red[ind_red];

    Real res = b[ind]
            + aW[ind] * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
            + aE[ind] * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
            + aS[ind] * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
            + aN[ind] * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)];

    Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);

    temp_red[ind_red] = temp_new;
    res = temp_new - temp_old;

    norm_L2[ind_red] = res * res;

} // end red_kernel

// Function to update temperature for black cells
void black_kernel (const Real *__restrict__ aP,
                   const Real *__restrict__ aW,
                   const Real *__restrict__ aE,
                   const Real *__restrict__ aS,
                   const Real *__restrict__ aN,
                   const Real *__restrict__ b,
                   const Real *__restrict__ temp_red,
                         Real *__restrict__ temp_black,
                         Real *__restrict__ norm_L2)
{
    int row = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int col = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (row > NUM / 2 || col > NUM) return;

    int ind_black = col * ((NUM >> 1) + 2) + row; // local (black) index
    int ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1); // global index

    Real temp_old = temp_black[ind_black];

    Real res = b[ind]
            + aW[ind] * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
            + aE[ind] * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
            + aS[ind] * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
            + aN[ind] * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)];

    Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);

    temp_black[ind_black] = temp_new;
    res = temp_new - temp_old;

    norm_L2[ind_black] = res * res;
} // end black_kernel

int main()
{
    // Declare host and device memory
    Real *aP_h, *aW_h, *aE_h, *aS_h, *aN_h, *b_h, *temp_black_h, *temp_red_h, *norm_L2_h;
    Real *aP_d, *aW_d, *aE_d, *aS_d, *aN_d, *b_d, *temp_black_d, *temp_red_d, *norm_L2_d;

    // Allocate memory
    cudaMalloc((void **) &aP_d, NUM * NUM * sizeof(Real));
    cudaMalloc((void **) &aW_d, NUM * NUM * sizeof(Real));
    cudaMalloc((void **) &aE_d, NUM * NUM * sizeof(Real));
    cudaMalloc((void **) &aS_d, NUM * NUM * sizeof(Real));
    cudaMalloc((void **) &aN_d, NUM * NUM * sizeof(Real));
    cudaMalloc((void **) &b_d, NUM * NUM * sizeof(Real));
    cudaMalloc((void **) &temp_black_d, NUM * NUM * sizeof(Real));
    cudaMalloc((void **) &temp_red_d, NUM * NUM * sizeof(Real));
    cudaMalloc((void **) &norm_L2_d, NUM * NUM * sizeof(Real));

    cudaMallocHost((void **) &aP_h, NUM * NUM * sizeof(Real));
    cudaMallocHost((void **) &aW_h, NUM * NUM * sizeof(Real));
    cudaMallocHost((void **) &aE_h, NUM * NUM * sizeof(Real));
    cudaMallocHost((void **) &aS_h, NUM * NUM * sizeof(Real));
    cudaMallocHost((void **) &aN_h, NUM * NUM * sizeof(Real));
    cudaMallocHost((void **) &b_h, NUM * NUM * sizeof(Real));
    cudaMallocHost((void **) &temp_black_h, NUM * NUM * sizeof(Real));
    cudaMallocHost((void **) &temp_red_h, NUM * NUM * sizeof(Real));
    cudaMallocHost((void **) &norm_L2_h, NUM * NUM * sizeof(Real));

    // Initialize memory
    for (int i = 0; i < NUM; i++)
    {
        for (int j = 0; j < NUM; j++)
        {
            int ind = i * NUM + j;
            aP_h[ind] = 1.0f;
            aW_h[ind] = 0.15f;
            aE_h[ind] = 0.15f;
            aS_h[ind] = 0.15f;
            aN_h[ind] = 0.15f;
            b_h[ind] = 0.0f;
            temp_black_h[ind] = 0.0f;
            temp_red_h[ind] = 0.0f;
            norm_L2_h[ind] = 0.0f;
        }
    }

    // Copy memory to device
    cudaMemcpy(aP_d, aP_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(aW_d, aW_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(aE_d, aE_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(aS_d, aS_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(aN_d, aN_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(temp_black_d, temp_black_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(temp_red_d, temp_red_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(norm_L2_d, norm_L2_h, NUM * NUM * sizeof(Real), cudaMemcpyHostToDevice);

    // Perform SOR operation
    int loops = 100;
    while (loops--)
    {
        int threads_per_block = BLOCK_SIZE * BLOCK_SIZE;
        int blocks_per_grid_red = (NUM / 2 - 1) / threads_per_block + 1;
        int blocks_per_grid_black = NUM / threads_per_block + 1;

        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_size_red(blocks_per_grid_red, blocks_per_grid_red);
        dim3 grid_size_black(blocks_per_grid_black, blocks_per_grid_black);

        // Perform GPU computation
        omp_set_dynamic(0);
        omp_set_num_threads(NUM);
#pragma omp parallel for schedule(static)
        for (int block = 0; block < NUM; block++)
        {
            int row = 1 + block * block_size.x + threadIdx.x;
            int col = 1 + block * block_size.y + threadIdx.y;
            if (row > NUM / 2 || col > NUM) continue;

            int ind_block = block * block_size.x * block_size.y + threadIdx.y * block_size.x + threadIdx.x;

            // Calculate local (red) index
            int ind_red = col * ((NUM >> 1) + 2) + row;
            // Calculate local (black) index
            int ind_black = col * ((NUM >> 1) + 2) + row;

            // Calculate global index
            int ind = 2 * row - (col & 1) - 1 + NUM * (col - 1);

            Real temp_old = temp_red_d[ind_red];

            Real res = b_d[ind] + aW_d[ind] * temp_black_d[row + (col - 1) * ((NUM >> 1) + 2)] +
                aE_d[ind] * temp_black_d[row + (col + 1) * ((NUM >> 1) + 2)] +
                aS_d[ind] * temp_black_d[row - (col & 1) + col * ((NUM >> 1) + 2)] +
                aN_d[ind] * temp_black_d[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)];

            Real temp_new = temp_old * (ONE - omega) + omega * (res / aP_d[ind]);

            temp_red_d[ind_red] = temp_new;
            res = temp_new - temp_old;

            norm_L2_d[ind_red] = res * res;

            // Perform GPU computation
            omp_set_dynamic(0);
            omp_set_num_threads(NUM);
#pragma omp parallel for schedule(static)
            for (int block = 0; block < NUM; block++)
            {
                int row = 1 + block * block_size.x + threadIdx.x;
                int col = 1 + block * block_size.y + threadIdx.y;
                if (row > NUM / 2 || col > NUM) continue;

                int ind_block = block * block_size.x * block_size.y + threadIdx.y * block_size.x + threadIdx.x;

                // Calculate local (red) index
                int ind_red = col * ((NUM >> 1) + 2) + row;
                // Calculate local (black) index
                int ind_black = col * ((NUM >> 1) + 2) + row;

                // Calculate global index
                int ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1);

                Real temp_old = temp_black_d[ind_black];

                Real res = b_d[ind] + aW_d[ind] * temp_red_d[row + (col - 1) * ((NUM >> 1) + 2)] +
                    aE_d[ind] * temp_red_d[row + (col + 1) * ((NUM >> 1) + 2)] +
                    aS_d[ind] * temp_red_d[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)] +
                    aN_d[ind] * temp_red_d[row + (col & 1) + col * ((NUM >> 1) + 2)];

                Real temp_new = temp_old * (ONE - omega) + omega * (res / aP_d[ind]);

                temp_black_d[ind_black] = temp_new;
                res = temp_new - temp_old;

                norm_L2_d[ind_black] = res * res;
            }
        }
    }

    // Store result in separate arrays
    cudaMemcpy(temp_black_h, temp_black_d, NUM * NUM * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_red_h, temp_red_d, NUM * NUM * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(norm_L2_h, norm_L2_d, NUM * NUM * sizeof(Real), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < NUM; i++)
    {
        for (int j = 0; j < NUM; j++)
        {
            int ind = i * NUM + j;
            printf("temp_red[%d]: %f, temp_black[%d]: %f, norm_L2: %f\n", ind, temp_red_h[ind], ind, temp_black_h[ind], norm_L2_h[ind]);
        }
    }

    // Free memory
    cudaFree(aP_d);
    cudaFree(aW_d);
    cudaFree(aE_d);
    cudaFree(aS_d);
    cudaFree(aN_d);
    cudaFree(b_d);
    cudaFree(temp_black_d);
    cudaFree(temp_red_d);
    cudaFree(norm_L2_d);

    cudaFreeHost(aP_h);
    cudaFreeHost(aW_h);
    cudaFreeHost(aE_h);
    cudaFreeHost(aS_h);
    cudaFreeHost(aN_h);
    cudaFreeHost(b_h);
    cudaFreeHost(temp_black_h);
    cudaFreeHost(temp_red_h);
    cudaFreeHost(norm_L2_h);

    return 0;
}