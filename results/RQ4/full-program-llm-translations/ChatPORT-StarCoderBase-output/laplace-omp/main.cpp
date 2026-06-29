#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <omp.h>
#include "timer.h"
#include "kernels.h"
#include "reference.h"


/** Function to evaluate coefficient matrix and right-hand side vector.
 *
 * \param[in]   rowmax   number of rows
 * \param[in]   colmax   number of columns
 * \param[in]   th_cond  thermal conductivity
 * \param[in]   dx       grid size in x dimension (uniform)
 * \param[in]   dy       grid size in y dimension (uniform)
 * \param[in]   width    width of plate (z dimension)
 * \param[in]   TN       temperature at top boundary
 * \param[out]  aP       array of self coefficients
 * \param[out]  aW       array of west neighbor coefficients
 * \param[out]  aE       array of east neighbor coefficients
 * \param[out]  aS       array of south neighbor coefficients
 * \param[out]  aN       array of north neighbor coefficients
 * \param[out]  b        right-hand side array
 */
void fill_coeffs (int rowmax, int colmax, Real th_cond, Real dx, Real dy,
    Real width, Real TN, Real * aP, Real * aW, Real * aE,
    Real * aS, Real * aN, Real * b)
{
  #pragma omp target teams distribute parallel for collapse(2) num_teams(colmax) num_threads(BLOCK_SIZE)
  for (int col = 0; col < colmax; ++col) {
    for (int row = 0; row < rowmax; ++row) {
      int ind = col * rowmax + row;

      b[ind] = 0.0;
      Real SP = 0.0;

      if (col == 0) {
        // left BC: temp = 0
        aW[ind] = 0.0;
        SP = -2.0 * th_cond * width * dy / dx;
      } else {
        aW[ind] = th_cond * width * dy / dx;
      }

      if (col == colmax - 1) {
        // right BC: temp = 0
        aE[ind] = 0.0;
        SP = -2.0 * th_cond * width * dy / dx;
      } else {
        aE[ind] = th_cond * width * dy / dx;
      }

      if (row == 0) {
        // bottom BC: temp = 0
        aS[ind] = 0.0;
        SP = -2.0 * th_cond * width * dx / dy;
      } else {
        aS[ind] = th_cond * width * dx / dy;
      }

      if (row == rowmax - 1) {
        // top BC: temp = TN
        aN[ind] = 0.0;
        b[ind] = 2.0 * th_cond * width * dx * TN / dy;
        SP = -2.0 * th_cond * width * dx / dy;
      } else {
        aN[ind] = th_cond * width * dx / dy;
      }

      aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
    } // end for row
  } // end for col
} // end fill_coeffs


/** Main function that solves Laplace's equation in 2D (heat conduction in plate)
 *
 * Contains iteration loop for red-black Gauss-Seidel with SOR GPU kernels
 */
int main (void) {

  // size of plate
  Real L = 1.0;
  Real H = 1.0;
  Real width = 0.01;

  // thermal conductivity
  Real th_cond = 1.0;

  // temperature at top boundary
  Real TN = 1.0;

  // SOR iteration tolerance
  Real tol = 1.e-6;

  // number of cells in x and y directions
  // including unused boundary cells
  int num_rows = (NUM / 2) + 2;
  int num_cols = NUM + 2;
  int size_temp = num_rows * num_cols;
  int size = NUM * NUM;

  // size of cells
  Real dx = L / NUM;
  Real dy = H / NUM;

  // iterations for Red-Black Gauss-Seidel with SOR
  int iter;
  int it_max = 1e6;

  // allocate memory
  Real *aP, *aW, *aE, *aS, *aN, *b;
  Real *temp_red, *temp_black;
  Real *temp_red_ref, *temp_black_ref;

  // arrays of coefficients
  aP = (Real *) calloc (size, sizeof(Real));
  aW = (Real *) calloc (size, sizeof(Real));
  aE = (Real *) calloc (size, sizeof(Real));
  aS = (Real *) calloc (size, sizeof(Real));
  aN = (Real *) calloc (size, sizeof(Real));

  // RHS
  b = (Real *) calloc (size, sizeof(Real));

  // temperature arrays
  temp_red = (Real *) calloc (size_temp, sizeof(Real));
  temp_black = (Real *) calloc (size_temp, sizeof(Real));
  temp_red_ref = (Real *) calloc (size_temp, sizeof(Real));
  temp_black_ref = (Real *) calloc (size_temp, sizeof(Real));

  // set coefficients
  fill_coeffs (NUM, NUM, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);

  #pragma omp target teams distribute parallel for num_teams(size_temp) num_threads(BLOCK_SIZE)
  for (int i = 0; i < size_temp; ++i) {
    temp_red[i] = 0.0;
    temp_black[i] = 0.0;
  }

  // block and grid dimensions
  dim3 dimBlock (BLOCK_SIZE, 2);
  dim3 dimGrid (NUM / (2 * BLOCK_SIZE), NUM/2);

  // print problem info
  printf("Problem size: %d x %d \n", NUM, NUM);

  // allocate device memory
  Real *aP_d, *aW_d, *aE_d, *aS_d, *aN_d, *b_d;
  Real *temp_red_d;
  Real *temp_black_d;
  Real *bl_norm_L2_d;
  Real *norm_L2_d;

  // residual
  // one for each temperature value
  int size_norm = size_temp;
  cudaMalloc ((void**) &bl_norm_L2_d, size_norm * sizeof(Real));
  cudaMemset (bl_norm_L2_d, 0, size_norm * sizeof(Real));

  // total residual
  cudaMalloc ((void**) &norm_L2_d, sizeof(Real));

  cudaMalloc ((void**) &aP_d, size * sizeof(Real));
  cudaMalloc ((void**) &aW_d, size * sizeof(Real));
  cudaMalloc ((void**) &aE_d, size * sizeof(Real));
  cudaMalloc ((void**) &aS_d, size * sizeof(Real));
  cudaMalloc ((void**) &aN_d, size * sizeof(Real));
  cudaMalloc ((void**) &b_d, size * sizeof(Real));
  cudaMalloc ((void**) &temp_red_d, size_temp * sizeof(Real));
  cudaMalloc ((void**) &temp_black_d, size_temp * sizeof(Real));

  // copy to device memory
  cudaMemcpy (aP_d, aP, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (aW_d, aW, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (aE_d, aE, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (aS_d, aS, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (aN_d, aN, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (b_d, b, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (temp_red_d, temp_red, size_temp * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (temp_black_d, temp_black, size_temp * sizeof(Real), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  StartTimer();

  // Determine temporary device storage requirements 
  void     *d_temp_storage = nullptr;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                         bl_norm_L2_d, norm_L2_d, size_norm);

  // Allocate temporary storage
  if (temp_storage_bytes != 0)
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // iteration loop
  for (iter = 1; iter <= it_max; ++iter) {

    Real norm_L2 = 0.0;

    #pragma omp target teams distribute parallel for collapse(2) num_teams(NUM/2) num_threads(BLOCK_SIZE)
    for (int col = 0; col < NUM; ++col) {
      for (int row = 0; row < 2; ++row) {
        int ind = col * rowmax + row;
        red_kernel (aP_d[ind], aW_d[ind], aE_d[ind], aS_d[ind], aN_d[ind], b_d[ind], temp_black_d[ind], temp_red_d[ind], bl_norm_L2_d);
      }
    }
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, bl_norm_L2_d, norm_L2_d, size_norm);
    #pragma omp target teams distribute parallel for num_teams(1) num_threads(BLOCK_SIZE)
    for (int i = 0; i < size_norm; ++i) {
      norm_L2 += norm_L2_d[i];
    }
    Real temp = norm_L2;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, bl_norm_L2_d, norm_L2_d, size_norm);
    for (int i = 0; i < size_norm; ++i) {
      norm_L2 += norm_L2_d[i];
    }

    // calculate residual
    norm_L2 = sqrt(norm_L2 / ((Real)size));

    if (iter % 1000 == 0) {
      printf("%5d, %f\n", iter, norm_L2);
    }

    // if tolerance has been reached, end SOR iterations
    if (norm_L2 < tol) {
      break;
    }
  }

  if (d_temp_storage != nullptr) {
    cudaFree(d_temp_storage);
  }

  double runtime = GetTimer();
  printf("Total time for %i iterations: %f s\n", iter, runtime / 1000.0);

  // transfer final temperature values back
  cudaMemcpy (temp_red, temp_red_d, size_temp * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy (temp_black, temp_black_d, size_temp * sizeof(Real), cudaMemcpyDeviceToHost);

  // Reference
  int count = 0;

  for (iter = 1; iter <= it_max; ++iter) {
    Real norm_L2;
    #pragma omp target teams distribute parallel for collapse(2) num_teams(NUM) num_threads(BLOCK_SIZE)
    for (int col = 0; col < NUM; ++col) {
      for (int row = 0; row < 2; ++row) {
        int ind = col * rowmax + row;
        red_ref (aP[ind], aW[ind], aE[ind], aS[ind], aN[ind], b[ind], temp_black_ref[ind], temp_red_ref[ind]);
      }
    }
    norm_L2 = black_ref (aP, aW, aE, aS, aN, b, temp_red_ref, temp_black_ref);
    norm_L2 += black_ref (aP, aW, aE, aS, aN, b, temp_red_ref, temp_black_ref);
    norm_L2 = sqrt(norm_L2 / ((Real)size));
    if (norm_L2 < tol) {
      break;
    }
  }

  // print temperature data to file
  FILE * pfile;
  pfile = fopen("temperature.dat", "w");

  if (pfile != NULL) {
    fprintf(pfile, "#x\ty\ttemp(K)\n");

    int row, col;
    for (row = 1; row < NUM + 1; ++row) {
      for (col = 1; col < NUM + 1; ++col) {
        Real x_pos = (col - 1) * dx + (dx / 2);
        Real y_pos = (row - 1) * dy + (dy / 2);

        if ((row + col) % 2 == 0) {
          // even, so red cell
          int ind = col * num_rows + (row + (col % 2)) / 2;
          if ((temp_red[ind] - temp_red_ref[ind]) >= 1e-3f) {
            count++;
          }
          fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_red[ind]);
        } else {
          // odd, so black cell
          int ind = col * num_rows + (row + ((col + 1) % 2)) / 2;
          if ((temp_black[ind] - temp_black_ref[ind]) >= 1e-3f) {
            count++;
          }
          fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_black[ind]);
        }
      }
      fprintf(pfile, "\n");
    }
  }

  fclose(pfile);
  if (count == 0) {
    printf("Pass\n");
  } else {
    printf("Fail\n");
  }

  cudaFree(aP_d);
  cudaFree(aW_d);
  cudaFree(aE_d);
  cudaFree(aS_d);
  cudaFree(aN_d);
  cudaFree(b_d);
  cudaFree(temp_red_d);
  cudaFree(temp_black_d);
  cudaFree(bl_norm_L2_d);
  cudaFree(norm_L2_d);

  free(aP);
  free(aW);
  free(aE);
  free(aS);
  free(aN);
  free(b);
  free(temp_red);
  free(temp_black);
  free(temp_red_ref);
  free(temp_black_ref);

  return 0;
}