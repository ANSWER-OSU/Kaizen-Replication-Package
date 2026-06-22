#include <omp.h>

void red_black_gauss_seidel(const float *aP, const float *aW, const float *aE, const float *aS, const float *aN, const float *b,
                             float *temp_red, float *temp_black, int num_rows, int num_cols) {
    int num_cells = num_rows * num_cols;

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // Red cells
                for (int i = 1; i < num_rows - 1; i++) {
                    for (int j = 1 + (i % 2); j < num_cols - 1; j += 2) {
                        int ind = i * num_cols + j;
                        temp_red[ind] = (aW[ind] * temp_red[ind - 1] + aE[ind] * temp_red[ind + 1] +
                                        aS[ind] * temp_red[ind - num_cols] + aN[ind] * temp_red[ind + num_cols] +
                                        b[ind]) / aP[ind];
                    }
                }
            }

            #pragma omp section
            {
                // Black cells
                for (int i = 1; i < num_rows - 1; i++) {
                    for (int j = 1 + ((i + 1) % 2); j < num_cols - 1; j += 2) {
                        int ind = i * num_cols + j;
                        temp_black[ind] = (aW[ind] * temp_black[ind - 1] + aE[ind] * temp_black[ind + 1] +
                                        aS[ind] * temp_black[ind - num_cols] + aN[ind] * temp_black[ind + num_cols] +
                                        b[ind]) / aP[ind];
                    }
                }
            }
        }
    }
}