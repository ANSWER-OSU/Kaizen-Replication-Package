// Convert the CUDA kernel calls in the original code to OpenMP

// block size
const int BLOCK_SIZE = 128;

void red_kernel(const float* aP, const float* aW, const float* aE, const float* aS, const float* aN, const float* b, const float* temp_black, float* temp_red, float* norm_L2) {
    #pragma omp parallel for collapse(2)
    for (int row = 1; row <= NUM / 2; row++) {
        for (int col = 1; col <= NUM; col++) {
            int ind_red = col * ((NUM >> 1) + 2) + row;  // local (red) index
            int ind = 2 * row - (col & 1) - 1 + NUM * (col - 1);  // global index

            float temp_old = temp_red[ind_red];

            float res = b[ind] + aW[ind] * temp_black[row + (col - 1) * ((NUM >> 1) + 2)] + aE[ind] * temp_black[row + (col + 1) * ((NUM >> 1) + 2)] + aS[ind] * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)] + aN[ind] * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)];

            float temp_new = temp_old * (1.0f - omega) + omega * (res / aP[ind]);

            temp_red[ind_red] = temp_new;
            res = temp_new - temp_old;

            norm_L2[ind_red] = res * res;
        }
    }
}

void black_kernel(const float* aP, const float* aW, const float* aE, const float* aS, const float* aN, const float* b, const float* temp_red, float* temp_black, float* norm_L2) {
    #pragma omp parallel for collapse(2)
    for (int row = 1; row <= NUM / 2; row++) {
        for (int col = 1; col <= NUM; col++) {
            int ind_black = col * ((NUM >> 1) + 2) + row;  // local (black) index
            int ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1);  // global index

            float temp_old = temp_black[ind_black];

            float res = b[ind] + aW[ind] * temp_red[row + (col - 1) * ((NUM >> 1) + 2)] + aE[ind] * temp_red[row + (col + 1) * ((NUM >> 1) + 2)] + aS[ind] * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)] + aN[ind] * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)];

            float temp_new = temp_old * (1.0f - omega) + omega * (res / aP[ind]);

            temp_black[ind_black] = temp_new;
            res = temp_new - temp_old;

            norm_L2[ind_black] = res * res;
        }
    }
}