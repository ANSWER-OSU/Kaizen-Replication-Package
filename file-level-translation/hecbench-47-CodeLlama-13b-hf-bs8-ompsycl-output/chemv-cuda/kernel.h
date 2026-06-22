void chemv_kernel0(struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y, float alpha_im, float alpha_re, float beta_im, float beta_re) {
    int t0 = threadIdx.x;
    int b0 = blockIdx.x;
    struct ComplexFloat private_var96;
    struct ComplexFloat private_var97;
    struct ComplexFloat private_var98;
    private_var96.Re = 0.0f;
    private_var96.Im = 0.0f;
    private_var97.Re = 0.0f;
    private_var97.Im = 0.0f;
    private_var98.Re = 0.0f;
    private_var98.Im = 0.0f;
    for (int c1 = 5888 * b0; c1 <= ppcg_min(67712, 5856 * b0 + 6016); c1 += 32) {
        int c3 = c1 + t0;
        struct ComplexFloat private_var99;
        struct ComplexFloat private_var100;
        private_var99.Re = 0.0f;
        private_var99.Im = 0.0f;
        private_var100.Re = 0.0f;
        private_var100.Im = 0.0f;
        if (c3 < 5856 * b0 + 6016) {
            private_var99.Re = AT[5 * c3 + 4].Re * X[-c3].Im - AT[5 * c3 + 3].Im * X[-c3].Re;
            private_var99.Im = AT[5 * c3 + 4].Re * X[-c3].Re + AT[5 * c3 + 3].Im * X[-c3].Im;
            private_var100.Re = AT[5 * c3 + 2].Re * X[-c3].Im - AT[5 * c3 + 1].Im * X[-c3].Re;
            private_var100.Im = AT[5 * c3 + 2].Re * X[-c3].Re + AT[5 * c3 + 1].Im * X[-c3].Im;
        }
        for (int c5 = ppcg_max(0, 5888 * b0 + t0 - c1); c5 <= ppcg_min(31, 5856 * b0 + 6016 + t0 - c1); c5 += 1) {
            int c6 = c3 + c5;
            private_var96.Re += AT[5 * c6 + 4].Re * X[-c6].Im - AT[5 * c6 + 3].Im * X[-c6].Re;
            private_var96.Im += AT[5 * c6 + 4].Re * X[-c6].Re + AT[5 * c6 + 3].Im * X[-c6].Im;
            private_var97.Re += AT[5 * c6 + 2].Re * X[-c6].Im - AT[5 * c6 + 1].Im * X[-c6].Re;
            private_var97.Im += AT[5 * c6 + 2].Re * X[-c6].Re + AT[5 * c6 + 1].Im * X[-c6].Im;
        }
        private_var96.Re += private_var99.Re;
        private_var96.Im += private_var99.Im;
        private_var97.Re += private_var100.Re;
        private_var97.Im += private_var100.Im;
        if (c3 < 5856 * b0 + 6016) {
            Y[c3].Re = private_var96.Re * alpha_re - private_var96.Im * alpha_im + X[c3].Re * beta_re - X[c3].Im * beta_im;
            Y[c3].Im = private_var96.Re * alpha_im + private_var96.Im * alpha_re + X[c3].Re * beta_im + X[c3].Im * beta_re;
            private_var98.Re += private_var97.Re;
            private_var98.Im += private_var97.Im;
        }
        for (int c5 = ppcg_max(0, 5888 * b0 + t0 - c1 + 1); c5 <= ppcg_min(31, 5856 * b0 + 6016 + t0 - c1 + 1); c5 += 1) {
            int c6 = c3 + c5;
            Y[c6].Re = private_var96.Re * AT[5 * c6 + 4].Im - private_var96.Im * AT[5 * c6 + 3].Re - X[c6].Re * beta_im + X[c6].Im * beta_re;
            Y[c6].Im = private_var96.Im * AT[5 * c6 + 4].Im + private_var96.Re * AT[5 * c6 + 3].Re - X[c6].Im * beta_im - X[c6].Re * beta_re;
        }
    }
    if (t0 < 5856 * b0 + 6016) {
        Y[t0].Re = private_var98.Re * alpha_re - private_var98.Im * alpha_im + X[t0].Re * beta_re - X[t0].Im * beta_im;
        Y[t0].Im = private_var98.Re * alpha_im + private_var98.Im * alpha_re + X[t0].Re * beta_im + X[t0].Im * beta_re;
    }
}
void chemv_kernel1(struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y, float alpha_im, float alpha_re) {
    int t0 = threadIdx.x;
    int b0 = blockIdx.x;
    struct ComplexFloat private_var94;
    struct ComplexFloat private_var95;
    private_var94.Re = 0.0f;
    private_var94.Im = 0.0f;
    private_var95.Re = 0.0f;
    private_var95.Im = 0.0f;
    for (int c1 = 5888 * b0; c1 <= ppcg_min(67712, 5856 * b0 + 6016); c1 += 32) {
        int c3 = c1 + t0;
        struct ComplexFloat private_var96;
        struct ComplexFloat private_var97;
        private_var96.Re = 0.0f;
        private_var96.Im = 0.0f;
        private_var97.Re = 0.0f;
        private_var97.Im = 0.0f;
        if (c3 < 5856 * b0 + 6016) {
            private_var96.Re = AT[5 * c3 + 4].Re * X[-c3].Im - AT[5 * c3 + 3].Im * X[-c3].Re;
            private_var96.Im = AT[5 * c3 + 4].Re * X[-c3].Re + AT[5 * c3 + 3].Im * X[-c3].Im;
            private_var97.Re = AT[5 * c3 + 2].Re * X[-c3].Im - AT[5 * c3 + 1].Im * X[-c3].Re;
            private_var97.Im = AT[5 * c3 + 2].Re * X[-c3].Re + AT[5 * c3 + 1].Im * X[-c3].Im;
        }
        for (int c5 = ppcg_max(0, 5888 * b0 + t0 - c1); c5 <= ppcg_min(31, 5856 * b0 + 6016 + t0 - c1); c5 += 1) {
            int c6 = c3 + c5;
            private_var94.Re += AT[5 * c6 + 4].Re * X[-c6].Im - AT[5 * c6 + 3].Im * X[-c6].Re;
            private_var94.Im += AT[5 * c6 + 4].Re * X[-c6].Re + AT[5 * c6 + 3].Im * X[-c6].Im;
            private_var95.Re += AT[5 * c6 + 2].Re * X[-c6].Im - AT[5 * c6 + 1].Im * X[-c6].Re;
            private_var95.Im += AT[5 * c6 + 2].Re * X[-c6].Re + AT[5 * c6 + 1].Im * X[-c6].Im;
        }
        private_var94.Re += private_var96.Re;
        private_var94.Im += private_var96.Im;
        private_var95.Re += private_var97.Re;
        private_var95.Im += private_var97.Im;
        if (c3 < 5856 * b0 + 6016) {
            Y[c3].Re = private_var94.Re * alpha_re - private_var94.Im * alpha_im + X[c3].Re;
            Y[c3].Im = private_var94.Re * alpha_im + private_var94.Im * alpha_re + X[c3].Im;
            private_var95.Re += X[c3].Re;
            private_var95.Im += X[c3].Im;
        }
        for (int c5 = ppcg_max(0, 5888 * b0 + t0 - c1 + 1); c5 <= ppcg_min(31, 5856 * b0 + 6016 + t0 - c1 + 1); c5 += 1) {
            int c6 = c3 + c5;
            Y[c6].Re = private_var94.Re * AT[5 * c6 + 4].Im - private_var94.Im * AT[5 * c6 + 3].Re - X[c6].Im;
            Y[c6].Im = private_var94.Im * AT[5 * c6 + 4].Im + private_var94.Re * AT[5 * c6 + 3].Re + X[c6].Im;
        }
    }
    if (t0 < 5856 * b0 + 6016) {
        Y[t0].Re = private_var95.Re * alpha_re - private_var95.Im * alpha_im + X[t0].Re;
        Y[t0].Im = private_var95.Re * alpha_im + private_var95.Im * alpha_re + X[t0].Im;
    }
}
template <typename T>
void chemv(const char* uplo, int n, T alpha, T* AT, int AT_start, int AT_end, T* X, int X_start, int X_end, T beta, T* Y, int Y_start, int Y_end) {
    int nThreads = 0;
    int nBlocks = 0;
    int shared_size = 0;
    cudaError_t err;
    err = cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, chemv_kernel0, shared_size);
    if (err != cudaSuccess) {
        printf("Error calculating density of chemv_kernel0 occuancy (error code %s)!\n", cudaGetErrorString(err));
        exit(1);
    }
    err = chemv_kernel0<<<nBlocks, nThreads>>>(AT, X, Y, alpha.im, alpha.re, beta.im, beta.re);
    if (err != cudaSuccess) {
        printf("Error executing chemv_kernel0 (error code %s)!\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, chemv_kernel1, shared_size);
    if (err != cudaSuccess) {
        printf("Error calculating density of chemv_kernel1 occuancy (error code %s)!\n", cudaGetErrorString(err));
        exit(1);
    }
    err = chemv_kernel1<<<nBlocks, nThreads>>>(AT, X, Y, alpha.im, alpha.re);
    if (err != cudaSuccess) {
        printf("Error executing chemv_kernel1 (error code %s)!\n", cudaGetErrorString(err));
        exit(1);
    }
}