#include <omp.h>

void attention_kernel1 (const float *__restrict__ key, const float *__restrict__ query, float *__restrict__ dot_product, float *__restrict__ exp_sum, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < n / d; i++) {
        #pragma omp parallel for
        for (int j = 0; j < d; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++)
                sum += key[k * d + j] * query[j];
            dot_product[i * d + j] = sum;
            #pragma omp atomic
            exp_sum[0] += expf(sum);
        }
    }
}

void attention_kernel2 (const float *__restrict__ exp_sum, const float *__restrict__ dot_product, float *__restrict__ score, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        score[i] = expf(dot_product[i]) / exp_sum[0];
    }
}

void attention_kernel3 (const float *__restrict__ score, const float *__restrict__ value, float *__restrict__ output, int n, int d) {
    #pragma omp parallel for
    for (int j = 0; j < d; j++) {
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum += score[i] * value[i * d + j];
        output[j] = sum;
    }
}

void attention_kernel1_blockReduce (const float *__restrict__ key, const float *__restrict__ query, float *__restrict__ dot_product, float *__restrict__ exp_sum, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < n / d; i++) {
        float sum = 0;
        #pragma omp parallel for
        for (int j = 0; j < d; j++) {
            sum += key[i * d + j] * query[j];
        }
        dot_product[i] = sum;
        #pragma omp atomic
        exp_sum[0] += expf(sum);
    }
}

void attention_kernel1_warpReduce (const float *__restrict__ key, const float *__restrict__ query, float *__restrict__ dot_product, float *__restrict__ exp_sum, int n, int d) {
    #pragma omp parallel for num_threads(d)
    for (int i = 0; i < n / d; i++) {
        float sum = 0;
        #pragma omp parallel for
        for (int j = 0; j < d; j++) {
            sum += key[i * d + j] * query[j];
        }
        dot_product[i] = sum;
        #pragma omp atomic
        exp_sum[0] += expf(sum);
    }
}

void attention_kernel2_blockReduce (const float *__restrict__ exp_sum, const float *__restrict__ dot_product, const float *__restrict__ value, float *__restrict__ output, int n, int d) {
    #pragma omp parallel for
    for (int j = 0; j < d; j++) {
        float sum = 0;
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            float score = expf(dot_product[i]) / exp_sum[0];
            sum += score * value[i * d + j];
        }
        output[j] = sum;
    }
}

void attention_kernel2_warpReduce (const float *__restrict__ exp_sum, const float *__restrict__ dot_product, const float *__restrict__ value, float *__restrict__ output, int n, int d) {
    #pragma omp parallel for num_threads(d)
    for (int j = 0; j < d; j++) {
        float sum = 0;
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            float score = expf(dot_product[i]) / exp_sum[0];
            sum += score * value[i * d + j];
        }
        output[j] = sum;
    }
}