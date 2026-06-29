#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>

const int BLOCK_SIZE = 32;
const float R_min = 1.f, R_max = 1.f;
const float a1 = 1.f, a2 = 1.f, a3 = 1.f, a4 = 1.f, a5 = 1.f;
const float EPS = 1e-5f;

__global__ void AIDW_Kernel(
    float *dx, float *dy, float *dz, int dnum, float *ix, float *iy, float *iz, int inum, float area, float *avg_dist)
{
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= inum) {
        return;
    }

    float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 1.f;

    float r_obs = avg_dist[tid];                // The observed average nearest neighbor distance
    float r_exp = 0.5f / sqrtf(dnum / area);    // The expected nearest neighbor distance for a random pattern
    float R_S0 = r_obs / r_exp;                 // The nearest neighbor statistic

    // Normalize the R(S0) measure such that it is bounded by 0 and 1 by a fuzzy membership function 
    float u_R = 0.f;
    if(R_S0 >= R_min) u_R = 0.5f-0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
    if(R_S0 >= R_max) u_R = 1.f;

    // Determine the appropriate distance-decay parameter alpha by a triangular membership function
    // Adaptive power parameter: a (alpha)
    if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
    if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
    if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
    if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
    if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
    if(u_R>0.9f && u_R<=1.f)  alpha = a5;
    alpha *= 0.5f; // Half of the power

    // Weighted average
    for(int j = 0; j < dnum; j++) {
        dist = (ix[tid] - dx[j]) * (ix[tid] - dx[j]) + (iy[tid] - dy[j]) * (iy[tid] - dy[j]) ;
        t = 1.f / powf(dist, alpha);  sum += t;  z += dz[j] * t;
    }
    iz[tid] = z / sum;
}

__global__ void AIDW_Kernel_Tiled(
    float *dx, float *dy, float *dz, int dnum, float *ix, float *iy, 
    float *iz, int inum, float area, float *avg_dist)
{
    // Shared Memory
    __shared__ float sdx[BLOCK_SIZE];
    __shared__ float sdy[BLOCK_SIZE];
    __shared__ float sdz[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE; 
    if (tid >= inum) {
        return;
    }

    float dist = 0.f, t = 0.f, alpha = 0.f;

    int part = (dnum - 1) / BLOCK_SIZE;
    int m, e;

    float sum_up = 0.f;
    float sum_dn = 0.f;   
    float six_s, siy_s;

    float r_obs = avg_dist[tid];               // The observed average nearest neighbor distance
    float r_exp = 0.5f / sqrtf(dnum / area);   // The expected nearest neighbor distance for a random pattern
    float R_S0 = r_obs / r_exp;                 // The nearest neighbor statistic

    float u_R = 0.f;
    if(R_S0 >= R_min) u_R = 0.5f-0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
    if(R_S0 >= R_max) u_R = 1.f;

    // Determine the appropriate distance-decay parameter alpha by a triangular membership function
    // Adaptive power parameter: a (alpha)
    if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
    if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
    if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
    if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
    if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
    if(u_R>0.9f && u_R<=1.f)  alpha = a5;
    alpha *= 0.5f; // Half of the power

    float six_t = ix[tid];
    float siy_t = iy[tid];
    int lid = threadIdx.x;
    for(m = 0; m <= part; m++) {  // Weighted Sum  
        int num_threads = min(BLOCK_SIZE, dnum - BLOCK_SIZE *m);
        if (lid < num_threads) {
            sdx[lid] = dx[lid + BLOCK_SIZE * m];
            sdy[lid] = dy[lid + BLOCK_SIZE * m];
            sdz[lid] = dz[lid + BLOCK_SIZE * m];
        }
        __syncthreads();

        for(e = 0; e < BLOCK_SIZE; e++) {
            six_s = six_t - sdx[e];
            siy_s = siy_t - sdy[e];
            dist = (six_s * six_s + siy_s * siy_s);
            t = 1.f / powf(dist, alpha);  sum_dn += t;  sum_up += t * sdz[e];
        }
        __syncthreads();
    }
    iz[tid] = sum_up / sum_dn;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("%s: <pts> <iterations>\n", argv[0]);
        return 1;
    }

    const int numk = atoi(argv[1]); // number of points (unit: 1K)
    const int iterations = atoi(argv[2]); // repeat kernel execution

    const int dnum = numk * 1024;
    const int inum = dnum;
    const size_t dnum_size = dnum * sizeof(float);
    const size_t inum_size = inum * sizeof(float);

    // Area of planar region
    const float width = 2000, height = 2000;
    const float area = width * height;

    std::vector<float> dx(dnum), dy(dnum), dz(dnum);
    std::vector<float> avg_dist(dnum);
    std::vector<float> ix(inum), iy(inum), iz(inum);

    srand(123);
    for(int i = 0; i < dnum; i++)
    {
        dx[i] = rand()/(float)RAND_MAX * 1000;
        dy[i] = rand()/(float)RAND_MAX * 1000;
        dz[i] = rand()/(float)RAND_MAX * 1000;
    }

    for(int i = 0; i < inum; i++)
    {
        ix[i] = rand()/(float)RAND_MAX * 1000;
        iy[i] = rand()/(float)RAND_MAX * 1000;
        iz[i] = 0.f;
    }

    for(int i = 0; i < dnum; i++)
    {
        avg_dist[i] = rand()/(float)RAND_MAX * 3;
    }

    printf("Size = : %d K \n", numk);
    printf("dnum = : %d\ninum = : %d\n", dnum, inum);

    float *d_dx, *d_dy, *d_dz;
    float *d_ix, *d_iy, *d_iz;
    float *d_avg_dist;

    cudaMalloc((void**)&d_dx, dnum_size); 
    cudaMalloc((void**)&d_dy, dnum_size); 
    cudaMalloc((void**)&d_dz, dnum_size); 
    cudaMalloc((void**)&d_ix, inum_size); 
    cudaMalloc((void**)&d_iy, inum_size); 
    cudaMalloc((void**)&d_iz, inum_size); 
    cudaMalloc((void**)&d_avg_dist, dnum_size); 

    cudaMemcpy(d_dx, dx.data(), dnum_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_dy, dy.data(), dnum_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_dz, dz.data(), dnum_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_ix, ix.data(), inum_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_iy, iy.data(), inum_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_avg_dist, avg_dist.data(), dnum_size, cudaMemcpyHostToDevice); 

    dim3 threadsPerBlock (BLOCK_SIZE);
    dim3 blocksPerGrid ((inum + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start = std::chrono::steady_clock::now();
    if (iterations > 1) {
        for (int i = 0; i < iterations; i++)
            AIDW_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_dx, d_dy, d_dz, dnum, d_ix, d_iy, d_iz, inum, area, d_avg_dist);
    } else {
        AIDW_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_dx, d_dy, d_dz, dnum, d_ix, d_iy, d_iz, inum, area, d_avg_dist);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of AIDW_Kernel: %f (s)\n", (time * 1e-9f) / iterations);

    cudaMemcpy(iz.data(), d_iz, inum_size, cudaMemcpyDeviceToHost); 

    AIDW_Kernel_Tiled<<<blocksPerGrid, threadsPerBlock>>>(
        d_dx, d_dy, d_dz, dnum, d_ix, d_iy, d_iz, inum, area, d_avg_dist);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of AIDW_Kernel_Tiled: %f (s)\n", (time * 1e-9f) / iterations);

    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_dz);
    cudaFree(d_ix);
    cudaFree(d_iy);
    cudaFree(d_iz);
    cudaFree(d_avg_dist);
    return 0;
}