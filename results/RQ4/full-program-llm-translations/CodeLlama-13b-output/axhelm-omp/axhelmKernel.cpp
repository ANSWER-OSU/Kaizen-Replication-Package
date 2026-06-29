#include <iostream>
#include <cstdlib>

// OpenMP 4.5 equivalent of CUDA __global__ and __shared__ directive
#define __global__ 
#define __shared__ 

// OpenMP 4.5 equivalent of CUDA syncthreads() function
#define syncthreads() __sync_synchronize()

// OpenMP 4.5 equivalent of CUDA DONT_REUSE_RESULT macro
#define DONT_REUSE_RESULT __temp__

// OpenMP 4.5 equivalent of CUDA inline function
#define inline __inline__

#define dfloat double

// Global variables
int nelements;
int offset;
dfloat *ggeo;
dfloat *D;
dfloat *lambda;
dfloat *Q;
dfloat *Aq;

// Function to calculate element stiffness matrix
void CalculateElementStiffnessMatrix(int e, dfloat *r_D, dfloat *r_U, dfloat *r_V, dfloat *r_W,
                                      dfloat *r_AU, dfloat *r_AV, dfloat *r_AW)
{
    for (int k = 0; k < 8; ++k)
    {
        const int id = e * 512 + k * 8 * 8 + j * 8 + i;
        const int gbase = e * 512 + k * 8 * 8 + j * 8 + i;
        dfloat r_G00 = ggeo[gbase + 0];
        dfloat r_G01 = ggeo[gbase + 1];
        dfloat r_G02 = ggeo[gbase + 2];
        dfloat r_G11 = ggeo[gbase + 3];
        dfloat r_G12 = ggeo[gbase + 4];
        dfloat r_G22 = ggeo[gbase + 5];
        dfloat r_GwJ = ggeo[gbase + 6];
        dfloat r_lam0 = lambda[id + 0];
        dfloat r_lam1 = lambda[id + 1];
        __sync_synchronize();
        for (int m = 0; m < 8; m++)
        {
            r_AU[k] += DONT_REUSE_RESULT(s_U[j*8+m]) * r_D[m*8+k];
            r_AV[k] += DONT_REUSE_RESULT(s_V[j*8+m]) * r_D[m*8+k];
            r_AW[k] += DONT_REUSE_RESULT(s_W[j*8+m]) * r_D[m*8+k];
            __sync_synchronize();
        }
        __sync_synchronize();
        for (int m = 0; m < 8; m++)
        {
            dfloat Dmk = s_D[k*8+m];
            dfloat Ut = DONT_REUSE_RESULT(s_U[j*8+m]) * r_U[m];
            dfloat Vt = DONT_REUSE_RESULT(s_V[j*8+m]) * r_V[m];
            dfloat Wt = DONT_REUSE_RESULT(s_W[j*8+m]) * r_W[m];
            r_AU[k] -= r_G00 * Dmk[0] * Ut;
            r_AU[k] -= r_G11 * Dmk[1] * Vt;
            r_AU[k] -= r_G22 * Dmk[2] * Wt;
            r_AV[k] -= r_G01 * Dmk[0] * Vt;
            r_AV[k] -= r_G11 * Dmk[1] * Ut;
            r_AV[k] -= r_G12 * Dmk[2] * Wt;
            r_AW[k] -= r_G02 * Dmk[0] * Wt;
            r_AW[k] -= r_G12 * Dmk[1] * Vt;
            r_AW[k] -= r_G22 * Dmk[2] * Ut;
        }
        __sync_synchronize();
    }
}

__global__ void axhelm_omp(const int Nelements,
                           const int offset,
                           const dfloat * __restrict__ ggeo,
                           const dfloat * __restrict__ D,
                           const dfloat * __restrict__ lambda,
                           const dfloat * __restrict__ Q,
                           dfloat * __restrict__ Aq)
{
    __shared__ dfloat s_D[64];
    __shared__ dfloat s_U[64];
    __shared__ dfloat s_V[64];
    __shared__ dfloat s_W[64];
    __shared__ dfloat s_AU[64];
    __shared__ dfloat s_AV[64];
    __shared__ dfloat s_AW[64];
    dfloat r_Ut, r_Vt, r_Wt;
    dfloat r_U[8], r_V[8], r_W[8];
    dfloat r_AU[8], r_AV[8], r_AW[8];
    dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
    dfloat r_lam0, r_lam1;

    int e = blockIdx.x;
    int j = threadIdx.y;
    int i = threadIdx.x;

    for (int k = 0; k < 8; k++)
    {
        s_D[j*8+i] = D[j*8+i];
        const int base = i + j * 8 + e * 512;
        for (int k = 0; k < 8; k++)
        {
            r_U[k] = Q[base + k * 8 * 8 + 0 * offset];
            r_V[k] = Q[base + k * 8 * 8 + 1 * offset];
            r_W[k] = Q[base + k * 8 * 8 + 2 * offset];
            r_AU[k] = 0;
            r_AV[k] = 0;
            r_AW[k] = 0;
        }

        r_Ut = 0;
        r_Vt = 0;
        r_Wt = 0;
        for (int m = 0; m < 8; m++)
        {
            dfloat Dkm = s_D[k*8+m];
            r_Ut += Dkm * r_U[m];
            r_Vt += Dkm * r_V[m];
            r_Wt += Dkm * r_W[m];
        }

        dfloat Ur = 0, Us = 0;
        dfloat Vr = 0, Vs = 0;
        dfloat Wr = 0, Ws = 0;
        for (int m = 0; m < 8; m++)
        {
            dfloat Dim = s_D[i*8+m];
            dfloat Djm = s_D[j*8+m];
            Ur += Dim * s_U[j*8+m];
            Us += Djm * s_U[m*8+i];
            Vr += Dim * s_V[j*8+m];
            Vs += Djm * s_V[m*8+i];
            Wr += Dim * s_W[j*8+m];
            Ws += Djm * s_W[m*8+i];
        }

        r_AU[k] += r_GwJ * r_lam1 * r_U[k];
        r_AV[k] += r_GwJ * r_lam1 * r_V[k];
        r_AW[k] += r_GwJ * r_lam1 * r_W[k];

        CalculateElementStiffnessMatrix(e, r_D, r_U, r_V, r_W,
                                         r_AU, r_AV, r_AW);

        for (int k = 0; k < 8; k++)
        {
            const int id = e * 512 + k * 8 * 8 + j * 8 + i;
            Aq[id + 0 * offset] = r_AU[k];
            Aq[id + 1 * offset] = r_AV[k];
            Aq[id + 2 * offset] = r_AW[k];
        }
    }

}

int main()
{
    // Initialize variables
    nelements = 8;
    offset = 8;
    // Set up CUDA array for ggeo and D
    ggeo = (dfloat*)malloc(nelements*offset*offset*sizeof(dfloat));
    D = (dfloat*)malloc(nelements*offset*offset*sizeof(dfloat));
    // Set up CUDA array for lambda and Q
    lambda = (dfloat*)malloc(nelements*offset*sizeof(dfloat));
    Q = (dfloat*)malloc(nelements*offset*sizeof(dfloat));
    // Set up CUDA array for Aq
    Aq = (dfloat*)malloc(nelements*offset*sizeof(dfloat));

    // Call the OpenMP version of the GPU kernel
    axhelm_omp(nelements, offset, ggeo, D, lambda, Q, Aq);
    return 0;
}