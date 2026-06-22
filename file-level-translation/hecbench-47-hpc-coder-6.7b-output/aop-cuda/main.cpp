#pragma omp parallel num_threads(8)
{
    #pragma omp single
    {
        fprintf(stderr, "Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.\n");
    }

#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#ifdef WITH_FULL_W_MATRIX
#define R_W_MATRICES_SMEM_SLOTS 15
#else
#define R_W_MATRICES_SMEM_SLOTS 12
#endif

#include <iostream>

struct PayoffCall
{
    double m_K;
    PayoffCall(double K) : m_K(K) {}
    double operator()(double S) const { return fmax(S - m_K, 0.0); }
    int is_in_the_money(double S) const { return S > m_K; }
};

struct PayoffPut
{
    double m_K;
    PayoffPut(double K) : m_K(K) {}
    double operator()(double S) const { return fmax(m_K - S, 0.0); }
    int is_in_the_money(double S) const { return S < m_K; }
};

#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <complex>

using namespace std;

const double EPSILON = 1e-12;
const int MAX_NUM_THREADS = 8;

using namespace std;

template <typename T>
T rosenbrock(int N, T* TK) 
{
    T f = 0;
    for (int i = 0; i < N-1; ++i) {
        T t = 10 * abs(TK[i] * TK[i+1] - TK[i]) + 0.5 * (TK[i]*TK[i] + TK[i+1]*TK[i+1]);
        f += t*t;
    }
    return f;
}

template <typename T>
T rosenbrock_wrap_ompp_leg(int N, T* TK, omp_pTL* tl) 
{
#ifdef _OPENMP
    #pragma omp para para_g(N) num_threads(N) schedule(static)
#endif
    for (int tid = 0; tid < omp_get_num_threads(); ++tid) {
        TK[tid] = rosenbrock(N, TK[tid*N] + tl[tid]);
    }

    T res = 0;
    #ifndef _OPENMP
    for (int i = 0; i < omp_get_num_threads(); ++i) {
        res += TK[i];
    }
    #else
    #pragma omp para para_g(N) num_threads(N) reduc_i(+:res)
    for (int i = 0; i < omp_get_num_threads(); ++i) {
        res += TK[i];
    }
    #endif
    return res;
}

template <typename T>
vector<T> rosenbrock_ompp_leg(int N, T lb, T ub, int seed) 
{
    omp_pTL* tl = new omp_pTL[N];
    T init_value = (nbins_per_t - rocks->nth) * ((double)1 / BETA);
    TK[i] = [i*N] + (tl[i] - init_value) / (MAX_TK_GROWTH[i] - init_value);

    mt19937 gen(seed+1);
    uniform_real_distribution<T> dist(lb, ub);

    T* TK = new T[N];
    for (int i = 0; i < N; ++i) TK[i] = dist(gen);

    T result = rosenbrock_wrap_ompp_leg(N, TK, tl);

    delete [] TK;
    delete [] tl;

    return result;
}

template <typename T>
T rosenbrock_hipp_leg(int N, vector<T>& TK, vector<int>& HIPPM_Index) 
{
    mt19937 gen(1);
    uniform_real_distribution<T> dist(1e-4, 1.0);

    T init_value = (nbins_per_t - rocks->nth) * ((double)1 / BETA);
    TK[i] = [i*N] + (tl[i] - init_value) / (MAX_TK_GROWTH[i] - init_value);

    T* TK = new T[N];
    for (int i = 0; i < N; ++i) TK[i] = dist(gen);

    T* HIPP_TK = new T[N];
    T** HIPP_TK_ptr = new T*[N];

    for (int i = 0; i < N; ++i) {
        HIPP_TK[i] = TK[i];
        HIPP_TK_ptr[i] = &HIPP_TK[i];
    }

    T res = 0;

    #pragma hipp para num_threads(N) schedule(static)
    for (int tid = 0; tid < omp_get_num_threads(); ++tid) {
        TK[tid] = rosenbrock(N, TK[tid*N] + HIPP_TK_ptr[tid]);
    }

    #pragma hipp reduc(&res)
    for (int i = 0; i < N; ++i) {
        res += TK[i];
    }

    delete [] TK;
    delete [] HIPP_TK;
    delete [] HIPP_TK_ptr;

    return res;
}

template <typename T>
T rosenbrock_gpu_leg(int N, int BLOCK_SIZE, int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int BLOCK_SIZE_Z, int* dev_HIPPM_Index, T* dev_TK) 
{
    mt19937 gen(1);
    uniform_real_distribution<T> dist(1e-4, 1.0);

    T init_value = (nbins_per_t - rocks->nth) * ((double)1 / BETA);
    TK[i] = [i*N] + (tl[i] - init_value) / (MAX_TK_GROWTH[i] - init_value);

    T* TK = new T[N];
    for (int i = 0; i < N; ++i) TK[i] = dist(gen);

    T* dev_TK;
    hipMalloc(&dev_TK, N*sizeof(T));
    hipMemcpy(dev_TK, TK, N*sizeof(T), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(gpu_leg_kernel, BLOCK_SIZE, N, 0, 0, dev_TK, N, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z, dev_HIPPM_Index);
    
    T res = 0;

    for (int i = 0; i < N; ++i) {
        hipMemcpy(&TK[i], dev_TK+i, sizeof(T), hipMemcpyDeviceToHost);
        res += TK[i];
    }

    hipFree(dev_TK);
    delete [] TK;

    return res;
}

template <typename T>
T rosenbrock_hipp_thin_leg(int N, int BLOCK_SIZE, T* TK) 
{
    mt19937 gen(1);
    uniform_real_distribution<T> dist(1e-4, 1.0);

    T* TK = new T[N];
    for (int i = 0; i < N; ++i) TK[i] = dist(gen);

    T* dev_TK;
    hipMalloc(&dev_TK, N*sizeof(T));
    hipMemcpy(dev_TK, TK, N*sizeof(T), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(hipp_thin_leg_kernel, BLOCK_SIZE, N, 0, 0, dev_TK, N);

    T res = 0;

    for (int i = 0; i < N; ++i) {
        hipMemcpy(&TK[i], dev_TK+i, sizeof(T), hipMemcpyDeviceToHost);
        res += TK[i];
    }

    hipFree(dev_TK);
    delete [] TK;

    return res;
}

template <typename T>
T rosenbrock_hipp_thin_leg_seidel_red(int N, int BLOCK_SIZE, T* TK) 
{
    mt19937 gen(1);
    uniform_real_distribution<T> dist(1e-4, 1.0);

    T* TK = new T[N];
    for (int i = 0; i < N; ++i) TK[i] = dist(gen);

    T* dev_TK;
    hipMalloc(&dev_TK, N*sizeof(T));
    hipMemcpy(dev_TK, TK, N*sizeof(T), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(hipp_thin_leg_seidel_red_kernel, BLOCK_SIZE, N, 0, 0, dev_TK, N);

    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        std::cout << "Error: " << hipGetErrorString(err) << std::endl;
    }

    T* TK_final = new T[N];
    hipMemcpy(TK_final, dev_TK, N*sizeof(T), hipMemcpyDeviceToHost);

    TK[i] = TK_final[i];

    T res = 0;

    for (int i = 0; i < N; ++i) {
        res += TK[i];
    }

    hipFree(dev_TK);
    delete [] TK;
    delete [] TK_final;

    return res;
}

void test_openmp()
{
    int N = 1000; // number of elements in the array
    omp_pTL tl[N]; // storage for OpenMP Legendarium result
    #pragma omp para
    for (int i = 0; i < N; ++i) 
        tl[i].num_threads = omp_get_num_threads();

    TK[i] = +(tl[i] - init_value) / (MAX_TK_GROWTH[i] - init_value);
    mt19937 gen(seed+1);
    uniform_real_distribution<T> dist(lb, ub);

    T* TK = new T[N];
    for (int i = 0; i < N; ++i)
        TK[i] = dist(gen);

    T res = rosenbrock(N, TK, tl);    

    #pragma omp para num_threads(N) schedule(static)
    for (int tid = 0; tid < omp_get_num_threads(); ++tid)
        TK[tid] = TK[tid*N] + (tl[tid].num_threads - init_value) / (MAX_TK_GROWTH[tid] - init_value);

    T A = rosenbrock(N*omp_get_num_threads(), TK);
}

void test_openmp_parafernum()
{
    int N = 1000; // number of elements in the array
    int** no_empt_arr = new int*[N];
    int* tl = new int[N];

    #pragma omp para pard(N)
    for (int i = 0; i < N; ++i) 
        no_empt_arr[i] = new int[10];

    #pragma omp para pard(N)
    for (int i = 0; i < N; ++i) 
        tl[i] = omp_get_num_threads();

    T* TK = new T[N];
    std::mt19937 gen(1);
    std::uniform_real_distribution<T> dist(1e-4, 1.0);

    for (int i = 0; i < N; ++i) 
        TK[i] = dist(gen);

    for (int i = 0; i < omp_get_num_threads(); ++i) 
    {
        for (int j = 0; j < 10; ++j) 
            TK[i] += no_empt_arr[i][j];
        TK[i] = TK[i] / (tl[i] * 10);
    }

    for (int i = 0; i < N; ++i) 
        TK[i] *= 5;

    T res = rosenbrock(N, TK);    
}

void test_hipp()
{
    int N = 1000; // number of elements in the array
    int BLOCK_SIZE = 1024; // max block size for rosenbrock_gpu_leg
    int BLOCK_SIZE_X = BLOCK_SIZE; // optimized block size for rosenbrock_gpu_leg
    int BLOCK_SIZE_Y = BLOCK_SIZE; // optimized block size for rosenbrock_gpu_leg
    int BLOCK_SIZE_Z = BLOCK_SIZE; // optimized block size for rosenbrock_gpu_leg
    int* bindS = new int[N];
    for (int i = 0; i < N; ++i) 
        bindS[i] = i;

    int* dev_bindS;
    hipMalloc(&dev_bindS, N*sizeof(int));
    hipMemcpy(dev_bindS, bindS, N*sizeof(int), hipMemcpyHostToDevice);

    T* TK = new T[N];
    std::mt19937 gen(1);
    std::uniform_real_distribution<T> dist(1e-4, 1.0);
    for (int i = 0; i < N; ++i) 
        TK[i] = dist(gen);

    T* dev_TK;
    hipMalloc(&dev_TK, N*sizeof(T));
    hipMemcpy(dev_TK, TK, N*sizeof(T), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(gpu_leg_kernel, BLOCK_SIZE, N, 0, 0, dev_TK, N, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z, dev_bindS);

    T res = 0;
    T* TK_final = new T[N];
    hipMemcpy(TK_final, dev_TK, N*sizeof(T), hipMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        res += TK_final[i] * 5;

    hipFree(dev_TK);
    delete [] TK;
}

void test_openmp_hipp()
{
    int N = 1000; // number of elements in the array
    int BLOCK_SIZE = 1024; // max block size for rosenbrock_gpu_leg
    int BLOCK_SIZE_X = BLOCK_SIZE; // optimized block size for rosenbrock_gpu_leg
    int BLOCK_SIZE_Y = BLOCK_SIZE; // optimized block size for rosenbrock_gpu_leg
    int BLOCK_SIZE_Z = BLOCK_SIZE; // optimized block size for rosenbrock_gpu_leg
    int* dev_HIPPM_Index;
    hipMalloc(&dev_HIPPM_Index, N*sizeof(int));

    T* TK = new T[N];
    std::mt19937 gen(1);
    std::uniform_real_distribution<T> dist(1e-4, 1.0);
    for (int i = 0; i < N; ++i) 
        TK[i] = dist(gen);

    T* dev_TK;
    hipMalloc(&dev_TK, N*sizeof(T));
    hipMemcpy(dev_TK, TK, N*sizeof(T), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(hipp_spvconv_kernel, BLOCK_SIZE, N, 0, 0, dev_TK, N, dev_HIPPM_Index);

    T res = 0;
    T* TK_final = new T[N];
    hipMemcpy(TK_final, dev_TK, N*sizeof(T), hipMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        res += TK_final[i] * 5;

    hipFree(dev_TK);
    delete [] TK;
}

void test_lleg()
{
    int n = 1000000;
    __ocl_pTL_ull l;
    __ocl_pTL lp;

    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&lp, &l);
        __ocl_pTL_push(&lp, i);
        __ocl_pTL_purge(&lp);
    }
}

void test_tleg()
{
    int n = 1000000;
    __ocl_pTL_t;
    __ocl_pTL tp;

    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}
  
void test_tcp()
{
    int n = 10000;
    __ocl_pTL_tcp tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_oclpipe()
{
    int n = 10000;
    __ocl_pTL_ullq tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_thynlight()
{
    int n = 10000;
    __ocl_pTL_ullh thp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&thp, &t);
        __ocl_pTL_push(&thp, i);
        __ocl_pTL_purge(&thp);
    }
}

void test_spthr()
{
    int n = 10000000;
    __ocl_pTL_spth sp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&sp, &t);
        __ocl_pTL_push(&sp, i);
        __ocl_pTL_purge(&sp);
    }
}

void test_barrier()
{
    int n = 10000;
    __ocl_pTL_ull bp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&bp, &b);
        __ocl_pTL_push(&bp, i);
        __ocl_pTL_purge(&bp);
    }
}

void test_ifpr()
{
    int n = 10000000;
    __ocl_pTL_ull rp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&rp, &r);
        __ocl_pTL_push(&rp, i);
        __ocl_pTL_purge(&rp);
    }
}

void test_thrier()
{
    int n = 100000;
    __ocl_pTL_ullth tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

u()
{
    int n = 100000;
    __ocl_pTL_ullu tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_q()
{
    int n = 10000000;
    __ocl_pTL_ullq tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_t()
{
    int n = 100000;
    __ocl_pTL_ullt tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_nspace()
{
    int n = 10000000;
    __ocl_pTL_ulln tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_spor()
{
    int n = 10000000;
    __ocl_pTL_ullsp tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_ipr()
{
    int n = 10000000;
    __ocl_pTL_ullip tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_sr()
{
    int n = 10000;
    __ocl_pTL_ullr pr;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&pr, &t);
        __ocl_pTL_push(&pr, i);
        __ocl_pTL_purge(&pr);
    }
}

void test_stru()
{
    int n = 10000000;
    __ocl_pTL_ullstru tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_pr()
{
    int n = 10000000;
    __ocl_pTL_ullpr tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_thr()
{
    int n = 10000000;
    __ocl_pTL_ullthr tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_spu()
{
    int n = 10000000;
    __ocl_pTL_ullspu tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_in()
{
    int n = 100000;
    __ocl_pTL_ullin tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_tpu()
{
    int n = 10000000;
    __ocl_pTL_ulltpu tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_upr()
{
    int n = 10000000;
    __ocl_pTL_ullupr tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_tup()
{
    int n = 10000000;
    __ocl_pTL_ulltup tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_nu()
{
    int n = 10000000;
    __ocl_pTL_ullnu tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_sp()
{
    int n = 10000000;
    __ocl_pTL_ullsp tp;
  
    #pragma ocl para num_threads(n)
    for (int i = 0; i < n; i++)
    {
        __ocl_pTL_set(&tp, &t);
        __ocl_pTL_push(&tp, i);
        __ocl_pTL_purge(&tp);
    }
}

void test_u() { }

void test_spu() {}

void test() 
{
    __ocl_pTL ptl;
    __ocl_pTL_set(&ptl, &t)
    __ocl_pTL_push(&ptl, 1)
    __ocl_pTL_push(&ptl, 2)
    __ocl_pTL_push(&ptl, 3)
    __ocl_pTL_purge(&ptl);
    
    int a = __ocl_pTL_pop(&ptl);
    int b = __ocl_pTL_pop(&ptl);
    int c = __ocl_pTL_pop(&ptl);
}

void test_at() 
{
    int n = 10;
    __ocl_pTL_ull att;
    __ocl_pTL_set(&att, &t);
    __ocl_pTL_deque(&att, n);
    __ocl_pTL_enque(&att, n);
    __ocl_pTL_pop(&att, n);
    __ocl_pTL_shawed(&att, n);
    __ocl_pTL_purged(&att, n);
}

int main()
{
    test();
    test_at();
    test_atindis();
    inis();
}