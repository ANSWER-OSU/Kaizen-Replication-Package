// OpenMP implementation of the AIDW algorithm

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "reference.h"

// Calculate the power parameter, and then weighted interpolating
// Without using shared memory
void AIDW(
    const float *dx, 
    const float *dy,
    const float *dz,
    const int dnum,
    const float *ix,
    const float *iy,
    float *iz,
    const int inum,
    const float area,
    const float *avg_dist,
    const float R_min,
    const float R_max,
    const float a1,
    const float a2,
    const float a3,
    const float a4,
    const float a5,
    const float EPS)
{
  #pragma omp parallel for
  for(int tid = 0; tid < inum; tid++) {  
    float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 1.f, u_R = 0.f;

    float r_obs = avg_dist[tid];               //The observed average nearest neighbor distance
    float r_exp = 0.5f / sqrtf(dnum / area); // The expected nearest neighbor distance for a random pattern
    float R_S0 = r_obs / r_exp;                //The nearest neighbor statistic

    // Normalize the R(S0) measure such that it is bounded by 0 and 1 by a fuzzy membership function 
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
}

// Calculate the power parameter, and then weighted interpolating
// With using shared memory (Tiled version of the stage 2)
void AIDW_Tiled(
    const float *dx, 
    const float *dy,
    const float *dz,
    const int dnum,
    const float *ix,
    const float *iy,
    float *iz,
    const int inum,
    const float area,
    const float *avg_dist,
    const float R_min,
    const float R_max,
    const float a1,
    const float a2,
    const float a3,
    const float a4,
    const float a5,
    const float EPS) 
{ 
  #pragma omp parallel for
  for(int i = 0; i <= dnum; i += BLOCK_SIZE) {
    float sum_up = 0.f;
    float sum_dn = 0.f;
    for(int m = 0; m < i + BLOCK_SIZE && m < dnum; m++) {
      float sdx[BLOCK_SIZE], sdy[BLOCK_SIZE], sdz[BLOCK_SIZE];
      for(int j = 0; j < m && j < BLOCK_SIZE; j++) {
        sdx[j] = dx[m + j];
        sdy[j] = dy[m + j];
        sdz[j] = dz[m + j];
      }
      AIDW(sdx, sdy, sdz, dnum, ix, iy, iz, inum, area, avg_dist, R_min, R_max, a1, a2, a3, a4, a5, EPS);
    }
   } 
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <pts> <check> <iterations>" << std::endl;
    std::cout << "pts: number of points (unit: 1K)" << std::endl;
    std::cout << "check: enable verification when the value is 1" << std::endl;
    return 1;
  }

  const int numk = atoi(argv[1]); // number of points (unit: 1K)
  const bool check = atoi(argv[2]); // do check for small problem sizes
  const int iterations = atoi(argv[3]); // repeat kernel execution

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
  std::vector<float> h_iz(inum);

  std::default_random_engine generator(123);
  std::uniform_real_distribution<float> real(0.0, 1.0);

  for(int i = 0; i < dnum; i++) {
    dx[i] = real(generator) * 1000;
    dy[i] = real(generator) * 1000;
    dz[i] = real(generator) * 1000;
  }

  for(int i = 0; i < inum; i++) {
    ix[i] = real(generator) * 1000;
    iy[i] = real(generator) * 1000;
    iz[i] = 0.f;
  }

  for(int i = 0; i < dnum; i++)
    avg_dist[i] = real(generator) * 3;

  std::cout << "Size = : " << numk << " K " << std::endl;
  std::cout << "dnum = : " << dnum << std::endl;
  std::cout << "inum = : " << inum << std::endl;
  
  if (check) {
    reference(dx.data(), dy.data(), dz.data(), dnum, ix.data(), 
               iy.data(), h_iz.data(), inum, area, avg_dist.data());
    std::cout << "Verification enabled" << std::endl;
  } else {
    std::cout << "Verification disabled" << std::endl;
  }

  auto start = std::chrono::steady_clock::now();
  
  for (int i = 0; i < iterations; i++) 
    AIDW(dx.data(), dy.data(), dz.data(), dnum, ix.data(), iy.data(), iz.data(), inum, area, avg_dist.data());

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of AIDW              " << time * 1e-9f / iterations << std::endl;

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++) 
    AIDW_Tiled(dx.data(), dy.data(), dz.data(), dnum, ix.data(), iy.data(), iz.data(), inum, area, avg_dist.data(), R_min, R_max, a1, a2, a3, a4, a5, EPS);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of AIDW_Tiled        " << time * 1e-9f / iterations << std::endl;

  if (check) {
    bool ok = verify(iz.data(), h_iz.data(), inum, EPS);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  }

  return 0;
}