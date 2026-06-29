#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


void haccmk_kernel(int n1, int n2, const float *__restrict__ xx, 
                    const float *__restrict__ yy, 
                    const float *__restrict__ zz, 
                    const float *__restrict__ mass,
                    float *__restrict__ vx2,
                    float *__restrict__ vy2, 
                    float *__restrict__ vz2, 
                    float fsrmax,
                    float mp_rsm,
                    float fcoeff ){
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n1; i++) {
    float ma0 = 0.269327;
    float ma1 = -0.0750978; 
    float ma2 = 0.0114808;
    float ma3 = -0.00109313; 
    float ma4 = 0.0000605491; 
    float ma5 = -0.00000147177;

    float dxc, dyc, dzc, m, r2, f, xi = 0.f, yi = 0.f, zi = 0.f, xxi = xx[i], yyi = yy[i], zzi = zz[i];

    for (int j = 0; j < n2; j++) {
      dxc = xx[j] - xxi;
      dyc = yy[j] - yyi;
      dzc = zz[j] - zzi;

      r2 = dxc * dxc + dyc * dyc + dzc * dzc;

      if (r2 < fsrmax)
        m = mass[j];
      else
        m = 0.f;

      f = r2 + mp_rsm;
      f = m * (1.f / (f * sqrtf(f)) - (ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))));

      xi += f * dxc;
      yi += f * dyc;
      zi += f * dzc;
    }

    vx2[i] += xi * fcoeff;
    vy2[i] += yi * fcoeff;
    vz2[i] += zi * fcoeff;
  }
}

void haccmk(int repeat,
            int n1,
            int n2,
            const float* xx, 
            const float* yy,
            const float* zz,
            const float* mass,
            float* vx2,
            float* vy2,
            float* vz2,
            float fsrmax,
            float mp_rsm,
            float fcoeff ) {

  float *xx_omp, *yy_omp, *zz_omp, *mass_omp, *vx2_omp, *vy2_omp, *vz2_omp;
  xx_omp = (float*) malloc (sizeof(float) * n2);
  yy_omp = (float*) malloc (sizeof(float) * n2);
  zz_omp = (float*) malloc (sizeof(float) * n2);
  mass_omp = (float*) malloc (sizeof(float) * n2);
  vx2_omp = (float*) malloc (sizeof(float) * n1);
  vy2_omp = (float*) malloc (sizeof(float) * n1);
  vz2_omp = (float*) malloc (sizeof(float) * n1);

  memcpy(xx_omp, xx, sizeof(float) * n2);
  memcpy(yy_omp, yy, sizeof(float) * n2);
  memcpy(zz_omp, zz, sizeof(float) * n2);
  memcpy(mass_omp, mass, sizeof(float) * n2);

  float total_time = 0.f;

  for (int i = 0; i < repeat; i++) {
    for (int i = 0; i < n1; i++) {
      vx2_omp[i] = 0.f;
      vy2_omp[i] = 0.f;
      vz2_omp[i] = 0.f;
    }

    auto start = std::chrono::steady_clock::now();

    haccmk_kernel(n1, n2, xx_omp, yy_omp, zz_omp, mass_omp, vx2_omp, vy2_omp, vz2_omp,
                  fsrmax, mp_rsm, fcoeff);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / repeat);

  memcpy(vx2, vx2_omp, sizeof(float) * n1);
  memcpy(vy2, vy2_omp, sizeof(float) * n1);
  memcpy(vz2, vz2_omp, sizeof(float) * n1);

  free(xx_omp);
  free(yy_omp);
  free(zz_omp);
  free(mass_omp);
  free(vx2_omp);
  free(vy2_omp);
  free(vz2_omp);
}

void haccmk_gold(int count1,
                 float xxi,
                 float yyi,
                 float zzi,
                 float fsrrmax2,
                 float mp_rsm2, 
                 float *__restrict__ xx1, 
                 float *__restrict__ yy1, 
                 float *__restrict__ zz1, 
                 float *__restrict__ mass1, 
                 float *__restrict__ dxi,
                 float *__restrict__ dyi,
                 float *__restrict__ dzi ){
  int j;
  float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, 
        ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;
  float dxc, dyc, dzc, m, r2, f, xi = 0.f, yi = 0.f, zi = 0.f;

  for ( j = 0; j < count1; j++ ) {
    dxc = xx1[j] - xxi;
    dyc = yy1[j] - yyi;
    dzc = zz1[j] - zzi;

    r2 = dxc * dxc + dyc * dyc + dzc * dzc;

    if ( r2 < fsrrmax2 ) m = mass1[j];
    else m = 0.f;

    f = r2 + mp_rsm2;
    f =  m * ( 1.f / ( f * sqrtf( f ) ) - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))));

    xi += f * dxc;
    yi += f * dyc;
    zi += f * dzc;
  }

  *dxi = xi;
  *dyi = yi;
  *dzi = zi;
}

int main( int argc, char *argv[] ){
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  float fcoeff, fsrrmax2, mp_rsm2, dx1, dy1, dz1;
  int n1, n2, i;
  n1 = 784;
  n2 = 15000;
  printf( "Outer loop count is set %d\n", n1 );
  printf( "Inner loop count is set %d\n", n2 );

  float *xx = (float*) malloc (sizeof(float) * n2);
  float *yy = (float*) malloc (sizeof(float) * n2);
  float *zz = (float*) malloc (sizeof(float) * n2);
  float *mass = (float*) malloc (sizeof(float) * n2);
  float *vx2 = (float*) malloc (sizeof(float) * n2);
  float *vy2 = (float*) malloc (sizeof(float) * n2);
  float *vz2 = (float*) malloc (sizeof(float) * n2);
  float *vx2_hw = (float*) malloc (sizeof(float) * n2);
  float *vy2_hw = (float*) malloc (sizeof(float) * n2);
  float *vz2_hw = (float*) malloc (sizeof(float) * n2);
  float dx2, dy2, dz2;

  /* Initial data preparation */
  fcoeff = 0.23f;
  fsrrmax2 = 0.5f; 
  mp_rsm2 = 0.03f;
  dx1 = 1.0f/(float)n2;
  dy1 = 2.0f/(float)n2;
  dz1 = 3.0f/(float)n2;
  xx[0] = 0.f;
  yy[0] = 0.0f;
  zz[0] = 0.f;
  mass[0] = 2.f;

  for ( i = 1; i < n2; i++ ) {
    xx[i] = xx[i-1] + dx1;
    yy[i] = yy[i-1] + dy1;
    zz[i] = zz[i-1] + dz1;
    mass[i] = (float)i * 0.01f + xx[i];
  }

  for ( i = 0; i < n2; i++ ) {
    vx2[i] = 0.f;
    vy2[i] = 0.f;
    vz2[i] = 0.f;
    vx2_hw[i] = 0.f;
    vy2_hw[i] = 0.f;
    vz2_hw[i] = 0.f;
  }

  for ( i = 0; i < n1; ++i) {
    haccmk_gold( n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, 
                  &dx2, &dy2, &dz2 );    
    vx2[i] = vx2[i] + dx2 * fcoeff;
    vy2[i] = vy2[i] + dy2 * fcoeff;
    vz2[i] = vz2[i] + dz2 * fcoeff;
  }

  haccmk(repeat, n1, n2, xx, yy, zz, mass,
         vx2_hw, vy2_hw, vz2_hw, fsrrmax2, mp_rsm2, fcoeff);

  // verify
  int error = 0;
  float eps = 1.0f;
  for (i = 0; i < n2; i++) {
    if (fabsf(vx2[i] - vx2_hw[i]) > eps) {
      printf("error at vx2[%d] %f %f\n", i, vx2[i], vx2_hw[i]);
      error = 1;
      break;
    }
    if (fabsf(vy2[i] - vy2_hw[i]) > eps) {
      printf("error at vy2[%d]: %f %f\n", i, vy2[i], vy2_hw[i]);
      error = 1;
      break;
    }
    if (fabsf(vz2[i] - vz2_hw[i]) > eps) {
      printf("error at vz2[%d]: %f %f\n", i, vz2[i], vz2_hw[i]);
      error = 1;
      break;
    }
  }
  
  free(xx);
  free(yy);
  free(zz);
  free(mass);
  free(vx2);
  free(vy2);
  free(vz2);
  free(vx2_hw);
  free(vy2_hw);
  free(vz2_hw);

  printf("%s\n", error ? "FAIL" : "PASS");

  return 0;
}