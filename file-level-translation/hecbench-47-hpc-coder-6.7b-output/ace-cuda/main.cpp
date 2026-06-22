#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>

//define the data set size (cubic volume)
#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400

typedef double nRarray[DATAYSIZE][DATAZSIZE];

// square
#define SQ(x) ((x)*(x))

#ifdef VERIFY
#include <string.h>
#include "reference.h"
#endif

double dFphi(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}

double GradientX(double phi[][DATAYSIZE][DATAZSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x+1][y][z] - phi[x-1][y][z]) / (2.0*dx);
}

double GradientY(double phi[][DATAYSIZE][DATAZSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y+1][z] - phi[x][y-1][z]) / (2.0*dy);
}

double GradientZ(double phi[][DATAYSIZE][DATAZSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y][z+1] - phi[x][y][z-1]) / (2.0*dz);
}

double Divergence(double phix[][DATAYSIZE][DATAZSIZE],
                  double phiy[][DATAYSIZE][DATAZSIZE],
                  double phiz[][DATAYSIZE][DATAZSIZE],
                  double dx, double dy, double dz, int x, int y, int z)
{
  return GradientX(phix,dx,dy,dz,x,y,z) +
         GradientY(phiy,dx,dy,dz,x,y,z) +
         GradientZ(phiz,dx,dy,dz,x,y,z);
}

double Laplacian(double phi[][DATAYSIZE][DATAZSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  double phixx = (phi[x+1][y][z] + phi[x-1][y][z] - 2.0 * phi[x][y][z]) / SQ(dx);
  double phiyy = (phi[x][y+1][z] + phi[x][y-1][z] - 2.0 * phi[x][y][z]) / SQ(dy);
  double phizz = (phi[x][y][z+1] + phi[x][y][z-1] - 2.0 * phi[x][y][z]) / SQ(dz);
  return phixx + phiyy + phizz;
}

double An(double phix, double phiy, double phiz, double epsilon)
{
  if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
    return ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon))*
           ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
           ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
  }
  else
  {
    return (1.0-((5.0/3.0)*epsilon));
  }
}

double Wn(double phix, double phiy, double phiz, double epsilon, double W0)
{
  return (W0*An(phix,phiy,phiz,epsilon));
}

double taun(double phix, double phiy, double phiz, double epsilon, double tau0)
{
  return tau0 * SQ(An(phix,phiy,phiz,epsilon));
}

double dFunc(double l, double m, double n)
{
  if (l != 0.0 || m != 0.0 || n != 0.0){
    return (((l*l*l*(SQ(m)+SQ(n)))-(l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
            ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
  }
  else
  {
    return 0.0;
  }
}

void calculateForce(double phi[][DATAYSIZE][DATAZSIZE],
                    double Fx[][DATAYSIZE][DATAZSIZE],
                    double Fy[][DATAYSIZE][DATAZSIZE],
                    double Fz[][DATAYSIZE][DATAZSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0){

#pragma omp target teams map(to: phi[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE], dx, dy, dz) map(to: epsilon, W0, tau0) map(tofrom: Fx[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE], Fy[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE], Fz[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE])
#pragma omp distribute parallel for collapse(3) num_teams(8*8*4) team_num_tasks(8*8*4)
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {

        if ((idx < (DATAXSIZE-1)) && (idy < (DATAYSIZE-1)) &&
              (idz < (DATAZSIZE-1)) && (idx > (0)) &&
              (idy > (0)) && (idz > (0))) {
          double phix = GradientX(phi,dx,dy,dz,idx,idy,idz);
          double phiy = GradientY(phi,dx,dy,dz,idx,idy,idz);
          double phiz = GradientZ(phi,dx,dy,dz,idx,idy,idz);
          double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
          double c = 16.0 * W0 * epsilon;
          double w = Wn(phix,phiy,phiz,epsilon,W0);
          double w2 = SQ(w);

          Fx[idx][idy][idz] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
          Fy[idx][idy][idz] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
          Fz[idx][idy][idz] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
        }
      }
    }
  }
}

void allenCahn(double phinew[][DATAYSIZE][DATAZSIZE],
               double phiold[][DATAYSIZE][DATAZSIZE],
               double uold[][DATAYSIZE][DATAZSIZE],
               double Fx[][DATAYSIZE][DATAZSIZE],
               double Fy[][DATAYSIZE][DATAZSIZE],
               double Fz[][DATAYSIZE][DATAZSIZE],
               double epsilon, double W0, double tau0, double lambda,
               double dt, double dx, double dy, double dz)
{
#pragma omp target teams map(to: phiold[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE], Fx[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE], Fy[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE], Fz[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE]) map(to: epsilon, W0, tau0, lambda) map(to: phinew[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE])
#pragma omp distribute parallel for collapse(3) num_teams(8*8*4) team_num_tasks(8*8*4)
  for (unsigned iz = 0; iz < DATAZSIZE; iz++) {
    for (unsigned iy = 0; iy < DATAYSIZE; iy++) {
      for (unsigned ix = 0; ix < DATAXSIZE; ix++) {
        if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) &&
            (iz < (DATAZSIZE-1)) && (ix > (0)) &&
            (iy > (0)) && (iz > (0))) {
          double phix = GradientX(phiold,dx,dy,dz,ix,iy,iz);
          double phiy = GradientY(phiold,dx,dy,dz,ix,iy,iz);
          double phiz = GradientZ(phiold,dx,dy,dz,ix,iy,iz);

          phinew[ix][iy][iz] = phiold[ix][iy][iz] +
               (dt / taun(phix,phiy,phiz,epsilon,tau0)) *
               (Divergence(Fx,Fy,Fz,dx,dy,dz,ix,iy,iz) -
                dFphi(phiold[ix][iy][iz], uold[ix][iy][iz],lambda));
        }
      }
    }
  }
}

void boundaryConditionsPhi(double phinew[][DATAYSIZE][DATAZSIZE]){
#pragma omp target teams map(to: phinew[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE])
#pragma omp distribute parallel for collapse(3) num_teams(8*8*4) team_num_tasks(8*8*4)
  for (unsigned iz = 0; iz < DATAZSIZE; iz++) {
    for (unsigned iy = 0; iy < DATAYSIZE; iy++) {
      for (unsigned ix = 0; ix < DATAXSIZE; ix++) {
        if (ix >= DATAXSIZE || iy >= DATAYSIZE || iz >= DATAZSIZE) continue;

        if (ix == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (ix == DATAXSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iy == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iy == DATAYSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iz == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iz == DATAZSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
      }
    }
  }
}

void thermalEquation(double unew[][DATAYSIZE][DATAZSIZE],
                     double uold[][DATAYSIZE][DATAZSIZE],
                     double phinew[][DATAYSIZE][DATAZSIZE],
                     double phiold[][DATAYSIZE][DATAZSIZE],
                     double D, double dt, double dx, double dy, double dz){
#pragma omp target teams map(to: phinew[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE], phiold[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE]) map(to: D) map(to: dt, dx, dy, dz) map(tofrom: unew[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE])
#pragma omp distribute parallel for collapse(3) num_teams(8*8*4) team_num_tasks(8*8*4)
  for (unsigned iz = 0; iz < DATAZSIZE; iz++) {
    for (unsigned iy = 0; iy < DATAYSIZE; iy++) {
      for (unsigned ix = 0; ix < DATAXSIZE; ix++) {
        if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) &&
            (iz < (DATAZSIZE-1)) && (ix > (0)) &&
            (iy > (0)) && (iz > (0))){
          unew[ix][iy][iz] = uold[ix][iy][iz] +
              0.5*(phinew[ix][iy][iz]- phiold[ix][iy][iz]) +
              dt * D * Laplacian(uold,dx,dy,dz,ix,iy,iz);
        }
      }
    }
  }
}

void boundaryConditionsU(double unew[][DATAYSIZE][DATAZSIZE], double delta){
#pragma omp target teams map(tofrom: unew[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE])
#pragma omp distribute parallel for collapse(3) num_teams(8*8*4) team_num_tasks(8*8*4)
  for (unsigned iz = 0; iz < DATAZSIZE; iz++) {
    for (unsigned iy = 0; iy < DATAYSIZE; iy++) {
      for (unsigned ix = 0; ix < DATAXSIZE; ix++) {
        if (iz >= DATAZSIZE || iy >= DATAYSIZE || ix >= DATAXSIZE) continue;

        if (ix == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (ix == DATAXSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iy == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iy == DATAYSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iz == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iz == DATAZSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
      }
    }
  }
}

void initializationPhi(double phi[][DATAYSIZE][DATAZSIZE], double r0)
{
#pragma omp target teams colaapse(3) map(to: r0) map(tofrom: phi[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE])
#pragma omp distribute parallel for num_teams(8*8*4) team_num_tasks(8*8*4)
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx-0.5*DATAXSIZE) + SQ(idy-0.5*DATAYSIZE) + SQ(idz-0.5*DATAZSIZE));
        if (r < r0)
          phi[idx][idy][idz] = 1.0;
        else
          phi[idx][idy][idz] = -1.0;
      }
    }
  }
}

void initializationU(double u[][DATAYSIZE][DATAZSIZE], double r0, double delta)
{
#pragma omp target teams colaapse(3) map(to: r0, delta) map(tofrom: u[:DATAXSIZE][:DATAYSIZE][:DATAZSIZE])
#pragma omp distribute parallel for num_teams(8*8*4) team_num_tasks(8*8*4)
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx-0.5*DATAXSIZE) + SQ(idy-0.5*DATAYSIZE) + SQ(idz-0.5*DATAZSIZE));
        if (r < r0)
          u[idx][idy][idz] = 0.0;
        else
          u[idx][idy][idz] = -delta * (1.0 - std::exp(-(r-r0)));
      }
    }
  }
}

int main(int argc, char *argv[])
{
  const int num_steps = atoi(argv[1]);  //6000;
  const double dx = 0.4;
  const double dy = 0.4;
  const double dz = 0.4;
  const double dt = 0.01;
  const double delta = 0.8;
  const double r0 = 5.0;
  const double epsilon = 0.07;
  const double W0 = 1.0;
  const double beta0 = 0.0;
  const double D = 2.0;
  const double d0 = 0.5;
  const double a1 = 1.25 / std::sqrt(2.0);
  const double a2 = 0.64;
  const double lambda = (W0*a1)/(d0);
  const double tau0 = ((W0*W0*W0*a1*a2)/(d0*D)) + ((W0*W0*beta0)/(d0));

  // overall data set sizes
  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_in_bytes = sizeof(double) * vol;

  // pointers for data set storage via malloc
  nRarray *phi_host;
  nRarray *d_phiold;
  nRarray *u_host;
  nRarray *d_phinew;
  nRarray *d_uold;
  nRarray *d_unew;
  nRarray *d_Fx;
  nRarray *d_Fy;
  nRarray *d_Fz;

  phi_host = (nRarray *)malloc(vol_in_bytes);
  u_host = (nRarray *)malloc(vol_in_bytes);

  initializationPhi(phi_host,r0);
  initializationU(u_host,r0,delta);

#ifdef VERIFY
  nRarray *phi_ref = (nRarray *)malloc(vol_in_bytes);
  nRarray *u_ref = (nRarray *)malloc(vol_in_bytes);
  memcpy(phi_ref, phi_host, vol_in_bytes);
  memcpy(u_ref, u_host, vol_in_bytes);
  reference(phi_ref, u_ref, vol, num_steps);
#endif

  auto offload_start = std::chrono::steady_clock::now();

  // allocate GPU device buffers
  d_phiold = (nRarray *)malloc(vol_in_bytes);
  d_phinew = (nRarray *)malloc(vol_in_bytes);
  d_uold = (nRarray *)malloc(vol_in_bytes);
  d_unew = (nRarray *)malloc(vol_in_bytes);
  d_Fx = (nRarray *)malloc(vol_in_bytes);
  d_Fy = (nRarray *)malloc(vol_in_bytes);
  d_Fz = (nRarray *)malloc(vol_in_bytes);

  memcpy(d_phiold, phi_host, vol_in_bytes);
  memcpy(d_uold, u_host, vol_in_bytes);

  const int num_threads = -1;
  const int tile_shape = 8;
  const int order = 0;
  const int continuity_order = 2;

  int t = 0;

  auto start = std::chrono::steady_clock::now();

  while (t <= num_steps) {

    calculateForce(d_phiold,d_Fx,d_Fy,d_Fz,
                   dx,dy,dz,epsilon,W0,tau0);

    allenCahn(d_phinew,d_phiold,d_uold,
              d_Fx,d_Fy,d_Fz,
              epsilon,W0,tau0,lambda,
              dt,dx,dy,dz);

    boundaryConditionsPhi(d_phinew);

    thermalEquation(d_unew,d_uold,d_phinew,d_phiold,
                    D,dt,dx,dy,dz);

    boundaryConditionsU(d_unew,delta);

    memcpy(d_phiold, d_phinew, vol_in_bytes);
    memcpy(d_uold, d_unew, vol_in_bytes);

    t++;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %.3f (ms)\n", time * 1e-6f);

  memcpy(phi_host, d_phiold, vol_in_bytes);
  memcpy(u_host, d_uold, vol_in_bytes);

  free(d_phiold);
  free(d_phinew);
  free(d_uold);
  free(d_unew);
  free(d_Fx);
  free(d_Fy);
  free(d_Fz);

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();
  printf("Offload time: %.3f (ms)\n", offload_time * 1e-6f);

#ifdef VERIFY
  bool ok = true;
  for (int idx = 0; idx < nx; idx++)
    for (int idy = 0; idy < ny; idy++)
      for (int idz = 0; idz < nz; idz++) {
        if (fabs(phi_ref[idx][idy][idz] - phi_host[idx][idy][idz]) > 1e-3) {
          ok = false; printf("phi: %lf %lf\n", phi_ref[idx][idy][idz], phi_host[idx][idy][idz]);
	}
        if (fabs(u_ref[idx][idy][idz] - u_host[idx][idy][idz]) > 1e-3) {
          ok = false; printf("u: %lf %lf\n", u_ref[idx][idy][idz], u_host[idx][idy][idz]);
        }
      }
  printf("%s\n", ok ? "PASS" : "FAIL");
  free(phi_ref);
  free(u_ref);
#endif

  free(phi_host);
  free(u_host);
  return 0;
}