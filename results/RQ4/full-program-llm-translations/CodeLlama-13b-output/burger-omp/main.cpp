#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>

void core(double *d_u_new, double *d_v_new, double *d_u, double *d_v, int x_points, int y_points, double nu, double del_t, double del_x, double del_y)
{
  #pragma omp parallel for
  for(int i = 1; i < y_points-1; i++){
    for(int j = 1; j < x_points-1; j++){
      d_u_new[idx(i,j)] = d_u[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (d_u[idx(i,j+1)] + d_u[idx(i,j-1)] - 2 * d_u[idx(i,j)]) + 
        (nu*del_t/(del_y*del_y)) * (d_u[idx(i+1,j)] + d_u[idx(i-1,j)] - 2 * d_u[idx(i,j)]) - 
        (del_t/del_x)*d_u[idx(i,j)] * (d_u[idx(i,j)] - d_u[idx(i,j-1)]) - 
        (del_t/del_y)*d_v[idx(i,j)] * (d_u[idx(i,j)] - d_v[idx(i-1,j)]);

      d_v_new[idx(i,j)] = d_v[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (d_v[idx(i,j+1)] + d_v[idx(i,j-1)] - 2 * d_v[idx(i,j)]) + 
        (nu*del_t/(del_y*del_y)) * (d_v[idx(i+1,j)] + d_v[idx(i-1,j)] - 2 * d_v[idx(i,j)]) -
        (del_t/del_x)*d_u[idx(i,j)] * (d_v[idx(i,j)] - d_u[idx(i,j-1)]) - 
        (del_t/del_y)*d_v[idx(i,j)] * (d_v[idx(i,j)] - d_v[idx(i-1,j)]);
    }
  }
}

void bound_h(double *d_u_new, double *d_v_new, int x_points, int y_points)
{
  #pragma omp parallel for
  for(int i = 0; i < x_points; i++){
    d_u_new[idx(0,i)] = 1.0;
    d_v_new[idx(0,i)] = 1.0;
    d_u_new[idx(y_points-1,i)] = 1.0;
    d_v_new[idx(y_points-1,i)] = 1.0;
  }
}

void bound_v(double *d_u_new, double *d_v_new, int x_points, int y_points)
{
  #pragma omp parallel for
  for(int i = 0; i < y_points; i++){
    d_u_new[idx(i,0)] = 1.0;
    d_v_new[idx(i,0)] = 1.0;
    d_u_new[idx(i,x_points-1)] = 1.0;
    d_v_new[idx(i,x_points-1)] = 1.0;
  }
}

void update(double *d_u, double *d_v, double *d_u_new, double *d_v_new, int grid_elems)
{
  #pragma omp parallel for
  for(int i = 0; i < grid_elems; i++){
    d_u[i] = d_u_new[i];
    d_v[i] = d_v_new[i];
  }
}
#pragma omp parallel
for(int itr = 0; itr < num_iters; itr++){

  core(d_u_new, d_v_new, d_u, d_v, x_points, y_points, nu, del_t, del_x, del_y);

  // Boundary conditions
  bound_h(d_u_new, d_v_new, x_points, y_points);

  bound_v(d_u_new, d_v_new, x_points, y_points);

  // Updating older values to newer ones
  update(d_u, d_v, d_u_new, d_v_new, grid_elems);
}