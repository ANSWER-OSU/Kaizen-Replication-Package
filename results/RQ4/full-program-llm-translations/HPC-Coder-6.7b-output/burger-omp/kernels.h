#include <omp.h>

#define idx(i,j)   (i)*x_points+(j)

void core(
    double *u_new,
    double *v_new,
    const double *u,
    const double *v,
    const int x_points,
    const int y_points,
    const double nu,
    const double del_t,
    const double del_x,
    const double del_y)
{
  #pragma omp parallel for schedule(static)
  for (int j = 1; j < x_points - 1; j++) {
    for (int i = 1; i < y_points - 1; i++) {
      u_new[idx(i,j)] = u[idx(i,j)] + 
        (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) + 
        (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) - 
        (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) - 
        (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

      v_new[idx(i,j)] = v[idx(i,j)] +
        (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) + 
        (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
        (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) - 
        (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
    }
  }
}

void bound_h(
    double *u_new,
    double *v_new,
    const int x_points,
    const int y_points)
{
  #pragma omp parallel for schedule(static)
  for (int j = 0; j < x_points; j++) {
    u_new[idx(0,j)] = 1.0;
    v_new[idx(0,j)] = 1.0;
    u_new[idx(y_points-1,j)] = 1.0;
    v_new[idx(y_points-1,j)] = 1.0;
  }
}

void bound_v(
    double *u_new,
    double *v_new,
    const int x_points,
    const int y_points)
{
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < y_points; i++) {
    u_new[idx(i,0)] = 1.0;
    v_new[idx(i,0)] = 1.0;
    u_new[idx(i,x_points-1)] = 1.0;
    v_new[idx(i,x_points-1)] = 1.0;
  }
}

void update(
    double *u,
    double *v,
    const double *u_new,
    const double *v_new,
    const int n)
{
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    u[i] = u_new[i];
    v[i] = v_new[i];
  }
}