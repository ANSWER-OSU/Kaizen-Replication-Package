void core (
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const double *__restrict__ u,
    const double *__restrict__ v,
    const int x_points,
    const int y_points,
    const double nu,
    const double del_t,
    const double del_x,
    const double del_y)
{
  #pragma omp target teams distribute parallel for collapse(2)
  for (int i = 1; i < y_points - 1; i++) {
    for (int j = 1; j < x_points - 1; j++) {
      u_new[i*x_points+j] = u[i*x_points+j] + 
        (nu*del_t/(del_x*del_x)) * (u[i*x_points+j+1] + u[i*x_points+j-1] - 2 * u[i*x_points+j]) + 
        (nu*del_t/(del_y*del_y)) * (u[(i+1)*x_points+j] + u[(i-1)*x_points+j] - 2 * u[i*x_points+j]) - 
        (del_t/del_x)*u[i*x_points+j] * (u[i*x_points+j] - u[i*x_points+j-1]) - 
        (del_t/del_y)*v[i*x_points+j] * (u[i*x_points+j] - u[(i-1)*x_points+j]);

      v_new[i*x_points+j] = v[i*x_points+j] +
        (nu*del_t/(del_x*del_x)) * (v[i*x_points+j+1] + v[i*x_points+j-1] - 2 * v[i*x_points+j]) + 
        (nu*del_t/(del_y*del_y)) * (v[(i+1)*x_points+j] + v[(i-1)*x_points+j] - 2 * v[i*x_points+j]) -
        (del_t/del_x)*u[i*x_points+j] * (v[i*x_points+j] - v[i*x_points+j-1]) - 
        (del_t/del_y)*v[i*x_points+j] * (v[i*x_points+j] - v[(i-1)*x_points+j]);
    }
  }
}

void bound_h (
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const int x_points,
    const int y_points)
{
  #pragma omp parallel for
  for (int i = 0; i < x_points; i++) {
    u_new[i] = 1.0;
    v_new[i] = 1.0;
    u_new[(y_points-1)*x_points+i] = 1.0;
    v_new[(y_points-1)*x_points+i] = 1.0;
  }
}

void bound_v (
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const int x_points,
    const int y_points)
{
  #pragma omp parallel for
  for (int j = 0; j < y_points; j++) {
    u_new[j*x_points] = 1.0;
    v_new[j*x_points] = 1.0;
    u_new[j*x_points+(x_points-1)] = 1.0;
    v_new[j*x_points+(x_points-1)] = 1.0;
  }
}

void update (
    double *__restrict__ u,
    double *__restrict__ v,
    const double *__restrict__ u_new,
    const double *__restrict__ v_new,
    const int n)
{
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    u[i] = u_new[i];
    v[i] = v_new[i];
  }
}