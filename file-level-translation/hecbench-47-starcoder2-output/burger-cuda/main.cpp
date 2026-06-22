#include <iostream>
#include <vector>
#include <omp.h>

void core(std::vector<double> &u_new, std::vector<double> &v_new, std::vector<double> &u, std::vector<double> &v,
          int x_points, int y_points, double nu, double del_t, double del_x, double del_y)
{
  #pragma omp parallel for collapse(2)
  for (int i = 1; i < y_points - 1; i++)
  {
    for (int j = 1; j < x_points - 1; j++)
    {
      u_new[i * x_points + j] = u[i * x_points + j] +
                                 (nu * del_t / (del_x * del_x)) * (u[i * x_points + j + 1] + u[i * x_points + j - 1] - 2 * u[i * x_points + j]) +
                                 (nu * del_t / (del_y * del_y)) * (u[(i + 1) * x_points + j] + u[(i - 1) * x_points + j] - 2 * u[i * x_points + j]) -
                                 (del_t / del_x) * u[i * x_points + j] * (u[i * x_points + j] - u[i * x_points + j - 1]) -
                                 (del_t / del_y) * v[i * x_points + j] * (u[i * x_points + j] - u[(i - 1) * x_points + j]);
      v_new[i * x_points + j] = v[i * x_points + j] +
                                 (nu * del_t / (del_x * del_x)) * (v[i * x_points + j + 1] + v[i * x_points + j - 1] - 2 * v[i * x_points + j]) +
                                 (nu * del_t / (del_y * del_y)) * (v[(i + 1) * x_points + j] + v[(i - 1) * x_points + j] - 2 * v[i * x_points + j]) -
                                 (del_t / del_x) * u[i * x_points + j] * (v[i * x_points + j] - v[i * x_points + j - 1]) -
                                 (del_t / del_y) * v[i * x_points + j] * (v[i * x_points + j] - v[(i - 1) * x_points + j]);
    }
  }
}

void bound_h(std::vector<double> &u_new, std::vector<double> &v_new, int x_points, int y_points)
{
  #pragma omp parallel for
  for (int i = 1; i < y_points - 1; i++)
  {
    u_new[i * x_points] = 1.0;
    v_new[i * x_points] = 1.0;
    u_new[(i + 1) * x_points - 1] = 1.0;
    v_new[(i + 1) * x_points - 1] = 1.0;
  }
}

void bound_v(std::vector<double> &u_new, std::vector<double> &v_new, int x_points, int y_points)
{
  #pragma omp parallel for
  for (int j = 0; j < x_points; j++)
  {
    u_new[j] = 1.0;
    v_new[j] = 1.0;
    u_new[(y_points - 1) * x_points + j] = 1.0;
    v_new[(y_points - 1) * x_points + j] = 1.0;
  }
}

void update(std::vector<double> &u, std::vector<double> &v,
            std::vector<double> &u_new, std::vector<double> &v_new, int grid_size)
{
  #pragma omp parallel for
  for (int i = 0; i < grid_size; i++)
  {
    u[i] = u_new[i];
    v[i] = v_new[i];
  }
}

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    printf("Usage: %s <dim_x> <dim_y> <nt>\n", argv[0]);
    printf("dim_x: number of grid points in the x axis\n");
    printf("dim_y: number of grid points in the y axis\n");
    printf("nt: number of time steps\n");
    return -1;
  }

  // Define the domain
  const int x_points = std::stoi(argv[1]);
  const int y_points = std::stoi(argv[2]);
  const int num_itrs = std::stoi(argv[3]);
  const double x_len = 2.0;
  const double y_len = 2.0;
  const double del_x = x_len / (x_points - 1);
  const double del_y = y_len / (y_points - 1);

  const int grid_elems = x_points * y_points;

  std::vector<double> x(x_points);
  std::vector<double> y(y_points);
  std::vector<double> u(grid_elems);
  std::vector<double> v(grid_elems);
  std::vector<double> u_new(grid_elems);
  std::vector<double> v_new(grid_elems);

  // Define the parameters
  const double nu = 0.01;
  const double sigma = 0.0009;
  const double del_t = sigma * del_x * del_y / nu; // CFL criteria

  for (int i = 0; i < x_points; i++)
  {
    x[i] = i * del_x;
  }
  for (int i = 0; i < y_points; i++)
  {
    y[i] = i * del_y;
  }

  for (int i = 0; i < y_points; i++)
  {
    for (int j = 0; j < x_points; j++)
    {
      u[i * x_points + j] = 1.0;
      v[i * x_points + j] = 1.0;
      u_new[i * x_points + j] = 1.0;
      v_new[i * x_points + j] = 1.0;
      if (x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0)
      {
        u[i * x_points + j] = 2.0;
        v[i * x_points + j] = 2.0;
        u_new[i * x_points + j] = 2.0;
        v_new[i * x_points + j] = 2.0;
      }
    }
  }

  double start = omp_get_wtime();

  for (int itr = 0; itr < num_itrs; itr++)
  {
    core(u_new, v_new, u, v, x_points, y_points, nu, del_t, del_x, del_y);

    // Boundary conditions
    bound_h(u_new, v_new, x_points, y_points);
    bound_v(u_new, v_new, x_points, y_points);

    // Updating older values to newer ones
    update(u, v, u_new, v_new, grid_elems);
  }

  double end = omp_get_wtime();
  double time = end - start;
  printf("Total kernel execution time %f (s)\n", time);
  return 0;
}