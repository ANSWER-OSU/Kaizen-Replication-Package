#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <chrono>

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <dim_x> <dim_y> <nt>\n", argv[0]);
    printf("dim_x: number of grid points in the x axis\n");
    printf("dim_y: number of grid points in the y axis\n");
    printf("nt: number of time steps\n");
    exit(-1);
  }

  // Define the domain
  const int x_points = atoi(argv[1]);
  const int y_points = atoi(argv[2]);
  const int num_itrs = atoi(argv[3]);
  const double x_len = 2.0;
  const double y_len = 2.0;
  const double del_x = x_len/(x_points-1);
  const double del_y = y_len/(y_points-1);

  const int grid_elems = x_points * y_points;
  const int grid_size = sizeof(double) * grid_elems;

  double *x = (double*) malloc (sizeof(double) * x_points);
  double *y = (double*) malloc (sizeof(double) * y_points);
  double *u = (double*) malloc (grid_size);
  double *v = (double*) malloc (grid_size);
  double *u_new = (double*) malloc (grid_size);
  double *v_new = (double*) malloc (grid_size);

  // store device results
  double *du = (double*) malloc (grid_size);
  double *dv = (double*) malloc (grid_size);

  // Define the parameters
  const double nu = 0.01;
  const double sigma = 0.0009;
  const double del_t = sigma * del_x * del_y / nu;      // CFL criteria

  printf("2D Burger's equation\n");
  printf("Grid dimension: x = %d y = %d\n", x_points, y_points);
  printf("Number of time steps: %d\n", num_itrs);

  for(int i = 0; i < x_points; i++) x[i] = i * del_x;
  for(int i = 0; i < y_points; i++) y[i] = i * del_y;

  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      u[i*x_points + j] = 1.0;
      v[i*x_points + j] = 1.0;
      u_new[i*x_points + j] = 1.0;
      v_new[i*x_points + j] = 1.0;

      if(x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0){
        u[i*x_points + j] = 2.0;
        v[i*x_points + j] = 2.0;
        u_new[i*x_points + j] = 2.0;
        v_new[i*x_points + j] = 2.0;
      }
    }
  }

  #pragma omp target teams distribute parallel for collapse(2) num_teams(16) num_threads(16)
  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      int ind = i*x_points + j;
      u_new[ind] = u[ind] + (nu*del_t/(del_x*del_x)) * (u[ind+1] + u[ind-1] - 2 * u[ind]) + 
          (nu*del_t/(del_y*del_y)) * (u[ind+x_points] + u[ind-x_points] - 2 * u[ind]) - 
          (del_t/del_x)*u[ind] * (u[ind] - u[ind-1]) - 
          (del_t/del_y)*v[ind] * (u[ind] - u[ind-x_points]);

      v_new[ind] = v[ind] + (nu*del_t/(del_x*del_x)) * (v[ind+1] + v[ind-1] - 2 * v[ind]) + 
          (nu*del_t/(del_y*del_y)) * (v[ind+x_points] + v[ind-x_points] - 2 * v[ind]) -
          (del_t/del_x)*u[ind] * (v[ind] - v[ind-1]) - 
          (del_t/del_y)*v[ind] * (v[ind] - v[ind-x_points]);
    }
  }

  // Boundary conditions
  #pragma omp target teams distribute parallel for num_teams(16) num_threads(16)
  for(int i = 0; i < y_points; i++){
    u_new[i*x_points] = u_new[i*x_points + 1];
    v_new[i*x_points] = v_new[i*x_points + 1];
    u_new[i*x_points + x_points - 1] = u_new[i*x_points + x_points - 2];
    v_new[i*x_points + x_points - 1] = v_new[i*x_points + x_points - 2];
  }

  #pragma omp target teams distribute parallel for num_teams(16) num_threads(16)
  for(int j = 0; j < x_points; j++){
    u_new[j] = u_new[j + x_points];
    v_new[j] = v_new[j + x_points];
    u_new[(y_points-1)*x_points + j] = u_new[(y_points-2)*x_points + j];
    v_new[(y_points-1)*x_points + j] = v_new[(y_points-2)*x_points + j];
  }

  // Updating older values to newer ones
  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      u[i*x_points + j] = u_new[i*x_points + j];
      v[i*x_points + j] = v_new[i*x_points + j];
    }
  }

  for(int itr = 1; itr < num_itrs; itr++){

    #pragma omp target teams distribute parallel for collapse(2) num_teams(16) num_threads(16)
    for(int i = 0; i < y_points; i++){
      for(int j = 0; j < x_points; j++){
        int ind = i*x_points + j;
        u_new[ind] = u[ind] + (nu*del_t/(del_x*del_x)) * (u[ind+1] + u[ind-1] - 2 * u[ind]) + 
            (nu*del_t/(del_y*del_y)) * (u[ind+x_points] + u[ind-x_points] - 2 * u[ind]) - 
            (del_t/del_x)*u[ind] * (u[ind] - u[ind-1]) - 
            (del_t/del_y)*v[ind] * (u[ind] - u[ind-x_points]);

        v_new[ind] = v[ind] + (nu*del_t/(del_x*del_x)) * (v[ind+1] + v[ind-1] - 2 * v[ind]) + 
            (nu*del_t/(del_y*del_y)) * (v[ind+x_points] + v[ind-x_points] - 2 * v[ind]) -
            (del_t/del_x)*u[ind] * (v[ind] - v[ind-1]) - 
            (del_t/del_y)*v[ind] * (v[ind] - v[ind-x_points]);
      }
    }

    // Boundary conditions
    #pragma omp target teams distribute parallel for num_teams(16) num_threads(16)
    for(int i = 0; i < y_points; i++){
      u_new[i*x_points] = u_new[i*x_points + 1];
      v_new[i*x_points] = v_new[i*x_points + 1];
      u_new[i*x_points + x_points - 1] = u_new[i*x_points + x_points - 2];
      v_new[i*x_points + x_points - 1] = v_new[i*x_points + x_points - 2];
    }

    #pragma omp target teams distribute parallel for num_teams(16) num_threads(16)
    for(int j = 0; j < x_points; j++){
      u_new[j] = u_new[j + x_points];
      v_new[j] = v_new[j + x_points];
      u_new[(y_points-1)*x_points + j] = u_new[(y_points-2)*x_points + j];
      v_new[(y_points-1)*x_points + j] = v_new[(y_points-2)*x_points + j];
    }

    // Updating older values to newer ones
    for(int i = 0; i < y_points; i++){
      for(int j = 0; j < x_points; j++){
        u[i*x_points + j] = u_new[i*x_points + j];
        v[i*x_points + j] = v_new[i*x_points + j];
      }
    }
  }

  printf("Serial computing for verification...\n");

  // Reset velocities
  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      u[i*x_points + j] = 1.0;
      v[i*x_points + j] = 1.0;
      u_new[i*x_points + j] = 1.0;
      v_new[i*x_points + j] = 1.0;

      if(x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0){
        u[i*x_points + j] = 2.0;
        v[i*x_points + j] = 2.0;
        u_new[i*x_points + j] = 2.0;
        v_new[i*x_points + j] = 2.0;
      }
    }
  }

  for(int itr = 0; itr < num_itrs; itr++){

    for(int i = 1; i < y_points-1; i++){
      for(int j = 1; j < x_points-1; j++){
        u_new[i*x_points + j] = u[i*x_points + j] + (nu*del_t/(del_x*del_x)) * (u[i*x_points + j+1] + u[i*x_points + j-1] - 2 * u[i*x_points + j]) + 
          (nu*del_t/(del_y*del_y)) * (u[(i+1)*x_points + j] + u[(i-1)*x_points + j] - 2 * u[i*x_points + j]) - 
          (del_t/del_x)*u[i*x_points + j] * (u[i*x_points + j] - u[i*x_points + j-1]) - 
          (del_t/del_y)*v[i*x_points + j] * (u[i*x_points + j] - u[(i-1)*x_points + j]);

        v_new[i*x_points + j] = v[i*x_points + j] + (nu*del_t/(del_x*del_x)) * (v[i*x_points + j+1] + v[i*x_points + j-1] - 2 * v[i*x_points + j]) + 
          (nu*del_t/(del_y*del_y)) * (v[(i+1)*x_points + j] + v[(i-1)*x_points + j] - 2 * v[i*x_points + j]) -
          (del_t/del_x)*u[i*x_points + j] * (v[i*x_points + j] - v[i*x_points + j-1]) - 
          (del_t/del_y)*v[i*x_points + j] * (v[i*x_points + j] - v[(i-1)*x_points + j]);
      }
    }

    // Boundary conditions
    for(int i = 0; i < x_points; i++){
      u_new[i] = 1.0;
      v_new[i] = 1.0;
      u_new[(y_points-1)*x_points + i] = 1.0;
      v_new[(y_points-1)*x_points + i] = 1.0;
    }

    for(int j = 0; j < y_points; j++){
      u_new[j*x_points] = 1.0;
      v_new[j*x_points] = 1.0;
      u_new[j*x_points + x_points - 1] = 1.0;
      v_new[j*x_points + x_points - 1] = 1.0;
    }

    // Updating older values to newer ones
    for(int i = 0; i < y_points; i++){
      for(int j = 0; j < x_points; j++){
        u[i*x_points + j] = u_new[i*x_points + j];
        v[i*x_points + j] = v_new[i*x_points + j];
      }
    }
  }

  bool ok = true;
  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      if (fabs(du[i*x_points + j] - u[i*x_points + j]) > 1e-6 || 
          fabs(dv[i*x_points + j] - v[i*x_points + j]) > 1e-6) ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(x);
  free(y);
  free(u);
  free(v);
  free(du);
  free(dv);
  free(u_new);
  free(v_new);

  return 0;
}