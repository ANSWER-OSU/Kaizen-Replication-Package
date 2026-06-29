#include <iostream>
#include <omp.h>
#include <chrono>
#include "utils.hpp"

void winograd_conv2d_omp(const DATA_TYPE *input, const DATA_TYPE *transformed_filter, DATA_TYPE *output, const int offset_i, const int offset_j) {
  int tile_i = offset_i + omp_get_thread_num() / DIM_LOCAL_WORK_GROUP_Y;
  int tile_j = offset_j + omp_get_thread_num() % DIM_LOCAL_WORK_GROUP_Y;

  // input transformation

  DATA_TYPE input_tile[4][4], transformed_tile[4][4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int x = 2 * tile_i + i;
      int y = 2 * tile_j + j;
      if (x >= MAP_SIZE || y >= MAP_SIZE) {
        input_tile[i][j] = 0;
        continue;
      }
      input_tile[i][j] = input[x * MAP_SIZE + y];
    }
  }

  // Bt * d
  DATA_TYPE tmp_tile[4][4];
  for (int j = 0; j < 4; j ++) {
    tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
    tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
    tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
    tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
  }

  // d * B
  for (int i = 0; i < 4; i ++) {
    transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
    transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
    transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
    transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
  }

  // element-wise multiplication

  DATA_TYPE multiplied_tile[4][4];
  for (int i = 0; i < 4; i ++) {
    for (int j = 0; j < 4; j ++) {
      multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
    }
  }

  // output transformation

  DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

  // At * I
  for (int j = 0; j < 4; j ++) {
    tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
    tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
  }

  // I * A
  for (int i = 0; i < 2; i ++) {
    final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
    final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
  }

  for (int i = 0; i < 2; i ++) {
    for (int j = 0; j < 2; j ++) {
      int x = 2 * tile_i + i;
      int y = 2 * tile_j + j;
      if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
        continue;
      }
      output[(x-2) * (MAP_SIZE - 2) + (y-2)] = final_tile[i][j];
    }
  }
}

int main(int argc, char* argv[]) {

  double start = rtclock();

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B_outputFromGpu = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;

  // transformed filter
  WinogradConv2D_2x2_filter_transformation(C);

  DATA_TYPE *d_A;
  cudaMalloc((void**)&d_A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  cudaMemcpy(d_A, A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  DATA_TYPE *d_B;
  cudaMalloc((void**)&d_B, (MAP_SIZE-2)*(MAP_SIZE-2) * sizeof(DATA_TYPE));

  DATA_TYPE *d_C;
  cudaMalloc((void**)&d_C, 16 * sizeof(DATA_TYPE));
  cudaMemcpy(d_C, C, 16 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  // Assuming cpu_offset to be passed as a command line argument
  int cpu_offset;
  if (argc != 2 || 
      !(cpu_offset = atoi(argv[1])) ||
      cpu_offset < 0 ||
      cpu_offset > 100) {
    std::cerr << "Please provide a valid CPU offset as a command-line argument (0-100)." << std::endl;
    return 0;
  }

  // Initial problem size
  int tile_n = (MAP_SIZE - 2 + 1) / 2;
  const int wg_x = (int)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X));
  const int wg_y = DIM_LOCAL_WORK_GROUP_Y;

  // adjust problem size for co-run
  int cpu_global_size_x = cpu_offset * wg_x / 100 * DIM_LOCAL_WORK_GROUP_X;
  int cpu_global_size_y = tile_n / wg_y * wg_y;

  int gpu_global_size_x = tile_n - cpu_global_size_x;
  int gpu_global_size_y = tile_n - cpu_global_size_y;

  int offset_x = cpu_global_size_x;
  int offset_y = 0;

  // openmp parallel for
  #pragma omp parallel num_threads((wg_x-cpu_global_size_x)*wg_y / DIM_LOCAL_WORK_GROUP_Y)
  {
    winograd_conv2d_omp(d_A, d_C, d_B, offset_x, offset_y);
  }

  cudaMemcpy(B_outputFromGpu, d_B, (MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

  DATA_TYPE *host_B = (DATA_TYPE *)malloc((MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE));
  DATA_TYPE *host_C = (DATA_TYPE *)malloc(4 * 4 * sizeof(DATA_TYPE));

  // copy data from d_B to host_B
  #pragma omp parallel for collapse(2) default(none) shared(d_B, host_B, MAP_SIZE) 
  for (int i = 0; i < MAP_SIZE-2; i++) {
    for (int j = 0; j < MAP_SIZE-2; j++) {
      host_B[i * (MAP_SIZE-2) + j] = d_B[i * (MAP_SIZE-2) + j];
    }
  }

  // copy data from d_C to host_C
  #pragma omp parallel for collapse(2) default(none) shared(d_C, host_C) 
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      host_C[i * 4 + j] = d_C[i * 4 + j];
    }
  }

  bool pass = true;

  #pragma omp parallel num_threads((wg_x) * (wg_y) / DIM_LOCAL_WORK_GROUP_Y)
  {
    int x = omp_get_thread_num() / wg_y;
    int y = omp_get_thread_num() % wg_y;

    if ((cpu_global_size_x <= x) && (x < gpu_global_size_x)) {
      if ((cpu_global_size_y <= y) && (y < gpu_global_size_y)) {
        WinogradConv2D_2x2_omp(host_A + y * (wg_x+1) * MAP_SIZE + x * MAP_SIZE,
                      host_B + y * (wg_x+1) * (MAP_SIZE-2) + x * (MAP_SIZE-2),
                      host_C, wg_x+1, MAP_SIZE-2);
      }
      else {
        WinogradConv2D_2x2_omp(host_A + y * wg_x * MAP_SIZE,
                      host_B + y * wg_x * (MAP_SIZE-2),
                      host_C, wg_x, MAP_SIZE-2);
      }
    }
  }

  printf("%s\n", pass ? "PASS" : "FAIL");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(B_outputFromGpu);
  free(C);

  free(host_B);
  free(host_C);

  double end = rtclock();
  printf("Total time: %lf s\n", end - start);

  return 0;
}