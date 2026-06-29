#include <chrono>
#include <omp.h>
#include "utils.h"

__global__ void winograd_conv2d(const int numTeams, const int numThreads, const DATA_TYPE *__restrict__ input, const DATA_TYPE *__restrict__ transformed_filter, DATA_TYPE *__restrict__ output, const int offset_i, const int offset_j) {
  int tile_i = blockIdx.x * blockDim.x + threadIdx.x + offset_i;
  int tile_j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

  // input transformation

  DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
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
  for (int j = 0; j < 4; j++) {
    tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
    tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
    tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
    tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
  }
  // d * B
  for (int i = 0; i < 4; i++) {
    transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
    transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
    transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
    transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
  }

  // element-wise multiplication

  DATA_TYPE multiplied_tile[4][4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
    }
  }

  // output transformation

  DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

  // At * I
  for (int j = 0; j < 4; j++) {
    tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
    tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
  }
  // I * A
  for (int i = 0; i < 2; i++) {
    final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
    final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      int x = 2 * tile_i + i;
      int y = 2 * tile_j + j;
      if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
        continue;
      }
      output[x * (MAP_SIZE - 2) + y] = final_tile[i][j];
    }
  }
}

int main(int argc, char* argv[]) {
  double start = rtclock();

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B_outputFromGpu = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i) {
    for (int j = 0; j < MAP_SIZE; ++j) {
      A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;
    }
  }

  // transformed filter
  WinogradConv2D_2x2_filter_transformation(C);

  DATA_TYPE *d_A;
  cudaMalloc((void**)&d_A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  cudaMemcpy(d_A, A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  DATA_TYPE *d_B;
  cudaMalloc((void**)&d_B, (MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));

  DATA_TYPE *d_C;
  cudaMalloc((void**)&d_C, 16 * sizeof(DATA_TYPE));
  cudaMemcpy(d_C, C, 16 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  const int tile_n = (MAP_SIZE - 2 + 1) / 2;

  size_t global_work_size[2];
  size_t local_work_size[2];
  int offset[2];

  bool success = true;
  double t_total = 0.0;
  int numTeams = 0, numThreads = 0;
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int tile_i = 0; tile_i < tile_n; tile_i++) {
    for (int tile_j = 0; tile_j < tile_n; tile_j++) {
      global_work_size[0] = (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
      global_work_size[1] = (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

      local_work_size[0] = DIM_LOCAL_WORK_GROUP_X;
      local_work_size[1] = DIM_LOCAL_WORK_GROUP_Y;

      offset[0] = global_work_size[0] * tile_i;
      offset[1] = global_work_size[1] * tile_j;

      dim3 grid(global_work_size[0] / local_work_size[0], global_work_size[1] / local_work_size[1]);
      dim3 block(local_work_size[0], local_work_size[1]);

      bool run_on_gpu = (global_work_size[0] > 0);
      bool run_on_cpu = (global_work_size[1] > 0);

      if (run_on_gpu) {
        winograd_conv2d<<<grid, block>>>(d_A, d_C, d_B, offset[0], offset[1]);
      }

      if (run_on_cpu) {
        WinogradConv2D_2x2_omp(A, B, C, global_work_size);

        cudaMemcpy(d_B, B, run_on_gpu ? offset[0] * 2 * (MAP_SIZE - 2) * sizeof(DATA_TYPE) : (MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
      }

      cudaMemcpy(B_outputFromGpu, d_B, (MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

      t_total += rtclock() - t_total;

#ifdef VERBOSE
      if (run_on_cpu) {
        printf("run on host\n");
      }
      if (run_on_gpu) {
        printf("run on device\n");
      }
      printf("tile %d %d\n", tile_i, tile_j);
#endif

      WinogradConv2D_2x2(A, B, C);
      success &= compareResults(B, B_outputFromGpu);
    }
  }

  printf("%s\n", success ? "PASS" : "FAIL");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(B_outputFromGpu);
  free(C);

  double end = rtclock();
  printf("Total time: %f s\n", end - start);
  printf("Number of teams: %d\n", numTeams);
  printf("Number of threads: %d\n", numThreads);
  printf("Team size: %d %d\n", global_work_size[0], global_work_size[1]);
  printf("Work size: %d %d\n", local_work_size[0], local_work_size[1]);
  printf("Offset: %d %d\n", offset[0], offset[1]);
  printf("Total Time: %f s\n", t_total);
  printf("Ratio of C execution to total: %.2f%%\n", 100.0f * t_total / (end - start));

  return 0;
}