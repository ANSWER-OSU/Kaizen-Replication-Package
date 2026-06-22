void WinogradConv2D_2x2_omp(const DATA_TYPE *__restrict__ input, 
    DATA_TYPE *__restrict__ output, 
    const DATA_TYPE *__restrict__ filter, 
    const size_t numTeams, 
    const size_t numThreads) { 
  #pragma omp target teams num_teams(numTeams)
  {
    #pragma omp parallel num_threads(numThreads)
    {
      int tile_i = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();
      int tile_j = tile_i;

      // input transformation

      DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
      for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j ++) { 
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
          multiplied_tile[i][j] = transformed_tile[i][j] * filter[i * 4 + j];
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
          output[x * (MAP_SIZE - 2) + y] = final_tile[i][j];
        }
      }
    }
  }
}

__global__ void winograd_conv2d(
    const DATA_TYPE *__restrict__ input,
    const DATA_TYPE *__restrict__ transformed_filter ,
    DATA_TYPE *__restrict__ output,
    const int offset_i,
    const int offset_j)
{
  int tile_i = blockIdx.x * blockDim.x + threadIdx.x;
  int tile_j = blockIdx.y * blockDim.y + threadIdx.y;

  WinogradConv2D_2x2(input, output, transformed_filter, tile_i, tile_j);

}

void WinogradConv2D_2x2_omp(const DATA_TYPE *__restrict__ input, 
    DATA_TYPE *__restrict__ output, 
    const DATA_TYPE *__restrict__ filter, 
    const size_t input_tile_size) { 
  #pragma omp target teams num_teams(input_tile_size)
  {
    #pragma omp parallel num_threads(input_tile_size)
    {
      int tile_i = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();
      int tile_j = tile_i;
      WinogradConv2D_2x2(input, output, filter, tile_i, tile_j);
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
  cudaMalloc((void**)&d_B, (MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE));

  DATA_TYPE *d_C;
  cudaMalloc((void**)&d_C, 16 * sizeof(DATA_TYPE));
  cudaMemcpy(d_C, C, 16 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  size_t input_tile_size = (MAP_SIZE - 2 + 1) / 2;

  // adjust problem size for co-run
  size_t cpu_global_size[2];
  size_t gpu_global_size[2];
  int global_offset[2];

  bool pass = true;

  // sweep over cpu_offset 
  double co_time = 0.0;

  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {
    cpu_global_size[0] = cpu_offset * input_tile_size / 100;
    cpu_global_size[1] = input_tile_size - cpu_global_size[0];

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    gpu_global_size[0] = input_tile_size - cpu_global_size[0];
    gpu_global_size[1] = input_tile_size - cpu_global_size[1];

    dim3 grid((gpu_global_size[0] + 1) / 2, (gpu_global_size[1] + 1) / 2);
    dim3 block(DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y);

    bool cpu_run = false, gpu_run = false;
    if (cpu_global_size[0] > 0) {
      cpu_run = true;
    }
    if (gpu_global_size[0] > 0) {
      gpu_run = true;
    }

    // co-execution of host and device
    double co_start = rtclock();

    if (gpu_run) {
      winograd_conv2d<<<grid, block>>>(d_A, d_C, d_B, global_offset[0], global_offset[1]);
    }

    if (cpu_run) {
      WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);

      cudaMemcpy(d_B, B, gpu_run ? global_offset[0]*2*(MAP_SIZE-2)*sizeof(DATA_TYPE) : 
          (MAP_SIZE-2)*(MAP_SIZE-2)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(B_outputFromGpu, d_B, (MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    co_time += rtclock() - co_start;

    WinogradConv2D_2x2(A, B, C);
    pass &= compareResults(B, B_outputFromGpu);

  } // sweep

  printf("%s\n", pass ? "PASS" : "FAIL");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(B_outputFromGpu);
  free(C);

  double end = rtclock();
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (end - start));

  return 0;
}