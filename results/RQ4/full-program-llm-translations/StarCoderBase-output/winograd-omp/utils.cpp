void WinogradConv2D_2x2_openmp(DATA_TYPE* input, DATA_TYPE* output, DATA_TYPE* transformed_filter, size_t* cpu_global_size) {
  int out_map_size = MAP_SIZE - 2;
  int tile_n = (out_map_size + 1) / 2;

  for (int tile_i = 0; tile_i < cpu_global_size[0]; tile_i++) {
    #pragma omp parallel for
    for (int tile_j = 0; tile_j < tile_n; tile_j++) {
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
      for (int j = 0; j < 4; j++) {
        tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
        tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
        tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
        tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
      }
      for (int i = 0; i < 4; i++) {
        transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
        transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
      }
      DATA_TYPE multiplied_tile[4][4];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
        }
      }
      DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];
      for (int j = 0; j < 4; j++) {
        tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
        tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
      }
      for (int i = 0; i < 2; i++) {
        final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
        final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
      }
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= out_map_size || y >= out_map_size) {
            continue;
          }
          output[x * out_map_size + y] = final_tile[i][j];
        }
      }
    }
  }
}