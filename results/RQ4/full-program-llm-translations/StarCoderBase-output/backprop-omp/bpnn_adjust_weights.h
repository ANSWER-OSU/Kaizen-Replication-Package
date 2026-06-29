void kernel_adjust_weights (
  const float*__restrict__ ly, 
       float *__restrict__ w, 
  const float*__restrict__ delta, 
        float*__restrict__ oldw, 
  const int hid)
{

#pragma omp target teams distribute parallel for collapse(2)
  for (int by = 0; by < num_blocks_y; by++) {
    for (int ty = 0; ty < block_size_y; ty++) {
      for (int tx = 0; tx < block_size_x; tx++) {
        int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
        int index_y = HEIGHT * by + ty + 1;
        int index_x = tx + 1;

        w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
        oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      }
    }
  }

#pragma omp target teams distribute parallel for
  for (int tx = 0; tx < block_size_x; tx++) {
    int index_x = tx + 1;
    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
  }
}