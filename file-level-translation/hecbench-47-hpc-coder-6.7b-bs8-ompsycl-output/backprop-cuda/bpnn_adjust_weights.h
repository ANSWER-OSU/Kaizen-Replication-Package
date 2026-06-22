void kernel_adjust_weights (
  const int numTeams, 
  const int numThreads, 
  const float*__restrict__ ly, 
       float *__restrict__ w, 
  const float*__restrict__ delta, 
        float*__restrict__ oldw, 
  const int hid)
{
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int by = 0; by < numTeams; by++) {
    for (int ty = 0; ty < BLOCK_HEIGHT; ty++) {
      for (int tx = 0; tx < BLOCK_WIDTH; tx++) {
        int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
        int index_y = HEIGHT * by + ty + 1;
        int index_x = tx + 1;

        w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
        oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      }
    }
  }

  #pragma omp target teams distribute parallel for num_teams(1) num_threads(1)
  for (int i = 0; i < 1; i++) {
    w[i+1] += ((ETA * delta[i+1]) + (MOMENTUM * oldw[i+1]));
    oldw[i+1] = ((ETA * delta[i+1]) + (MOMENTUM * oldw[i+1]));
  }
}