void kernel_adjust_weights (
  const int numTeams,
  const int numThreads,
  const float*__restrict__ ly, 
       float *__restrict__ w, 
  const float*__restrict__ delta, 
        float*__restrict__ oldw, 
  const int hid)
{
  #pragma omp target teams num_teams(numTeams)
  {
    #pragma omp parallel num_threads(numThreads)
    {
      int by = omp_get_team_num(); 
      int tx = omp_get_thread_num() % BLOCK_SIZE; 
      int ty = omp_get_thread_num() / BLOCK_SIZE;

      int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
      int index_y = HEIGHT * by + ty + 1;
      int index_x = tx + 1;

      #pragma omp atomic
      w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

      #pragma omp barrier

      if (ty == 0 && by ==0){
        #pragma omp atomic
        w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
      }
    }
  }
}