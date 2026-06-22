#include <omp.h>

#pragma omp declare simd
void kernel_adjust_weights_cpu(
  const float* ly, 
  float* w, 
  const float* delta, 
  float* oldw, 
  int hid)
{
  int by = omp_get_thread_num() / GRID_SIZE;
  int tx = omp_get_thread_num() % GRID_SIZE;
  int ty = omp_get_thread_num() / GRID_SIZE;

  int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;

  w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
  oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

  #pragma omp barrier

  if (by == 0 && ty == 0) {
    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
  }
}