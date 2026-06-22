void kernel_adjust_weights(const float* restrict ly, float* restrict w, const float* restrict delta, float* restrict oldw, const int hid) {
  int by = omp_get_thread_num() / omp_get_num_threads();
  int tx = omp_get_thread_num() % omp_get_num_threads();
  int ty = omp_get_thread_num() / omp_get_num_threads();

  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;

  w[index] += (ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]);
  oldw[index] = (ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]);

  omp_barrier();

  if (ty == 0 && by == 0) {
    w[index_x] += (ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]);
    oldw[index_x] = (ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]);
  }
}