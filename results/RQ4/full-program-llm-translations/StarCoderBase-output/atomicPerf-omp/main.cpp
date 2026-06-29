void block_range_atomic_on_global_mem(void *data, int n) {
  int tid = omp_get_thread_num();
  #pragma omp for
  for (int i = tid; i < n; i += omp_get_num_threads()) {
    atomic_fetch_add(static_cast<int*>(data) + tid, 1);
  }
}

void warp_range_atomic_on_global_mem(void *data, int n) {
  int tid = omp_get_thread_num();
  #pragma omp for
  for (int i = tid; i < n; i += omp_get_num_threads()) {
    atomic_fetch_add(static_cast<int*>(data) + (i & 0x1F), 1);
  }
}

void single_range_atomic_on_global_mem(void *data, int offset, int n) {
  int tid = omp_get_thread_num();
  #pragma omp for
  for (int i = tid; i < n; i += omp_get_num_threads()) {
    atomic_fetch_add(static_cast<int*>(data) + offset, 1);
  }
}

__managed__ int smem_data[BLOCK_SIZE];

void block_range_atomic_on_shared_mem(void *data, int n) {
  int tid = omp_get_thread_num();
  #pragma omp for
  for (int i = tid; i < n; i += omp_get_num_threads()) {
    atomic_fetch_add(&smem_data[tid], 1);
  }
  if (omp_get_thread_num() == 0) {
    static_cast<int*>(data)[tid] = smem_data[tid];
  }
}

void warp_range_atomic_on_shared_mem(void *data, int n) {
  int tid = omp_get_thread_num();
  #pragma omp for
  for (int i = tid; i < n; i += omp_get_num_threads()) {
    atomic_fetch_add(&smem_data[i & 0x1F], 1);
  }
  if (omp_get_thread_num() == 0) {
    static_cast<int*>(data)[tid] = smem_data[tid];
  }
}

void single_range_atomic_on_shared_mem(void *data, int offset, int n) {
  int tid = omp_get_thread_num();
  #pragma omp for
  for (int i = tid; i < n; i += omp_get_num_threads()) {
    atomic_fetch_add(&smem_data[offset], 1);
  }
  if (omp_get_thread_num() == 0) {
    static_cast<int*>(data)[tid] = smem_data[tid];
  }
}

void atomic_perf(int n, int t, int repeat) {
  int data_size = sizeof(int) * t;

  int *data = new int[data_size];
  int *h_data = new int[data_size];
  int *r_data = new int[data_size];
  int fail;

  for (int i = 0; i < t; i++) {
    data[i] = i % 1024 + 1;
  }

  void *d_data;
  d_data = malloc(data_size);

  #pragma omp parallel num_threads(n / BLOCK_SIZE)
  {
    CHECK_ERROR(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      block_range_atomic_on_global_mem(d_data, n);
    }
    CHECK_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (omp_get_thread_num() == 0) {
      printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n", time * 1e-3f / repeat);
    }

    CHECK_ERROR(cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost));
    memcpy(r_data, data, data_size);
    for (int i = 0; i < repeat; i++) {
      block_range_atomic_on_global_mem_ref(r_data, n);
    }
    fail = memcmp(h_data, r_data, data_size);
    if (omp_get_thread_num() == 0) {
      printf("%s\n", fail ? "FAIL" : "PASS");
    }
    
    CHECK_ERROR(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      warp_range_atomic_on_global_mem(d_data, n);
    }
    CHECK_ERROR(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (omp_get_thread_num() == 0) {
      printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n", time * 1e-3f / repeat);
    }

    CHECK_ERROR(cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost));
    memcpy(r_data, data, data_size);
    for (int i = 0; i < repeat; i++) {
      warp_range_atomic_on_global_mem_ref(r_data, n);
    }
    fail = memcmp(h_data, r_data, data_size);
    if (omp_get_thread_num() == 0) {
      printf("%s\n", fail ? "FAIL" : "PASS");
    }

    CHECK_ERROR(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      single_range_atomic_on_global_mem(d_data, i % BLOCK_SIZE, n);
    }
    CHECK_ERROR(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (omp_get_thread_num() == 0) {
      printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n", time * 1e-3f / repeat);
    }

    CHECK_ERROR(cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost));
    memcpy(r_data, data, data_size);
    for (int i = 0; i < repeat; i++) {
      single_range_atomic_on_global_mem_ref(r_data, i % BLOCK_SIZE, n);
    }
    fail = memcmp(h_data, r_data, data_size);
    if (omp_get_thread_num() == 0) {
      printf("%s\n", fail ? "FAIL" : "PASS");
    }
  }

  CHECK_ERROR(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaDeviceSynchronize());
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    block_range_atomic_on_shared_mem(d_data, n);
  }
  CHECK_ERROR(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (omp_get_thread_num() == 0) {
    printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n", time * 1e-3f / repeat);
  }

  CHECK_ERROR(cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost));
  fail = memcmp(h_data, data, data_size);
  if (omp_get_thread_num() == 0) {
    printf("%s\n", fail ? "FAIL" : "PASS");
  }

  CHECK_ERROR(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaDeviceSynchronize());
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    warp_range_atomic_on_shared_mem(d_data, n);
  }
  CHECK_ERROR(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (omp_get_thread_num() == 0) {
    printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n", time * 1e-3f / repeat);
  }

  CHECK_ERROR(cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost));
  fail = memcmp(h_data, data, data_size);
  if (omp_get_thread_num() == 0) {
    printf("%s\n", fail ? "FAIL" : "PASS");
  }

  CHECK_ERROR(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaDeviceSynchronize());
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    single_range_atomic_on_shared_mem(d_data, i % BLOCK_SIZE, n);
  }
  CHECK_ERROR(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (omp_get_thread_num() == 0) {
    printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n", time * 1e-3f / repeat);
  }

  CHECK_ERROR(cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost));
  fail = memcmp(h_data, data, data_size);
  if (omp_get_thread_num() == 0) {
    printf("%s\n", fail ? "FAIL" : "PASS");
  }

  delete[] data;
  delete[] h_data;
  delete[] r_data;
  cudaFree(d_data);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int n = 3 * 4 * 7 * 8 * 9 * 256; // number of threads
  const int len = 1024; // data array length
  atomic_perf(n, len, repeat);
  return 0;
}