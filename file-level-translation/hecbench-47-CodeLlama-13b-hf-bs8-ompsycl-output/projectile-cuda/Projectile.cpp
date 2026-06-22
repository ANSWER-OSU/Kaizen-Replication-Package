void GpuParallel(std::vector<Projectile>& in_vect, std::vector<Projectile>& out_vect, const int repeat) {
  Projectile *bufin_vect, *bufout_vect;

  cudaMalloc((void**)&bufin_vect, sizeof(Projectile) * num_elements);
  cudaMalloc((void**)&bufout_vect, sizeof(Projectile) * num_elements);
  cudaMemcpy(bufin_vect, in_vect.data(), sizeof(Projectile) * num_elements, cudaMemcpyHostToDevice);

  dim3 grids ((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 blocks (BLOCK_SIZE);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

#pragma omp target parallel for num_teams(grids) num_threads(blocks)
  for (int i = 0; i < repeat; i++)
    CalculateRange (bufin_vect, bufout_vect);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(out_vect.data(), bufout_vect, sizeof(Projectile) * num_elements, cudaMemcpyDeviceToHost);
  cudaFree(bufin_vect);
  cudaFree(bufout_vect);
}