template <bool is_black>
void update_lattice(const int numTeams, const int numThreads, signed char* lattice, const signed char* __restrict__ op_lattice, const float* __restrict__ randvals, const float inv_temp, const long long nx, const long long ny) {
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      // Set stencil indices with periodicity
      int ipp = (i + 1 < nx) ? i + 1 : 0;
      int inn = (i - 1 >= 0) ? i - 1: nx - 1;
      int jpp = (j + 1 < ny) ? j + 1 : 0;
      int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

      // Select off-column index based on color and row index parity
      int joff;
      if (is_black) {
        joff = (i % 2) ? jpp : jnn;
      } else {
        joff = (i % 2) ? jnn : jpp;
      }

      // Compute sum of nearest neighbor spins
      signed char nn_sum = op_lattice[inn*ny + j] + op_lattice[i*ny + j] + op_lattice[ipp*ny + j] + op_lattice[i*ny + joff];

      // Determine whether to flip spin
      signed char lij = lattice[i*ny + j];
      float acceptance_ratio = expf(-2.0f * inv_temp * nn_sum * lij);
      if (randvals[i*ny + j] < acceptance_ratio) {
        lattice[i*ny + j] = -lij;
      }
    }
  }
}

void update(signed char *lattice_b, 
    signed char *lattice_w, 
    float* randvals, 
#ifdef CURAND
    curandGenerator_t rng, 
#endif
    float inv_temp, 
    long long nx, 
    long long ny) {

  // Setup CUDA launch configuration
  int numTeams = (nx * ny/2 + get_num_threads() - 1) / get_num_threads();
  int numThreads = get_num_threads();

  // Update black
#ifdef CURAND
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
#endif
  update_lattice<true>(numTeams, numThreads, lattice_b, lattice_w, randvals, inv_temp, nx, ny/2);

  // Update white
#ifdef CURAND
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
#endif
  update_lattice<false>(numTeams, numThreads, lattice_w, lattice_b, randvals, inv_temp, nx, ny/2);
}