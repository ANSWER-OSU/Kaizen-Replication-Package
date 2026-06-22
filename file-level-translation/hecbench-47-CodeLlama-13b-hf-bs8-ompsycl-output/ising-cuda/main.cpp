void update(SignedChar *lattice_b, 
    SignedChar *lattice_w, 
    Float* randvals_host, 
    int number_of_threads, 
    int number_of_blocks, 
    float inv_temp, 
    long long nx, 
    long long ny) {

  // Setup CUDA launch configuration

  // Update black
  init_spins_ref(lattice_b, randvals_host, nx, ny/2);
  update_lattice_ref(lattice_b, lattice_w, randvals_host, inv_temp, nx, ny/2);

  // Update white
  init_spins_ref(lattice_w, randvals_host, nx, ny/2);
  update_lattice_ref(lattice_w, lattice_b, randvals_host, inv_temp, nx, ny/2);
}