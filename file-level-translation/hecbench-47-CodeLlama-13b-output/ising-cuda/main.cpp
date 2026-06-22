#include <iostream>
#include <thread>
#ifdef CURAND
#include <curand.h>
#endif
#include <fstream>
#include <cstring>
#include <sstream>

#define TCRIT 2.26918531421f
#define THREADS  128

void init_spins(signed char* lattice,
                const float* __restrict__ randvals,
                const long long nx,
                const long long ny) {
  const long long tid = static_cast<long long>(std::thread::hardware_concurrency()) * std::this_thread::get_id().native_handle();
  if (tid >= nx * ny) return;

  float randval = randvals[tid];
  signed char val = (randval < 0.5f) ? -1 : 1;
  lattice[tid] = val;
}

template<bool is_black>
void update_lattice_ref(signed char* lattice,
                        const signed char* __restrict__ op_lattice,
                        const float* __restrict__ randvals,
                        const float inv_temp,
                        const long long nx,
                        const long long ny) {
  const long long tid = static_cast<long long>(std::thread::hardware_concurrency()) * std::this_thread::get_id().native_handle();
  const int i = tid / ny;
  const int j = tid % ny;

  if (i >= nx || j >= ny) return;

  // Set stencil indices with periodicity
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int inn = (i - 1 < 0) ? nx - 1 : i - 1;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 < 0) ? ny - 1 : j - 1;

  // Select off-column index based on color and row index parity
  int joff;
  if (is_black) {
    joff = (i % 2) ? jpp : jnn;
  } else {
    joff = (i % 2) ? jnn : jpp;
  }

  // Compute sum of nearest neighbor spins
  signed char nn_sum = op_lattice[inn * ny + j] + op_lattice[i * ny + j] + op_lattice[ipp * ny + j] + op_lattice[i * ny + joff];

  // Determine whether to flip spin
  signed char lij = lattice[i * ny + j];
  float acceptance_ratio = expf(-2.0f * inv_temp * nn_sum * lij);
  if (randvals[i * ny + j] < acceptance_ratio) {
    lattice[i * ny + j] = -lij;
  }
}

void update_ref(signed char *lattice_b,
                signed char *lattice_w,
                float* randvals,
#ifdef CURAND
                curandGenerator_t rng, 
#endif
                float inv_temp,
                long long nx,
                long long ny) {

  // Update black
#ifdef CURAND
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
#endif
  update_lattice_ref<true>(lattice_b, lattice_w, randvals, inv_temp, nx/2, ny/2);

  // Update white
#ifdef CURAND
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
#endif
  update_lattice_ref<false>(lattice_w, lattice_b, randvals, inv_temp, nx/2, ny/2);
}

int main(int argc, char **argv) {

  // Defaults
  long long nx = 5120;
  long long ny = 5120;
  float alpha = 0.1f;
  int nwarmup = 100;
  int niters = 1000;
  unsigned long long seed = 1234ULL;

  std::cout << "(*)Warming up the appliciation..." << std::endl;
  init_subsystems();

  float inv_temp = 1.0f / (alpha*TCRIT);

#ifndef CURAND
  srand(seed);
#else
  // for verification across difference platforms, generate random values once
  float* randvals_host = (float*) malloc(nx * ny/2 * sizeof(float));
  for (int i = 0; i < nx * ny/2; i++)
    randvals_host[i] = (float)rand() / (float)RAND_MAX;
#endif

  signed char *lattice_b_r;
  lattice_b_r = (signed char*) malloc(nx * ny/2 * sizeof(signed char));
  init_spins_ref(lattice_b_r, randvals_host, nx, ny/2);

  float *randvals_device;
  CHECK_CUDA(cudaMalloc(&randvals_device, nx * ny/2 * sizeof(*randvals_device)));

  // Setup black and white lattice arrays on device
  signed char *lattice_b, *lattice_w;
  CHECK_CUDA(cudaMalloc(&lattice_b, nx * ny/2 * sizeof(*lattice_b)));
  CHECK_CUDA(cudaMalloc(&lattice_w, nx * ny/2 * sizeof(*lattice_w)));

#ifndef CURAND
  cudaMemcpy(randvals_device, randvals_host, nx*ny/2*sizeof(float), cudaMemcpyHostToDevice);
#endif

#ifdef CURAND
  // Setup cuRAND generator
  curandGenerator_t rng;
  CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
#else
  CHECK_CURAND(curandGenerateUniform(rng, randvals_device, nx*ny/2));
#endif
  init_spins<<<1,128>>>(lattice_b, randvals_device, nx, ny/2);

#ifdef CURAND
  CHECK_CURAND(curandGenerateUniform(rng, randvals_device, nx*ny/2));
#endif
  init_spins<<<1,128>>>(lattice_w, randvals_device, nx, ny/2);

  // Warmup iterations
  std::cout << "(*)Starting warmup..." << std::endl;
  for (int i = 0; i < nwarmup; i++) {
    update_ref(lattice_b, lattice_w, randvals_device, 
#ifdef CURAND
        rng, 
#endif
        inv_temp, nx, ny);
  }    

  std::cout << "(*)Starting trial iterations..." << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    update_ref(lattice_b, lattice_w, randvals_device, 
#ifdef CURAND
        rng, 
#endif
        inv_temp, nx, ny);
  }

  std::cout << "(*)Completion..." << std::endl;

  free(lattice_b_r);
#ifndef CURAND
  free(randvals_host);
#endif

  cudaFree(lattice_b);
  cudaFree(lattice_w);
  cudaFree(randvals_device);

  return 0;
}