#include <omp.h>

void init_spins(signed char* lattice, const float* randvals, const long long nx, const long long ny) {
  #pragma omp parallel for
  for (long long tid = 0; tid < nx * ny; tid++) {
    float randval = randvals[tid];
    signed char val = (randval < 0.5f) ? -1 : 1;
    lattice[tid] = val;
  }
}

template<bool is_black>
void update_lattice(signed char* lattice, const signed char* op_lattice, const float* randvals, const float inv_temp, const long long nx, const long long ny) {
  #pragma omp parallel for
  for (long long tid = 0; tid < nx * ny; tid++) {
    int i = tid / ny;
    int j = tid % ny;

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
    signed char nn_sum = op_lattice[inn * ny + j] + op_lattice[i * ny + j] + op_lattice[ipp * ny + j] + op_lattice[i * ny + joff];

    // Determine whether to flip spin
    signed char lij = lattice[i * ny + j];
    float acceptance_ratio = expf(-2.0f * inv_temp * nn_sum * lij);
    if (randvals[i*ny + j] < acceptance_ratio) {
      lattice[i * ny + j] = -lij;
    }
  }
}

void update(signed char *lattice_b,
    signed char *lattice_w,
    float* randvals,
    float inv_temp,
    long long nx,
    long long ny) {
  update_lattice<true>(lattice_b, lattice_w, randvals, inv_temp, nx, ny/2);
  update_lattice<false>(lattice_w, lattice_b, randvals, inv_temp, nx, ny/2);
}