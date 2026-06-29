#include <omp.h>
#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <cstring>
#include "reference.h"

#define TCRIT 2.26918531421f
#define THREADS 128

// Initialize lattice spins
void init_spins(signed char* lattice,
    const float* __restrict__ randvals,
    const long long nx,
    const long long ny) {
  #pragma omp parallel for default(none) shared(lattice, randvals, nx, ny)
  for (long long tid = 0; tid < nx * ny; tid += THREADS) {
    if (tid >= nx * ny) continue;

    float randval = randvals[tid];
    signed char val = (randval < 0.5f) ? -1 : 1;
    lattice[tid] = val;
  }
}

template<bool is_black>
void update_lattice(signed char* lattice,
    const signed char* __restrict__ op_lattice,
    const float* __restrict__ randvals,
    const float inv_temp,
    const long long nx,
    const long long ny) {
  #pragma omp parallel for default(none) shared(lattice, op_lattice, randvals, inv_temp, nx, ny) schedule (static)
  for (long long tid = 0; tid < nx * ny/2; tid += THREADS) {
    const int i = tid / ny;
    const int j = tid % ny;

    if (i >= nx || j >= ny) continue;

    // Set stencil indices with periodicity
    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    // Select off-column index based on color and row index parity
    int joff;
    if (is_black) {
      joff = (i % 2 == 0) ? jpp : jnn;
    } else {
      joff = (i % 2 == 0) ? jnn : jpp;
    }

    // Compute sum of nearest neighbor spins
    signed char nn_sum = op_lattice[inn * ny + j] + op_lattice[i * ny + j] + op_lattice[ipp * ny + j] + op_lattice[i * ny + joff];

    // Determine whether to flip spin
    signed char lij = lattice[i * ny + j];
    float acceptance_ratio = expf(-2.0f * inv_temp * nn_sum * lij);
    if (randvals[tid] < acceptance_ratio) {
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
  int updates = nx * ny/2;

  // Update black
  update_lattice<true>(lattice_b, lattice_w, randvals, inv_temp, nx, ny/2);

  // Update white
  update_lattice<false>(lattice_w, lattice_b, randvals, inv_temp, nx, ny/2);
}

static void usage(const char *pname) {
  const char *bname = rindex(pname, '/');
  if (!bname) {bname = pname;}
  else        {bname++;}

  fprintf(stdout,
          "Usage: %s [options]\n"
          "options:\n"
          "\t-x|--lattice-n <LATTICE_N>\n"
          "\t\tnumber of lattice rows\n"
          "\n"
          "\t-y|--lattice_m <LATTICE_M>\n"
          "\t\tnumber of lattice columns\n"
          "\n"
          "\t-w|--nwarmup <NWARMUP>\n"
          "\t\tnumber of warmup iterations\n"
          "\n"
          "\t-n|--niters <NITERS>\n"
          "\t\tnumber of trial iterations\n"
          "\n"
          "\t-a|--alpha <ALPHA>\n"
          "\t\tcoefficient of critical temperature\n"
          "\n"
          "\t-s|--seed <SEED>\n"
          "\t\tseed for random number generation\n\n",
          bname);
  exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {
  // Defaults
  long long nx = 5120;
  long long ny = 5120;
  float alpha = 0.1f;
  int nwarmup = 100;
  int niters = 1000;
  unsigned long long seed = 1234ULL;

  while (1) {
    static struct option long_options[] = {
        {     "lattice-n", required_argument, 0, 'x'},
        {     "lattice-m", required_argument, 0, 'y'},
        {         "alpha", required_argument, 0, 'y'},
        {          "seed", required_argument, 0, 's'},
        {       "nwarmup", required_argument, 0, 'w'},
        {        "niters", required_argument, 0, 'n'},
        { "write-lattice",       no_argument, 0, 'o'},
        {          "help",       no_argument, 0, 'h'},
        {               0,                 0, 0,   0}
    };

    int option_index = 0;
    int ch = getopt_long(argc, argv, "x:y:a:s:w:n:h", long_options, &option_index);
    if (ch == -1) break;

    switch(ch) {
      case 0:
        break;
      case 'x':
        nx = atoll(optarg); break;
      case 'y':
        ny = atoll(optarg); break;
      case 'a':
        alpha = atof(optarg); break;
      case 's':
        seed = atoll(optarg); break;
      case 'w':
        nwarmup = atoi(optarg); break;
      case 'n':
        niters = atoi(optarg); break;
      case 'h':
        usage(argv[0]); break;
      case '?':
        exit(EXIT_FAILURE);
      default:
        fprintf(stderr, "unknown option: %c\n", ch);
        exit(EXIT_FAILURE);
    }
  }

  // Check arguments
  if (nx % 2 != 0 || ny % 2 != 0) {
    fprintf(stderr, "ERROR: Lattice dimensions must be even values.\n");
    exit(EXIT_FAILURE);
  }

  float inv_temp = 1.0f / (alpha*TCRIT);

  // for verification across difference platforms, generate random values once
  srand(seed);
  float* randvals_host = (float*) malloc(nx * ny/2 * sizeof(float));
  for (int i = 0; i < nx * ny/2; i++)
    randvals_host[i] = (float)rand() / (float)RAND_MAX;

  signed char *lattice_b_r, *lattice_w_r;
  lattice_b_r = (signed char*) malloc(nx * ny/2 * sizeof(signed char));
  lattice_w_r = (signed char*) malloc(nx * ny/2 * sizeof(signed char));

  // Setup black and white lattice arrays on host
  signed char *lattice_b, *lattice_w;
  lattice_b = (signed char*) malloc(nx * ny/2 * sizeof(signed char));
  lattice_w = (signed char*) malloc(nx * ny/2 * sizeof(signed char));

  // Warmup iterations
  printf("Starting warmup...\n");
  for (int i = 0; i < nwarmup; i++) {
    update(lattice_b, lattice_w, randvals_host, inv_temp, nx, ny);
  }

  printf("Starting trial iterations...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    update(lattice_b, lattice_w, randvals_host, inv_temp, nx, ny);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();

  printf("REPORT:\n");
  printf("\tnGPUs: %d\n", 1);
  printf("\ttemperature: %f * %f\n", alpha, TCRIT);
  printf("\tseed: %llu\n", seed);
  printf("\twarmup iterations: %d\n", nwarmup);
  printf("\ttrial iterations: %d\n", niters);
  printf("\tlattice dimensions: %lld x %lld\n", nx, ny);
  printf("\telapsed time: %f sec\n", duration * 1e-6);
  printf("\tupdates per ns: %f\n", (double) (nx * ny) * niters / duration * 1e-3);

  printf("Starting verification iterations ...\n");
  init_spins_ref(lattice_b_r, randvals_host, nx, ny/2);
  init_spins_ref(lattice_w_r, randvals_host, nx, ny/2);
  for (int i = 0; i < nwarmup + niters; i++) {
    update_ref(lattice_b_r, lattice_w_r, randvals_host, inv_temp, nx, ny);
  }

  bool ok = true;
  for (int i = 0; i < nx*ny/2; i++) {
    ok  = (lattice_b[i] == lattice_b_r[i]) && 
          (lattice_w[i] == lattice_w_r[i]);
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(randvals_host);
  free(lattice_b_r);
  free(lattice_w_r);

  free(lattice_b);
  free(lattice_w);

  return 0;
}