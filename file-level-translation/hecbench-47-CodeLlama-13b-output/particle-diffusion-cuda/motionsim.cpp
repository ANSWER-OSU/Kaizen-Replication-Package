#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

// Helper functions
void usage(std::string programName, int argc, char* argv[]) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " <Number of iterations within the kernel> ";
  std::cout << "<Kernel execution count>\n\n";
}

template <typename T>
void print_matrix(T** matrix, size_t size_X, size_t size_Y) {
  std::cout << std::endl;
  for (size_t i = 0; i < size_X; ++i) {
    for (size_t j = 0; j < size_Y; ++j) {
      std::cout << std::setw(3) << matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void print_vector(T* vector, size_t n) {
  std::cout << std::endl;
  for (size_t i = 0; i < n; ++i) {
    std::cout << vector[i] << " ";
  }
  std::cout << std::endl;
}

// OpenMP kernel function
void Simulation(float* a_particleX, float* a_particleY,
                const float* a_randomX, const float* ab_randomY,
                size_t* a_map, size_t n_particles, unsigned int nIterations,
                int grid_size, float radius, int prob, int nThreads) {
  unsigned int num_threads = omp_get_num_threads();
  unsigned int thread_id = omp_get_thread_num();
  int chunk_size = n_particles / num_threads;
  int offset = thread_id * chunk_size;

  float *local_particleX = a_particleX + offset;
  float *local_particleY = a_particleY + offset;
  float *local_randomX = a_randomX + thread_id * nIterations * n_particles;
  float *local_randomY = ab_randomY + thread_id * nIterations * n_particles;
  float *local_map = a_map + thread_id * n_particles * grid_size * grid_size;

  // Start iterations
  // Each iteration:
  //  1. Updates the position of all water molecules
  //  2. Checks if water molecule is inside a cell or not.
  //  3. Updates counter in cells array
  size_t iter = 0;
  float pX = 0.0f;
  float pY = 0.0f;

#pragma omp for schedule(static)
  for (int ii = 0; ii < n_particles; ii++) {
    pX = local_particleX[ii];
    pY = local_particleY[ii];
    int map_base = ii * grid_size * grid_size;
    while (iter < nIterations) {
      // Computes random displacement for each molecule
      // This example shows random distances between
      // -0.05 units and 0.05 units in both X and Y directions
      // Moves each water molecule by a random vector
      float randnumX = local_randomX[iter * n_particles + ii];
      float randnumY = local_randomY[iter * n_particles + ii];
      // Transform the scaled random numbers into small displacements
      float displacementX = randnumX / 1000.0f - 0.0495f;
      float displacementY = randnumY / 1000.0f - 0.0495f;

      // Move particles using random displacements
      pX += displacementX;
      pY += displacementY;

      // Compute distances from particle position to grid point
      float dX = pX - truncf(pX);
      float dY = pY - truncf(pY);

      // Compute grid point indices
      int iX = floorf(pX);
      int iY = floorf(pY);

      // Check if particle is still in computation grid
      if ((pX < grid_size) && (pY < grid_size) && (pX >= 0) && (pY >= 0)) {
        // Check if particle is (or remained) inside cell.
        // Increment cell counter in map array if so

        if ((dX * dX + dY * dY <= radius * radius) && (rand() % 100 < prob))
          // The map array is organized as (particle, y, x)
          local_map[map_base + iY * grid_size + iX]++;
      }
      iter++;
    }  // Next iteration

    local_particleX[ii] = pX;
    local_particleY[ii] = pY;
  }

#pragma omp barrier

  // Merge the local map into one array
  for (size_t i = 0; i < n_particles; ++i) {
    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        if (local_map[i * grid_size * grid_size + y * grid_size + x] > 0) {
          a_map[i * grid_size * grid_size + y * grid_size + x] +=
              local_map[i * grid_size * grid_size + y * grid_size + x];
        }
      }
    }
  }
}

// OpenMP host function
void motion_device(float* particleX, float* particleY,
                   float* randomX, float* randomY, int** grid, size_t grid_size,
                   size_t n_particles, int nIterations, float radius,
                   size_t* map, int nRepeat) {

  int num_devices;
  cudaGetDeviceCount(&num_devices);

  std::cout << " Running on " << num_devices << " devices" << std::endl;
  std::cout << " The number of iterations is " << nIterations << std::endl;
  std::cout << " The number of kernel execution is " << nRepeat << std::endl;
  std::cout << " The number of particles is " << n_particles << std::endl;

  float *d_randomX, *d_randomY, *d_particleX, *d_particleY;
  size_t *d_map;

  cudaMalloc((void**)&d_randomX, sizeof(float) * n_particles * nIterations);
  cudaMalloc((void**)&d_randomY, sizeof(float) * n_particles * nIterations);
  cudaMalloc((void**)&d_particleX, sizeof(float) * n_particles);
  cudaMalloc((void**)&d_particleY, sizeof(float) * n_particles);

  cudaMemcpy(d_randomX, randomX, sizeof(float) * n_particles * nIterations, cudaMemcpyHostToDevice);
  cudaMemcpy(d_randomY, randomY, sizeof(float) * n_particles * nIterations, cudaMemcpyHostToDevice);

  dim3 dimGrid = dim3(num_devices);
  dim3 dimBlock = dim3(256);

  double time_total = 0.0;

  for (int i = 0; i < nRepeat; i++) {
    // reset the device data
    cudaMemcpy(d_particleX, particleX, sizeof(float) * n_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleY, particleY, sizeof(float) * n_particles, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    Simulation<<< dimGrid, dimBlock >>> (d_particleX, d_particleY, d_randomX, d_randomY, d_map,
                                          n_particles, nIterations, grid_size, radius,
                                          num_devices, 0);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    time_total += time;
  }

  std::cout << "\nAverage kernel execution time: " << (time_total / nRepeat) << " (ms)" << std::endl;
  std::cout << std::endl;

  cudaMemcpy(map, d_map, sizeof(size_t) * n_particles * grid_size * grid_size, cudaMemcpyDeviceToHost);

  // For every cell in the grid, add all the counters from different
  // particles (workers) which are stored in the 3rd dimension of the 'map'
  // array
  for (size_t i = 0; i < n_particles; ++i) {
    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
	if (map[i * grid_size * grid_size + y * grid_size + x] > 0) {
          grid[y][x] += map[i * grid_size * grid_size + y * grid_size + x];
        }
      }
    }
  }  // End loop for number of particles

  cudaFree(d_randomX);
  cudaFree(d_randomY);
  cudaFree(d_particleX);
  cudaFree(d_particleY);
  cudaFree(d_map);
}  // End of function motion_device()


int main(int argc, char* argv[]) {
  if (argc != 3) {
    usage(argv[0], argc, argv);
    return 1;
  }

  // Read command-line arguments
  int nIterations = std::stoi(argv[1]);
  int nRepeat = std::stoi(argv[2]);

  // Cell and Particle parameters
  const size_t grid_size = 21;    // Size of square grid
  const size_t n_particles = 147456;  // Number of particles
  const float radius = 0.5;       // Cell radius = 0.5*(grid spacing)

  // Stores a grid of cells
  int** grid = new int*[grid_size];
  for (size_t i = 0; i < grid_size; i++) grid[i] = new int[grid_size];

  // Stores all random numbers to be used in the simulation
  float* randomX = new float[n_particles * nIterations];
  float* randomY = new float[n_particles * nIterations];

  // Stores X and Y position of particles in the cell grid
  float* particleX = new float[n_particles];
  float* particleY = new float[n_particles];

  // 'map' array replicates grid to be used by each particle
  size_t MAP_SIZE = n_particles * grid_size * grid_size;
  size_t* map = new size_t[MAP_SIZE];
  size_t* map_ref = new size_t[MAP_SIZE];

  // Initialize arrays
  for (size_t i = 0; i < n_particles; i++) {
    // Initial position of particles in cell grid
    particleX[i] = 10.0;
    particleY[i] = 10.0;

    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        map[i * grid_size * grid_size + y * grid_size + x] = 0;
        map_ref[i * grid_size * grid_size + y * grid_size + x] = 0;
      }
    }
  }

  for (size_t y = 0; y < grid_size; y++) {
    for (size_t x = 0; x < grid_size; x++) {
      grid[y][x] = 0;
    }
  }

  // Compute vectors of random values for X and Y directions
  // Set the seed for rand() function.
  // Use a fixed seed for reproducibility/debugging
  srand(17);
  const size_t scale = 100; // Scale of random numbers
  for (size_t i = 0; i < n_particles * nIterations; i++) {
    randomX[i] = rand() % scale;
    randomY[i] = rand() % scale;
  }

  // Start timers
  auto start = std::chrono::steady_clock::now();

  // Call simulation function
  motion_device(particleX, particleY, randomX, randomY, grid, grid_size,
                n_particles, nIterations, radius, map, nRepeat);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "\nSimulation time: " << time << " (ms) ";
  std::cout << std::endl;

  motion_host(particleX, particleY, randomX, randomY, grid, grid_size,
              n_particles, nIterations, radius, map_ref, nRepeat);

  size_t count = 0;
  for (size_t i = 0; i < MAP_SIZE; i++) {
    if (map[i] != map_ref[i])
      count++;
  }
  std::cout << (count <= 2 ? "PASS" : "FAIL") << std::endl;

#ifdef DEBUG
  // Displays final grid only if grid small.
  if (grid_size <= 64) {
    std::cout << "\n ********************** OUTPUT GRID: " << std::endl;
    print_matrix<int>(grid, grid_size, grid_size);
  }
#endif

  // Cleanup
  for (size_t i = 0; i < grid_size; i++) {
    delete grid[i];
  }

  delete[] grid;
  delete[] particleX;
  delete[] particleY;
  delete[] randomX;
  delete[] randomY;
  delete[] map;
  delete[] map_ref;

  return 0;
}