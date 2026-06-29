#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <omp.h>
#include "reference.h"

// Helper functions

// This function displays correct usage and parameters
void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " <Number of iterations within the kernel> ";
  std::cout << "<Kernel execution count>\n\n";
}

// This function prints a 2D matrix
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

// This function prints a vector
template <typename T>
void print_vector(T* vector, size_t n) {
  std::cout << std::endl;
  for (size_t i = 0; i < n; ++i) {
    std::cout << vector[i] << " ";
  }
  std::cout << std::endl;
}

void Simulation(float*__restrict__ a_particleX,
                float*__restrict__ a_particleY,
		const float*__restrict__ a_randomX,
                const float*__restrict__ a_randomY, 
		size_t *__restrict__ a_map,
                const size_t n_particles,
                unsigned int nIterations,
                int grid_size,
                float radius)
{
  size_t ii = omp_get_thread_num();
  if (ii >= n_particles) return;
  // Start iterations
  // Each iteration:
  //  1. Updates the position of all water molecules
  //  2. Checks if water molecule is inside a cell or not.
  //  3. Updates counter in cells array
  size_t iter = 0;
  float pX = a_particleX[ii];
  float pY = a_particleY[ii];
  size_t map_base = ii * grid_size * grid_size;
  while (iter < nIterations) {
    // Computes random displacement for each molecule
    // This example shows random distances between
    // -0.05 units and 0.05 units in both X and Y directions
    // Moves each water molecule by a random vector

    float randnumX = a_randomX[iter * n_particles + ii];
    float randnumY = a_randomY[iter * n_particles + ii];

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
      if ((dX * dX + dY * dY <= radius * radius)) {
        // The map array is organized as (particle, y, x)
        #pragma omp atomic
        a_map[map_base + iY * grid_size + iX]++;
      }
    }

    iter++;

  }  // Next iteration

  a_particleX[ii] = pX;
  a_particleY[ii] = pY;
}

class motion_device {
 private:
  float* particleX;
  float* particleY;
  float* randomX;
  float* randomY;
  int** grid;
  size_t grid_size;
  size_t n_particles;
  unsigned int nIterations;
  float radius;
  size_t* map;
  size_t MAP_SIZE;
  int nRepeat;

 public:
  motion_device(int argc, char* argv[]) {
    if (argc != 3) {
      usage(argv[0]);
      exit(1);
    }

    // Read command-line arguments
    nIterations = std::stoi(argv[1]);
    nRepeat = std::stoi(argv[2]);

    // Cell and Particle parameters
    grid_size = 21;                                           // Size of square grid
    n_particles = 147456;                                      // Number of particles
    radius = 0.5;                                              // Cell radius = 0.5*(grid spacing)

    grid = new int*[grid_size];
    for (size_t i = 0; i < grid_size; i++) grid[i] = new int[grid_size];

    randomX = new float[n_particles * nIterations];
    randomY = new float[n_particles * nIterations];
    particleX = new float[n_particles];
    particleY = new float[n_particles];

    MAP_SIZE = n_particles * grid_size * grid_size;
    map = new size_t[MAP_SIZE];
    size_t* map_ref = new size_t[MAP_SIZE];

    // initialize arrays
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

    srand(17);
    const size_t scale = 100;
    for (size_t i = 0; i < n_particles * nIterations; i++) {
      randomX[i] = rand() % scale;
      randomY[i] = rand() % scale;
    }

    use_device();

    motion_host(particleX, particleY, randomX, randomY, grid, grid_size,
              n_particles, nIterations, radius, map_ref, nRepeat);

    size_t count = 0;
    for (size_t i = 0; i < MAP_SIZE; i++) {
      if (map[i] != map_ref[i])
        count++; 
    }
    std::cout << (count <= 2 ? "PASS" : "FAIL") << std::endl;

    for (size_t i = 0; i < grid_size; i++) delete grid[i];
    delete[] grid;
    delete[] particleX;
    delete[] particleY;
    delete[] randomX;
    delete[] randomY;
    delete[] map;
    delete[] map_ref;
    }

  void use_device();
};

void motion_device::use_device() {
    auto start = std::chrono::steady_clock::now();

    #pragma omp parallel
    Simulation(particleX, particleY, randomX, randomY, map, n_particles, nIterations, grid_size, radius);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << std::endl;
    std::cout << "Simulation time: " << time * 1e-9 << " (s) ";
    std::cout << std::endl;

    // For every cell in the grid, add all the counters from different
    // particles which are stored in the 3rd dimension of the 'map'
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
}

int main(int argc, char* argv[]) {
  motion_device device(argc, argv);
  return 0;
}