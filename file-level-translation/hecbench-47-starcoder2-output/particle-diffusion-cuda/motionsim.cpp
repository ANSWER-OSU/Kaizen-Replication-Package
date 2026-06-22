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
 
void Simulation(float* particleX,
                float* particleY,
		const float* randomX,
                const float* randomY, 
		size_t *map,
                const size_t n_particles,
                unsigned int nIterations,
                int grid_size,
                float radius)
{
  #pragma omp parallel for
  for (size_t ii = 0; ii < n_particles; ++ii) {
  // Start iterations
  // Each iteration:
  //  1. Updates the position of all water molecules
  //  2. Checks if water molecule is inside a cell or not.
  //  3. Updates counter in cells array
  size_t iter = 0;
  float pX = particleX[ii];
  float pY = particleY[ii];
  size_t map_base = ii * grid_size * grid_size;
  while (iter < nIterations) {
    // Computes random displacement for each molecule
    // This example shows random distances between
    // -0.05 units and 0.05 units in both X and Y directions
    // Moves each water molecule by a random vector

    float randnumX = randomX[iter * n_particles + ii];
    float randnumY = randomY[iter * n_particles + ii];

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
      if ((dX * dX + dY * dY <= radius * radius))
        // The map array is organized as (particle, y, x)
        map[map_base + iY * grid_size + iX]++;
    }

    iter++;

  }  // Next iteration

  particleX[ii] = pX;
  particleY[ii] = pY;
 }
}