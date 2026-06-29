//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <omp.h>

int main(int argc, char** argv) {
  int n;      // number of particles
  int nstep;  // number ot integration steps

  if (argc > 1) {
    n = std::atoi(argv[1]);
#pragma omp parallel
    {
      // Code that needs to run in parallel using OpenMP
      // ...
    }
    if (argc == 3) {
      nstep = std::atoi(argv[2]);
      // OpenMP parallelism can be applied in parallel loops or data structures,
      // depending on what needs to be done.
      // Code to parallelize parts of Loops, Arrays and ordinary data structures...
      // ...
    }
  }

  return 0;
}