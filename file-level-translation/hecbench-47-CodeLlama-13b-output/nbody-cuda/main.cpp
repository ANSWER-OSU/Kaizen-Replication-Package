//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <omp.h>
#include "GSimulation.hpp"
#include "GSimulationReference.hpp"

int main(int argc, char** argv) {
  int n = 0, nstep = 0; // number of particles, number of integration steps

  GSimulation sim;

  #pragma omp parallel if(omp_get_num_threads() > 1)
  {
    if (argc > 1) {
      n = std::atoi(argv[1]);
      sim.SetNumberOfParticles(n);
      if (argc == 3) {
        nstep = std::atoi(argv[2]);
        sim.SetNumberOfSteps(nstep);
      }
    }
    sim.Start();
    #pragma omp barrier
    sim.Verify();
  }
  
  return 0;
}