#include <iostream>
#include "GSimulation.hpp"
#include "GSimulationReference.hpp"

int main(int argc, char** argv) {
  int n;      // number of particles
  int nstep;  // number ot integration steps

  GSimulation sim;

  if (argc > 1) {
    n = std::atoi(argv[1]);
    sim.SetNumberOfParticles(n);
    if (argc == 3) {
      nstep = std::atoi(argv[2]);
      sim.SetNumberOfSteps(nstep);
    }
  }

#pragma omp parallel
  {
    sim.Start();
  }

  sim.Verify();

  return 0;
}