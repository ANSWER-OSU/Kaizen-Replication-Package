#include <omp.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <random>
#include "GSimulation.hpp"
#include "GSimulationKernels.hpp"

using namespace std;

/* Default Constructor for the GSimulation class which sets up the default
 * values for number of particles, number of integration steps, time steo and
 * sample frequency */
GSimulation::GSimulation() {
  std::cout << "==============================="
            << "\n";
  std::cout << " Initialize Gravity Simulation"
            << "\n";
  set_npart(16000);
  set_nsteps(10);
  set_tstep(0.1);
  set_sfreq(1);
}

/* Set the number of particles */
void GSimulation::SetNumberOfParticles(int N) { set_npart(N); }

/* Set the number of integration steps */
void GSimulation::SetNumberOfSteps(int N) { set_nsteps(N); }

/* Initialize the position of all the particles using random number generator
 * between 0 and 1.0 */
void GSimulation::InitPos() {
  #pragma omp target teams distribute parallel for num_teams(1) num_threads(1)
  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].pos[0] = rand() / (RealType)RAND_MAX;
    particles_[i].pos[1] = rand() / (RealType)RAND_MAX;
    particles_[i].pos[2] = rand() / (RealType)RAND_MAX;
  }
}

/* Initialize the velocity of all the particles using random number generator
 * between -1.0 and 1.0 */
void GSimulation::InitVel() {
  #pragma omp target teams distribute parallel for num_teams(1) num_threads(1)
  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].vel[0] = -1.0 + 2.0 * rand() / (RealType)RAND_MAX;
    particles_[i].vel[1] = -1.0 + 2.0 * rand() / (RealType)RAND_MAX;
    particles_[i].vel[2] = -1.0 + 2.0 * rand() / (RealType)RAND_MAX;
  }
}

/* Initialize the acceleration of all the particles to 0 */
void GSimulation::InitAcc() {
  #pragma omp target teams distribute parallel for num_teams(1) num_threads(1)
  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].acc[0] = 0.f;
    particles_[i].acc[1] = 0.f;
    particles_[i].acc[2] = 0.f;
  }
}

/* Initialize the mass of all the particles using a random number generator
 * between 0 and 1 */
void GSimulation::InitMass() {
  RealType n = static_cast<RealType>(get_npart());
  #pragma omp target teams distribute parallel for num_teams(1) num_threads(1)
  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].mass = n * rand() / (RealType)RAND_MAX;
  }
}

/* This function does the simulation logic for Nbody */
void GSimulation::Start() {
  int num_threads = omp_get_num_threads();
  int num_teams = omp_get_num_teams();
  RealType dt = get_tstep();
  int n = get_npart();
  std::vector<RealType> energy(n, 0.f);
  // allocate particles
  particles_.resize(n);

  InitPos();
  InitVel();
  InitAcc();
  InitMass();

#ifdef DEBUG
  PrintHeader();
#endif

  total_time_ = 0.;

  constexpr float kSofteningSquared = 1e-3f;
  // prevents explosion in the case the particles are really close to each other
  constexpr float kG = 6.67259e-11f;
  double gflops = 1e-9 * ((11. + 18.) * n * n + n * 19.);
  int nf = 0;
  double av = 0.0, dev = 0.0;

  #pragma omp target teams distribute num_teams(num_teams)
  for (int t = 0; t < num_teams; ++t) {
    #pragma omp parallel num_threads(num_threads)
    {
      Particle *p;
      cudaMalloc((void**)&p, sizeof(Particle) * n);
      #pragma omp for
      for (int i = 0; i < n; ++i) {
        p[i] = particles_[i];
      }
      RealType *e;
      cudaMalloc((void**)&e, sizeof(RealType) * n);
      #pragma omp for
      for (int i = 0; i < n; ++i) {
        e[i] = energy[i];
      }

      accelerate_particles(p, n, kSofteningSquared, kG);
      update_particles(p, e, n, dt);
      accumulate_energy(e, n);

      cudaDeviceSynchronize();
      #pragma omp barrier
      double elapsed_seconds = omp_get_wtime();

      #pragma omp for
      for (int i = 0; i < n; ++i) {
        energy[i] = e[i];
      }

      if ((t % get_sfreq()) == 0) {
        nf += 1;
#ifdef DEBUG
        std::cout << " " << std::left << std::setw(8) << t << std::left
                  << std::setprecision(5) << std::setw(8) << t * get_tstep()
                  << std::left << std::setprecision(5) << std::setw(12)
                  << energy[0] << std::left << std::setprecision(5)
                  << std::setw(12) << elapsed_seconds << std::left
                  << std::setprecision(5) << std::setw(12)
                  << gflops * get_sfreq() / elapsed_seconds << "\n";
#endif
        if (nf > 2) {
          av += gflops * get_sfreq() / elapsed_seconds;
          dev += gflops * get_sfreq() * gflops * get_sfreq() /
                 (elapsed_seconds * elapsed_seconds);
        }
      }
    }
  }  // end of the team loop
  total_time_ = omp_get_wtime();
  total_flops_ = gflops * get_nsteps();
  av /= (double)(nf - 2);
  dev = sqrt(dev / (double)(nf - 2) - av * av);

  std::cout << "\n";
  std::cout << "# Total Energy        : " << energy[0] << "\n";
  std::cout << "# Total Time (s)      : " << total_time_ << "\n";
  std::cout << "# Average Performance : " << av << " +- " << dev << "\n";
  std::cout << "==============================="
            << "\n";

  #pragma omp target teams distribute parallel for num_teams(1) num_threads(1)
  for (int i = 0; i < n; ++i) {
    particles_[i] = p[i];
  }
  cudaFree(p);
  cudaFree(e);
}

#ifdef DEBUG
/* Print the headers for the output */
void GSimulation::PrintHeader() {
  std::cout << " nPart = " << get_npart() << "; "
            << "nSteps = " << get_nsteps() << "; "
            << "dt = " << get_tstep() << "\n";

  std::cout << "------------------------------------------------"
            << "\n";
  std::cout << " " << std::left << std::setw(8) << "s" << std::left
            << std::setw(8) << "dt" << std::left << std::setw(12) << "energy"
            << std::left << std::setw(12) << "time (s)" << std::left
            << std::setw(12) << "GFLOPS"
            << "\n";
  std::cout << "------------------------------------------------"
            << "\n";
}
#endif