#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>

#define CUDA_SAFE_CALL(x) (x)
#define CUBLAS_SAFE_CALL(x) (x)

#define real float
#define PI_f ((float)1.0f)
#define PI_r ((float)10e-3f)

typedef struct {
  float pos[3];
  float vel[3];
  float acc[3];
  float mass;
} Particle;

typedef struct {
  int npart;
  int nsteps;
  float tstep;
  int sfreq;
  float total_time_, total_flops_, kenergy_, av, dev;
} GSimulation;

void SelectParticles(Particle *p, int n, int nPart) {
  for (int i = 0; i < nPart; ++i) {
    particles_[i].vel[0] = p[i].vel[0];
    particles_[i].vel[1] = p[i].vel[1];
    particles_[i].vel[2] = p[i].vel[2];
    particles_[i].pos[0] = p[i].pos[0];
    particles_[i].pos[1] = p[i].pos[1];
    particles_[i].pos[2] = p[i].pos[2];
    particles_[i].mass = p[i].mass;
  }
}

void ComputeForce(Particle *p, int n, const real kSofteningSquared, const real kG) {
  for (int i = 0; i < n; ++i) {
    part_acc_[i] = 0;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (p[i].mass * p[j].mass == 0.0) {
        continue;
      }

      real softeningSquared = kSofteningSquared;
      real dx = p[j].pos[0] - p[i].pos[0];
      real dy = p[j].pos[1] - p[i].pos[1];
      real dz = p[j].pos[2] - p[i].pos[2];
      real d = sqrt(dx * dx + dy * dy + dz * dz + softeningSquared);
      real invDist = 1.0 / d;
      real s = invDist * invDist * invDist;

      real ax = (p[j].mass * dx * s) * invDist;
      real ay = (p[j].mass * dy * s) * invDist;
      real az = (p[j].mass * dz * s) * invDist;

      atomicAdd(&part_acc_[i], (kG * ax));
      atomicAdd(&part_acc_[i], (kG * ay));
      atomicAdd(&part_acc_[i], (kG * az));
    }
  }
}

void IntegrateSystem(Particle *p, int n, double dt) {
  for (int i = 0; i < n; ++i) {
    p[i].vel[0] += p[i].acc[0] * dt;
    p[i].vel[1] += p[i].acc[1] * dt;
    p[i].vel[2] += p[i].acc[2] * dt;
    p[i].pos[0] += p[i].vel[0] * dt;
    p[i].pos[1] += p[i].vel[1] * dt;
    p[i].pos[2] += p[i].vel[2] * dt;
  }
}

void AccumulateEnergy(real *e, int n) {
  for (int i = 0; i < n; ++i) {
    e[i] = 0.5 * particles_[i].mass * (particles_[i].vel[0] * particles_[i].vel[0] + particles_[i].vel[1] * particles_[i].vel[1] + particles_[i].vel[2] * particles_[i].vel[2]);
  }

  GSimulation *gSimulation = new GSimulation();
  gSimulation->kenergy_ = 0.5;
  gSimulation->total_flops_ = 1;
  gSimulation->total_time_ = 0;

  for (int s = 1; s <= gSimulation.nsteps; s++) {
    TimeInterval ts0;

    ComputeForce(p, n, kSofteningSquared, kG);
    IntegrateSystem(p, n, dt);
    AccumulateEnergy(e, n);

    double elapsed_seconds = ts0.Elapsed();

    GSimulation->kenergy_ += energy;
    GSimulation->total_time_ += elapsed_seconds;

    if ((s % get_sfreq()) == 0) {
      gSimulation->nf = gSimulation.nf + 1;
      GSimulation->av += gSimulation.total_flops_ / gSimulation.nf;
      GSimulation->dev += (gSimulation.total_flops_ * gSimulation.nf) / (gSimulation.nf * gSimulation.nf);
    }
  }

  GSimulation->av /= (double)(gSimulation->nf - 2);
  GSimulation->dev = sqrt((gSimulation->dev / (double)(gSimulation->nf - 2) - GSimulation->av * GSimulation->av));

  cout << "Total energy: " << kenergy_ << "\n";
  cout << "# Total Time (s)      : " << total_time_ << "\n";
  cout << "# Average Performance : " << GSimulation.av << " +- " << GSimulation.dev << "\n";
}

int main() {
  GSimulation *gSimulation = new GSimulation();

  gSimulation->InitPos();
  gSimulation->InitVel();
  gSimulation->InitAcc();
  gSimulation->InitMass();

  gSimulation.Start();
  return 0;
}