using namespace std;

struct Particle {
  float pos[3], vel[3], acc[3];
  float mass;
};

void accelerate_particles( float *p, int n, float kSofteningSquared, float kG ) {
  float acc0, acc1, acc2;
  float dx, dy, dz, distance_sqr, distance_inv;

  for ( int i = 0; i < n; i++ ) {
    auto pi = p[i];
    acc0 = pi.acc[0];
    acc1 = pi.acc[1];
    acc2 = pi.acc[2];

    for ( int j = 0; j < n; j++ ) {
      auto pj = p[j];
      if (i == j) continue;

      dx = pj.pos[0] - pi.pos[0];
      dy = pj.pos[1] - pi.pos[1];
      dz = pj.pos[2] - pi.pos[2];

      distance_sqr = dx*dx + dy*dy + dz*dz + kSofteningSquared;
      distance_inv = sqrt(distance_sqr);

      acc0 += dx * kG * pj.mass * distance_inv * distance_inv * distance_inv;
      acc1 += dy * kG * pj.mass * distance_inv * distance_inv * distance_inv;
      acc2 += dz * kG * pj.mass * distance_inv * distance_inv * distance_inv;
    }

    pi.acc[0] = acc0;
    pi.acc[1] = acc1;
    pi.acc[2] = acc2;
    p[i] = pi;
  }
}

void update_particles( float *__restrict__ p, float *__restrict__ e, int n, float dt ) {
  float vel0, vel1, vel2;

  for ( int i = 0; i < n; i++ ) {
    auto pi = p[i];

    vel0 = pi.vel[0];
    vel1 = pi.vel[1];
    vel2 = pi.vel[2];

    pi.vel[0] += pi.acc[0] * dt;
    pi.vel[1] += pi.acc[1] * dt;
    pi.vel[2] += pi.acc[2] * dt;

    pi.pos[0] += pi.vel[0] * dt;
    pi.pos[1] += pi.vel[1] * dt;
    pi.pos[2] += pi.vel[2] * dt;

    pi.acc[0] = 0.f;
    pi.acc[1] = 0.f;
    pi.acc[2] = 0.f;

    e[i] = pi.mass * ( pi.vel[0]*pi.vel[0] + pi.vel[1]*pi.vel[1] + pi.vel[2]*pi.vel[2] );

    p[i] = pi;
  }
}

void accumulate_energy( float *e, int n ) {
  for ( int i = 1; i < n; i++ ) e[0] += e[i];
}