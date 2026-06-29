int n = get_npart();
std::vector<RealType> energy(n, 0.f);
particles_.resize(n);

InitPos();
InitVel();
InitAcc();
InitMass();

total_time_ = 0.;

constexpr float kSofteningSquared = 1e-3f;
constexpr float kG = 6.67259e-11f;
double gflops = 1e-9 * ((11. + 18.) * n * n + n * 19.);
int nf = 0;
double av = 0.0, dev = 0.0;

Particle *p;
RealType *e;

accelerate_particles(p, n, kSofteningSquared, kG);
update_particles(p, e, n, dt);
accumulate_energy(e, n);

kenergy_ = 0.5 * energy[0];

double elapsed_seconds = ts0.Elapsed();

gflops * get_sfreq() / elapsed_seconds;

av += gflops * get_sfreq() / elapsed_seconds;
dev += gflops * get_sfreq() * gflops * get_sfreq() /
       (elapsed_seconds * elapsed_seconds);