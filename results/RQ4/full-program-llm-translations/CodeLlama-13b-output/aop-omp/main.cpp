#include <iostream>
#include <cassert>

#pragma omp declare target

// Function Bodies

template< typename Payoff >
static inline 
void do_run(double *h_samples,
            int num_timesteps, 
            int num_paths, 
            const Payoff &payoff, 
            double dt,
            double S0,
            double r,
            double sigma,
            double *d_samples,
            double *d_paths,
            double *d_cashflows,
            double *d_svds,
            int    *d_all_out_of_the_money,
            double *d_temp_storage,
            double *h_price)
{
  // Error checking
  assert(h_samples != NULL);
  assert(num_timesteps > 0);
  assert(num_paths > 0);
  assert(dt > 0);

  // Memory transfer to the device
  #pragma omp target data map(to: h_samples[0:num_timesteps*num_paths]) \
                        map(tofrom: d_samples[0:num_timesteps*num_paths]) \
                        map(to: payoff) \
                        map(to: dt, S0, r, sigma) \
                        map(tofrom: d_paths[0:num_timesteps*num_paths]) \
                        map(tofrom: d_cashflows[0:num_paths]) \
                        map(tofrom: d_svds[0