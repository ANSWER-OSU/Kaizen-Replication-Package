template <typename Payoff> 
void longstaff_schwartz(int num_timesteps, 
                        int num_paths, 
                        const Payoff &payoff, 
                        double dt, 
                        double S0, 
                        double r,
                        double sigma,
                        int max_grid_size,
                        const std::vector <double> &host_samples, 
                        std::vector <double> &path_buffer, 
                        std::vector <double> &price_buffer)
{
  // Reset the path buffer.
  std::fill(path_buffer.begin(), path_buffer.end(), 0);

  // Memory to keep intermediate SVD matrices.
  std::vector <double> svds(16*num_timesteps);

  // Storage to keep reductions.
  std::vector<double> partial_sums(3*max_grid_size);
  std::vector<double> beta(9);
  for(int i = 0; i < 9; ++i) beta[i] = 0.0;

  // GPU memory.
  thrust::device_vector<double> path_buffer_d(path_buffer.size());
  thrust::device_vector<double> price_buffer_d(price_buffer.size());
  thrust::device_vector<double> svds_d(svds.size());
  thrust::device_vector<double> partial_sums_d(partial_sums.size());
  thrust::device_vector<double> beta_d(beta.size());