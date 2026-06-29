void lstm_inference(const int numTeams, const int numThreads, const float* x, 
             const float* inW, 
             const float* intW, 
             const float* intB, 
             const float* outW, 
             const float* outB,
                   float* y)
{
  #pragma omp target teams distribute parallel for num_teams(numTeams) num_threads(numThreads) 
  for (int i = 0; i < N; i++) {
    float hidden_in[5];
    float cell_in[5];
    float cell_out[5];
    float hidden_out[5];
    float int_out[5];

    for (int j = 0; j < 5; ++j) {
      hidden_in[j]  = x[i * SAMPLE_TEST_LEN + j];
      cell_in[j]    = 0.0f;
      cell_out[j]   = 0.0f;
      hidden_out[j] = 0.0f;
      int_out[j]    = 0.0f;
    }

    for (int j = 0; j < 128; j++) {
      float input_gate[5];
      float forget_gate[5];
      float output_gate[5];
      for (int k = 0; k < 5; ++k) {
        input_gate[k]   = inW[k] * hidden_in[k];
        forget_gate[k]  = inW[5 + k] * hidden_in[k];
        output_gate[k]  = inW[10 + k] * hidden_in[k];
      }
      for (int l = 0; l < 5; ++l) {
        int idx = 11 + l;
        input_gate[l]   += intW[idx + 0 * 5 + j] * cell_in[0];
        input_gate[l]   += intW[idx + 1 * 5 + j] * cell_in[1];
        input_gate[l]   += intW[idx + 2 * 5 + j] * cell_in[2];
        input_gate[l]   += intW[idx + 3 * 5 + j] * cell_in[3];
        input_gate[l]   += intW[idx + 4 * 5 + j] * cell_in[4];
        forget_gate[l]  += intW[idx + 5 * 5 + j] * cell_in[0];
        forget_gate[l]  += intW[idx + 6 * 5 + j] * cell_in[1];
        forget_gate[l]  += intW[idx + 7 * 5 + j] * cell_in[2];
        forget_gate[l]  += intW[idx + 8 * 5 + j] * cell_in[3];
        forget_gate[l]  += intW[idx + 9 * 5 + j] * cell_in[4];
        output_gate[l]  += intW[idx + 10 * 5 + j] * cell_in[0];
        output_gate[l]  += intW[idx + 11 * 5 + j] * cell_in[1];
        output_gate[l]  += intW[idx + 12 * 5 + j] * cell_in[2];
        output_gate[l]  += intW[idx + 13 * 5 + j] * cell_in[3];
        output_gate[l]  += intW[idx + 14 * 5 + j] * cell_in[4];
      }
      for (int k = 0; k < 5; ++k) {
        input_gate[k]   += intB[k];
        forget_gate[k]  += intB[5 + k];
        output_gate[k]  += intB[10 + k];
      }
      sigmoid_layer(input_gate, cell_in);
      sigmoid_layer(forget_gate, cell_out);
      tanh_layer(output_gate, hidden_out);
      for (int l = 0; l < 5; ++l) {
        cell_in[l] = cell_out[l] * hidden_in[l] + cell_in[l] * input_gate[l];
        hidden_in[l] = cell_out[l] * hidden_out[l];
      }
    }
    for (int j = 0; j < 5; ++j) {
      int idx = 11 + j;
      y[i * SAMPLE_TEST_LEN + j] = outW[j] * hidden_in[j];
      y[i * SAMPLE_TEST_LEN + j] += intW[idx + 0 * 5 + 128] * cell_in[0];
      y[i * SAMPLE_TEST_LEN + j] += intW[idx + 1 * 5 + 128] * cell_in[1];
      y[i * SAMPLE_TEST_LEN + j] += intW[idx + 2 * 5 + 128] * cell_in[2];
      y[i * SAMPLE_TEST_LEN + j] += intW[idx + 3 * 5 + 128] * cell_in[3];
      y[i * SAMPLE_TEST_LEN + j] += intW[idx + 4 * 5 + 128] * cell_in[4];
      y[i * SAMPLE_TEST_LEN + j] += outB;
    }
  }
}