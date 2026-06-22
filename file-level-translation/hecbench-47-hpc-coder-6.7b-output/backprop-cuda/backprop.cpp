#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <omp.h>

const float ABS(const float x)
{
  return x > 0 ? x : -x;
}

const void fastcopy(void *to, const void *from, const int len)
{
  memcpy(to, from, len);
}

float drnd()
{
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  return dis(gen);
}

float dpn1()
{
  return 2.0 * drnd() - 1.0f;
}

const float squash(const float x)
{
  return 1.0f / (1.0f + std::exp(-x));
}

std::vector<std::vector<float>> alloc_2d_dflt(const int rows, const int cols)
{
  std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));
  return result;
}

std::vector<float> alloc_1d_dflt(const int size)
{
  std::vector<float> result(size, 0.0f);
  return result;
}

void bpnn_randomize_weights(std::vector<std::vector<float>>& w,
                            const int m, const int n)
{
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      w[i][j] = drnd();
    }
  }
}

void bpnn_randomize_row(std::vector<float>& w, const int m)
{
  for (int i = 0; i < m; i++) {
    w[i] = drnd();
  }
}

void bpnn_zero_weights(std::vector<std::vector<float>>& w,
                        const int m, const int n)
{
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      w[i][j] = 0.0f;
    }
  }
}

void bpnn_initialize(const int seed)
{
  std::srand(seed);
}

struct BPNN
{
  int input_n;
  int hidden_n;
  int output_n;

  std::vector<float> input_units;
  std::vector<float> hidden_units;
  std::vector<float> output_units;

  std::vector<float> hidden_delta;
  std::vector<float> output_delta;
  std::vector<float> target;

  std::vector<std::vector<float>> input_weights;
  std::vector<std::vector<float>> hidden_weights;

  std::vector<std::vector<float>> input_prev_weights;
  std::vector<std::vector<float>> hidden_prev_weights;
};

BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
  BPNN* newnet = new BPNN();

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;

  newnet->input_units = alloc_1d_dflt(n_in + 1);
  newnet->hidden_units = alloc_1d_dflt(n_hidden + 1);
  newnet->output_units = alloc_1d_dflt(n_out + 1);

  newnet->hidden_delta = alloc_1d_dflt(n_hidden + 1);
  newnet->output_delta = alloc_1d_dflt(n_out + 1);
  newnet->target = alloc_1d_dflt(n_out + 1);

  newnet->input_weights = alloc_2d_dflt(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dflt(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dflt(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dflt(n_hidden + 1, n_out + 1);

  return newnet;
}

void bpnn_free(BPNN* net)
{
  if (net != NULL)
  {
    delete net;
  }
}

BPNN *bpnn_create(const int n_in, const int n_hidden, const int n_out)
{
  BPNN *newnet = bpnn_internal_create(n_in, n_hidden, n_out);
#ifndef INITZERO
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
#endif
  bpnn_randomize_row(newnet->target, n_out);

  return newnet;
}

void bpnn_layerforward(const std::vector<float>& l1,
                             std::vector<float>& l2,
                             const std::vector<std::vector<float>> &conn,
                             const int n1, const int n2)
{
  for (int j = 1; j <= n2; j++)
  {
    float sum = 0.0f;

    for (int k = 0; k <= n1; k++)
    {
      sum += conn[k][j] * l1[k];
    }

    l2[j] = squash(sum);
  }
}

void bpnn_output_error(std::vector<float>& delta,
                         const std::vector<float>& target,
                         const std::vector<float>& output,
                         const int nj, float& err)
{
  float errsum = 0.0f;
  for (int j = 1; j <= nj; j++)
  {
    const float o = output[j];
    const float t = target[j];
    delta[j] = o * (1.0f - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  err = errsum;
}

void bpnn_hidden_error(std::vector<float>& delta_h,
                          const int nh,
                          std::vector<float>& delta_o,
                          const int no,
                          std::vector<std::vector<float>>& who,
                          std::vector<float>& hidden,
                          float& err)
{
  float errsum = 0.0f;

  for (int j = 1; j <= nh; j++)
  {
    const float h = hidden[j];
    float sum = 0.0f;
    for (int k = 1; k <= no; k++)
    {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0f - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  err = errsum;
}

void bpnn_adjust_weights(std::vector<float>& delta,
                             const int ndelta,
                             const std::vector<float>& ly,
                             const int nly,
                             std::vector<std::vector<float>>& w,
                             std::vector<std::vector<float>>& oldw)
{
#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for
#endif
  for (int j = 1; j <= ndelta; j++)
  {
    for (int k = 0; k <= nly; k++)
    {
      float new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
          
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}

void bpnn_feedforward(BPNN* net)
{
  int in = net->input_n;
  int hid = net->hidden_n;
  int out = net->output_n;

  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
}

void bpnn_train(BPNN* net, float& eo, float& eh)
{
  int in = net->input_n;
  int hid = net->hidden_n;
  int out = net->output_n;

  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);

  float out_err, hid_err;
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, hid_err);
  eo = out_err;
  eh = hid_err;

  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);
}

void bpnn_save(BPNN* net, const char* filename)
{
  int n1 = net->input_n;
  int n2 = net->hidden_n;
  int n3 = net->output_n;

  std::ofstream s(filename, std::ios::binary);

  s.write(reinterpret_cast<const char*>(&n1), sizeof(int));
  s.write(reinterpret_cast<const char*>(&n2), sizeof(int));
  s.write(reinterpret_cast<const char*>(&n3), sizeof(int));

  float* to = &net->input_weights[0][0];
  float* from = &net->input_weights[0][0];
  s.write(reinterpret_cast<const char*>(&to[0]), (n1 + 1) * (n2 + 1) * sizeof(float));

  to = &net->hidden_weights[0][0];
  from = &net->hidden_weights[0][0];
  s.write(reinterpret_cast<const char*>(&to[0]), (n2 + 1) * (n3 + 1) * sizeof(float));

  s.close();
}

BPNN* bpnn_read(const char* filename)
{
  std::ifstream s(filename, std::ios::binary);
  if (!s.is_open()) {
    return NULL;
  }

  int n1, n2, n3;
  s.read(reinterpret_cast<char*>(&n1), sizeof(int));
  s.read(reinterpret_cast<char*>(&n2), sizeof(int));
  s.read(reinterpret_cast<char*>(&n3), sizeof(int));

  BPNN* net = bpnn_internal_create(n1, n2, n3);

  s.read(reinterpret_cast<char*>(net->input_weights[0]), (n1 + 1) * (n2 + 1) * sizeof(float));
  s.read(reinterpret_cast<char*>(net->hidden_weights[0]), (n2 + 1) * (n3 + 1) * sizeof(float));

  s.close();

  return net;
}