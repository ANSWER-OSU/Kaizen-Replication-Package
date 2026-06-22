#include <iostream>
#include <cmath>
#include <chrono>

void reference(int numNeurons, int neurons_per_item, float dt, float* encode_result, float* voltage_array, float* reftime_array, float tau_rc, float tau_ref, float* bias, float* gain, float* spikes) {
  for (int i = 0; i < numNeurons; i++) {
    int neuron_index = i % neurons_per_item;
    int item_index = i / neurons_per_item;

    float voltage = voltage_array[i];
    float ref_time = reftime_array[i];
    float current = bias[neuron_index] + gain[neuron_index] * encode_result[item_index];
    float dV, spike, mult;

    dV = -std::expm1(-dt / tau_rc) * (current - voltage);
    voltage = std::fmax(voltage + dV, 0.0f);

    ref_time -= dt;

    mult = ref_time;
    mult *= -1.0f / dt;
    mult += 1.0f;

    mult = std::fmin(mult, 1.0f);
    mult = std::fmax(mult, 0.0f);

    voltage *= mult;

    if (voltage > 1.0f) {
      spike = 1.0f / dt;
      ref_time = tau_ref + dt * (1.0f - (voltage - 1.0f) / dV);
      voltage = 0.0f;
    } else {
      spike = 0.0f;
    }
    
    reftime_array[i] = ref_time;
    voltage_array[i] = voltage;
    spikes[i] = spike;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <neurons_per_item> <num_items> <num_steps>" << std::endl;
    return 1;
  }

  const int neurons_per_item = std::atoi(argv[1]);
  const int num_items = std::atoi(argv[2]);
  const int num_steps = std::atoi(argv[3]);

  const int num_neurons = neurons_per_item * num_items;

  float dt = 0.1f;    // time step
  float tau_rc = 10.0f; // membrane time constant
  float tau_ref = 2.0f; // refactory time

  float* encode_result = new float[num_items];
  float* bias = new float[neurons_per_item];
  float* gain = new float[neurons_per_item];

  // test
  float* voltage = new float[num_neurons];
  float* reftime = new float[num_neurons];
  float* spikes = new float[num_neurons];

  // expected
  float* voltage_gold = new float[num_neurons];
  float* reftime_gold = new float[num_neurons];
  float* spikes_gold = new float[num_neurons];

  std::srand(123);
  for (int i = 0; i < num_items; i++) {
    encode_result[i] = std::rand() / (float) RAND_MAX;
  }
  for (int i = 0; i < num_neurons; i++) {
    voltage_gold[i] = voltage[i] = 1.0f + std::rand() / (float) RAND_MAX;
    reftime_gold[i] = reftime[i] = std::rand() % 5 / 10.0f;
  }
  for (int i = 0; i < neurons_per_item; i++) {
    bias[i] = std::rand() / (float) RAND_MAX;
    gain[i] = std::rand() / (float) RAND_MAX + 0.5f;
  }

  #pragma omp parallel for collapse(2)
  for (int step = 0; step < num_steps; step++) {
    for (int i = 0; i < num_neurons; i++) {
      int neuron_index = i % neurons_per_item;
      int item_index = i / neurons_per_item;

      float voltage = voltage_gold[i];
      float ref_time = reftime_gold[i];
      float current = bias[neuron_index] + gain[neuron_index] * encode_result[item_index];
      float dV, spike, mult;
      dV = -std::expm1(-dt / tau_rc) * (current - voltage);
      voltage = std::fmax(voltage + dV, 0.0f);
      ref_time -= dt;
      mult = ref_time;
      mult *= -1.0f / dt;
      mult += 1.0f;
      mult = std::fmin(mult, 1.0f);
      mult = std::fmax(mult, 0.0f);
      voltage *= mult;
      if (voltage > 1.0f) {
        spike = 1.0f / dt;
        ref_time = tau_ref + dt * (1.0f - (voltage - 1.0f) / dV);
        voltage = 0.0f;
      } else {
        spike = 0.0f;
      }
      reftime_gold[i] = ref_time;
      voltage_gold[i] = voltage;
      spikes_gold[i] = spike;
    }
  }

  bool ok = true;
  for (int i = 0; i < num_neurons; i++) {
    if (std::fabs(spikes_gold[i] - spikes[i]) > 1e-3) {
      std::cout << "@" << i << ": " << spikes[i] << " " << spikes_gold[i] << std::endl;
      ok = false;
      break;
    }
  }

  delete[] encode_result;
  delete[] voltage;
  delete[] voltage_gold;
  delete[] reftime;
  delete[] reftime_gold;
  delete[] bias;
  delete[] gain;
  delete[] spikes;
  delete[] spikes_gold;

  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  return 0;
}