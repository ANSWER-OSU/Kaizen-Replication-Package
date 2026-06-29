void reference (
    int numNeurons, int neurons_per_item, float dt, 
    const float*__restrict__ encode_result,
    float*__restrict__ voltage_array,
    float*__restrict__ reftime_array,
    float tau_rc, float tau_ref,
    const float*__restrict__ bias,
    const float*__restrict__ gain,
    float*__restrict__ spikes)
{
  #pragma omp target teams distribute parallel for num_teams(numNeurons) num_threads(32)
  for (int i = 0; i < numNeurons; i++)
  {
    int neuron_index = i % neurons_per_item;
    int item_index = i / neurons_per_item;

    float voltage = voltage_array[i];
    float ref_time = reftime_array[i];
    float current = bias[neuron_index] + gain[neuron_index] * encode_result[item_index];
    float dV, spike, mult;

    dV = -expm1f(-dt / tau_rc) * (current - voltage);
    voltage = fmaxf(voltage + dV, 0.f);

    ref_time -= dt;

    mult = ref_time;
    mult *= -1.f / dt;
    mult += 1.f;

    mult = fminf(mult, 1.f);
    mult = fmaxf(mult, 0.f);
    
    voltage *= mult;

    if(voltage > 1.f){
      spike = 1.f / dt;
      ref_time = tau_ref + dt * (1.f - (voltage - 1.f) / dV);
      voltage = 0.f;
    }else{
      spike = 0.f;
    }

    reftime_array[i] = ref_time;
    voltage_array[i] = voltage;
    spikes[i] = spike;
  }
}