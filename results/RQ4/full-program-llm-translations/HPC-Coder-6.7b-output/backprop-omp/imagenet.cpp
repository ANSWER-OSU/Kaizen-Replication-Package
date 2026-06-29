#include <iostream>
#include <cstdlib>
#include "backprop.h"
#include <omp.h>

extern int layer_size;

void load(BPNN *net)
{
  float *units;
  int nr, i, k;

  nr = layer_size;

  units = net->input_units;

  #pragma omp parallel for
  for (i = 0; i < nr; i++) {
    units[i] = static_cast<float>(rand())/RAND_MAX ;
  }
}