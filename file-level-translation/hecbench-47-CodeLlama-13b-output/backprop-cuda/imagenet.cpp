#include <cstdlib>
#include <cstdio>
#include <omp.h>

extern int layer_size;

void load(BPNN *net) {
    float *units;
    int nr, i, k;

    nr = layer_size;

    units = net->input_units;

#pragma omp parallel for
    for (i = 0; i < nr; i++) {
        units[k] = (float)std::rand() / RAND_MAX;
        k++;
    }
}