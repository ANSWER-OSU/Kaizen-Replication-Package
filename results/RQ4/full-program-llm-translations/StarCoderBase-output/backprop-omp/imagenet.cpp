void load(BPNN *net)
{
    float *units = net->input_units;
    int nr = layer_size;

    int k = 1;

#pragma omp parallel for
    for (int i = 0; i < nr; i++) {
        units[k] = static_cast<float>(rand()) / RAND_MAX;
        k++;
    }
}