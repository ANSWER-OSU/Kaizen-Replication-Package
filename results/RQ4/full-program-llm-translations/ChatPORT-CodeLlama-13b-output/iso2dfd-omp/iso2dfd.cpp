int main(void) {
    #pragma omp target teams distribute parallel for schedule(static, 1) num_teams(numTeams) num_threads(numThreads)
    for (size_t t = 0; t < numTeams; t++) {
        iso_2dfd_kernel(prev_or_next[t % 2], prev_or_next[(t + 1) % 2], vel, dtDIVdxy, nRows, nCols);
    }
}