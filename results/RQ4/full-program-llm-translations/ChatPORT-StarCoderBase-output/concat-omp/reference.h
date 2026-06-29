template <typename T>
void concat_cpu (const int numTeams, const int numThreads, const T *__restrict__ inp1, const T *__restrict__ inp2, T *output, int sz0, int sz2, int sz1_1, int sz1_2) {
  #pragma omp target teams distribute parallel for collapse(3) num_teams(numTeams) num_threads(numThreads)
  for (int idx0 = 0; idx0 < sz0; idx0++) {
    for (int idx1 = 0; idx1 < (sz1_1 + sz1_2); idx1++) {
      for (int idx2 = 0; idx2 < sz2; idx2++) {
        float *dst_ptr = (float *)output + flat_3dim(idx0, idx1, idx2, sz1_1 + sz1_2, sz2);
        float *src_ptr;
        int sz1;
        int idx1_t = idx1;
        if (idx1_t < sz1_1) {
          sz1 = sz1_1;
          src_ptr = (float *)inp1;
        } else {
          idx1_t -= sz1_1;
          sz1 = sz1_2;
          src_ptr = (float *)inp2;
        }
        src_ptr += flat_3dim(idx0, idx1_t, idx2, sz1, sz2);
        *dst_ptr = *src_ptr;
      }
    }
  }
}