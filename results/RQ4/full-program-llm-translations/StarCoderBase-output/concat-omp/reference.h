template <typename T>
void concat_cpu (const T *inp1, const T *inp2, T *output, int sz0, int sz2, int sz1_1, int sz1_2) {
  for (int idx0 = 0; idx0 < sz0; idx0++) {
  for (int idx1 = 0; idx1 < (sz1_1 + sz1_2); idx1++) {
  for (int idx2 = 0; idx2 < sz2; idx2++) {
  #pragma omp parallel
  {
    int dst_idx = flat_3dim(idx0, idx1, idx2, sz1_1+sz1_2, sz2);
    int src_idx;
    int sz1;
    int idx1_t = idx1;
    if (idx1_t < sz1_1) {
      sz1 = sz1_1;
      src_idx = flat_3dim(idx0, idx1_t, idx2, sz1, sz2);
    } else {
      idx1_t -= sz1_1;
      sz1 = sz1_2;
      src_idx = flat_3dim(idx0, idx1_t, idx2, sz1, sz2);
    }
    output[dst_idx] = (idx1 < sz1_1) ? inp1[src_idx] : inp2[src_idx];
  }}}}
}