#include <iostream>
#include <chrono>

template <typename T>
void concat(const T* inp1,
            const T* inp2,
                  T* output,
            int sz0, int sz2, int sz1_1, int sz1_2) {
  int nele = sz0 * sz2 * (sz1_1 + sz1_2);

  const size_t n = sz0 * sz2 * (sz1_1 + sz1_2);
  const size_t nblock = (n + 255) / 256;

  #pragma omp parallel for num_threads(nblock)
  for (int idx = 0; idx < nele; idx++) {
    float *dst_ptr = (float *)output + idx;
    int idx2 = idx % sz2;
    idx = idx / sz2;
    int idx1 = idx % (sz1_1 + sz1_2);
    int idx0 = idx / (sz1_1 + sz1_2);
    float *src_ptr;
    int sz1;
    if (idx1 < sz1_1) {
      sz1 = sz1_1;
      src_ptr = (float *)inp1;
    } else {
      idx1 -= sz1_1;
      sz1 = sz1_2;
      src_ptr = (float *)inp2;
    }
    src_ptr += flat_3dim(idx0, idx1, idx2, sz1, sz2);
    *dst_ptr = *src_ptr;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = std::stoi(argv[1]);

  for (int nhead = 6; nhead <= 48; nhead *= 2) {
    srand(nhead);

    const int seq_len = 1024;
    const int batch_size = 8;
    const int hidden_dim = nhead * 128;
    const int head_dim = hidden_dim / nhead;

    const int sl1 = rand() % (seq_len - 1) + 1;
    const int sl2 = seq_len - sl1;
    const int beam_size = 8;

    std::cout << "\n";
    std::cout << "num_head = " << nhead << "\t";
    std::cout << "seq_len = " << seq_len << "\t";
    std::cout << "batch_size = " << batch_size << "\t";
    std::cout << "hidden_dimension = " << hidden_dim << "\t";
    std::cout << "beam_size = " << beam_size << "\n";

    const size_t inp1_size = batch_size * beam_size * hidden_dim * sl1;
    const size_t inp2_size = batch_size * beam_size * hidden_dim * sl2;
    const size_t outp_size = batch_size * beam_size * hidden_dim * seq_len;

    const size_t inp1_size_bytes = inp1_size * sizeof(float);
    const size_t inp2_size_bytes = inp2_size * sizeof(float);
    const size_t outp_size_bytes = outp_size * sizeof(float);

    float size_bytes = 2 * outp_size_bytes * 1e-9;
    std::cout << "Total device memory usage (GB) = " << size_bytes << "\n";

    float *inp1 = (float*) malloc (inp1_size_bytes);
    float *inp2 = (float*) malloc (inp2_size_bytes);
    float *outp = (float*) malloc (outp_size_bytes);
    float *outp_ref = (float*) malloc (outp_size_bytes);

    for (size_t i = 0; i < inp1_size; i++) {
      inp1[i] = rand() % inp1_size;
    }

    for (size_t i = 0; i < inp2_size; i++) {
      inp2[i] = rand() % inp2_size;
    }

    const size_t n = batch_size * beam_size * nhead * head_dim * (sl1 + sl2);
    const size_t nblock = (n + 255) / 256;

    // warmup and verify
    concat(
      inp1, inp2, outp, batch_size * beam_size * nhead, head_dim, sl1, sl2);

    concat_cpu(
      inp1, inp2, outp_ref, batch_size * beam_size * nhead, head_dim, sl1, sl2);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      concat(
        inp1, inp2, outp, batch_size * beam_size * nhead, head_dim, sl1, sl2);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    float avg_time = (time * 1e-3f) / repeat;
    std::cout << "Average kernel execution time: " << avg_time << " (us)\n";
    std::cout << "Average kernel throughput : " << size_bytes / (avg_time * 1e-6) << " (GB/s)\n";

    free(inp1);
    free(inp2);
    free(outp);
    free(outp_ref);
  }

  return 0;
}