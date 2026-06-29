#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
template <typename T>
void flip(const int64_t num_dims, const int64_t num_flip_dims, const int32_t dim_size, const int32_t repeat) {
    vector<int64_t> flip;
    vector<int64_t> shape;
    vector<int64_t> stride;
    for (int64_t i = 0; i < num_dims; i++) {
        shape.push_back(dim_size);
    }
    int64_t n = 1;
    for (int64_t i = 0; i < num_dims; i++) {
        n *= shape[i];
    }
    for (int64_t i = 0; i < num_flip_dims; i++) {
        flip.push_back(i);
    }
    stride.push_back(shape[1] * shape[2]);
    stride.push_back(shape[2]);
    stride.push_back(1);
    int64_t dims_bytes = num_dims * sizeof(int64_t);
    int64_t flip_dims_bytes = num_flip_dims * sizeof(int64_t);
    int64_t input_size_bytes = n * sizeof(T);
    int64_t output_size_bytes = input_size_bytes;
    T *input = new T[input_size_bytes];
    for (int i = 0; i < n; i++) {
        input[i] = (T)i;
    }
    T *output = new T[output_size_bytes];
    T *output_ref = new T[output_size_bytes];
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
        
    }
    auto end = chrono::steady_clock::now();
    auto time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the flip kernel: %f (ms)\n", (time * 1e-6f) / repeat);
    delete [] input;
    delete [] output;
    delete [] output_ref;
}
int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
        return 1;
    }
    const int64_t num_dims = atoi(argv[1]);
    const int64_t dim_size = atoi(argv[2]);
    const int32_t repeat = atoi(argv[3]);
    const int64_t num_flip_dims = num_dims;
    printf("=========== Data type is FP32 ==========\n");
    flip<float>(num_dims, num_flip_dims, dim_size, repeat);
    printf("=========== Data type is FP64 ==========\n");
    flip<double>(num_dims, num_flip_dims, dim_size, repeat);
    return 0;
}