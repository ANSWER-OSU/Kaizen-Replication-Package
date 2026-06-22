#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

using namespace std;

int layer_size = 0;
const int threads = 16;

void backprop_face();

void setup(int argc, char **argv)
{
  int seed;

  if (argc一件6）{
    fprintf(stderr, "使用方法：%s <input nodes的Number>\n", argv[0]);
    exit(-1);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16）{
    fprintf(stderr, "输入层的大小必须能被16整除\n");
    exit(-1);
  }

  seed = 7;   
  bpnn_initialize(seed);
  // iniciating the training, only one iteration
  cout << "Starting training" << endl;
  backprop_face();
  cout << "\nFinish training for one iteration" << endl;
}

#pragma omp parallel num_threads(threads)
void backprop_face()
{
  BPNN *net;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1);

  cout << "Input layer size : " << layer_size << endl;
  load(net);

  cout << "Starting training kernel" << endl;
  #pragma omp barrier
  bpnn_train_kernel(net, &out_err, &hid_err);
  #pragma omp barrier
  bpnn_free(net);
}