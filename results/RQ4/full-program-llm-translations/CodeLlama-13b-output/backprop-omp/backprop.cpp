#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))
#define fastcopy(to,from,len)    \
   {                             
      char *_to,*_from;           
      int _i,_l;                   
      _to = (char *)(to);        
      _from = (char *)(from);    
      _l = (len);                  
      for (_i = 0; _i < _l; _i++)  *_to++ = *_from++;           
   }                                
#define drnd()   ((float) rand() / (float) BIGRND)
#define dpn1()   ((drnd() * 2.0) - 1.0)


#define BIGRND  32767.0
#define ETA     0.01
#define MOMENTUM 0.9
#define RAND_MAX BIGRND
#define NUM_THREAD 4

using namespace std;

struct BPNN {
  int input_n;
  int hidden_n;
  int output_n;
  float *input_units;
  float *hidden_units;
  float *output_units;
  float *hidden_delta;
  float *output_delta;
  float *target;
  float **input_weights;
  float **hidden_weights;
  float **input_prev_weights;
  float **hidden_prev_weights;
};

void bpnn_feedforward(BPNN *net) {
  int in, hid, out;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
                    net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
                    net->hidden_weights, hid, out);
}

void bpnn_initialize(int seed) {
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}

void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
                           float **who, float *hidden, float *err) {
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

void bpnn_output_error(float *delta, float *target, float *output, int nj,
                           float *err) {
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}

void bpnn_randomize_weights(float **w, int m, int n) {
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = (float)rand() / RAND_MAX;
      //w[i][j] = dpn1();
    }
  }
}


void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
                             float **w, float **oldw) {
  float new_dw;
  int k, j;
  ly[0] = 1.0;

#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for shared(oldw, w, delta) private(j, k, new_dw) \
                                firstprivate(ndelta, nly, momentum)
#endif
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}

void bpnn_train(BPNN *net, float *eo, float *eh) {
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
                    net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
                    net->hidden_weights, hid, out);

  /*** Compute error on output and hidden units. ***/
  bpnn_output_error(net->output_delta, net->target,
                        net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta,
                        out, net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units,
                          hid, net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units,
                          in, net->input_weights, net->input_prev_weights);
}

BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out) {
  BPNN *newnet;

  newnet = (BPNN *)malloc(sizeof(BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return NULL;
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);
  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
  return newnet;
}
float squash(float x) {
  return (1.0 / (1.0 + exp(-x)));
}
float *alloc_1d_dbl(int n) {
  float *new_t;

  new_t = (float *)malloc((unsigned)(n * sizeof(float)));
  if (new_t == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return NULL;
  }
  return new_t;
}
float **alloc_2d_dbl(int m, int n) {
  int i;
  float **new_t;

  new_t = (float **)malloc((unsigned)(m * sizeof(float *)));
  if (new_t == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return NULL;
  }

  for (i = 0; i < m; i++) {
    new_t[i] = alloc_1d_dbl(n);
  }
  return new_t;
}
void bpnn_randomize_row(float *w, int m) {
  int i;
  for (i = 0; i <= m; i++) {
    w[i] = 0.1;  // w[i] = dpn1();
  }
}

void bpnn_zero_weights(float **w, int m, int n) {
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}


BPNN *bpnn_create(int n_in, int n_hidden, int n_out) {
  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return newnet;
}

void bpnn_free(BPNN *net) {
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *)net->input_units);
  free((char *)net->hidden_units);
  free((char *)net->output_units);

  free((char *)net->hidden_delta);
  free((char *)net->output_delta);
  free((char *)net->target);

  for (i = 0; i <= n1; i++) {
    free((char *)net->input_weights[i]);
    free((char *)net->input_prev_weights[i]);
  }
  free((char *)net->input_weights);
  free((char *)net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *)net->hidden_weights[i]);
    free((char *)net->hidden_prev_weights[i]);
  }
  free((char *)net->hidden_weights);
  free((char *)net->hidden_prev_weights);

  free((char *)net);
}

void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2) {
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for shared(conn, n1, n2, l1) private(k, j) \
                           reduction(+: sum) schedule(static)
#endif
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {
    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) {  ///Sorry for the indentation for the pre#ifdef#else#endif
      sum += conn[k][j] * l1[k]; 
    }
    l2[j] = squash(sum);
  }
}

void bpnn_save(BPNN *net, char *filename) {
  int n1, n2, n3, i, j, memcnt;
  float dvalue, **w;
  char *mem;
  ///add//
  FILE *pFile;
  pFile = fopen(filename, "w+");
  ///////
  /*
  if ((fd = creat(filename, 0644)) == -1) {
    printf("BPNN_SAVE: Cannot create '%s'\n", filename);
    return;
  }
  */



  n1 = net->input_n;
  n2 = net->hidden_n;
  n3 = net->output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  //fflush(stdout);

  //write(fd, (char *) &n1, sizeof(int));
  //write(fd, (char *) &n2, sizeof(int));
  //write(fd, (char *) &n3, sizeof(int));
  fwrite(&n1, sizeof(char), sizeof(char), pFile);
  fwrite(&n2, sizeof(char), sizeof(char), pFile);
  fwrite(&n3, sizeof(char), sizeof(char), pFile);

  memcnt = 0;
  w = net->input_weights;
  mem = (char *)malloc((unsigned)((n1 + 1) * (n2 + 1) * sizeof(float)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n1 + 1) * (n2 + 1) * sizeof(float));
  fwrite(mem, sizeof(float), (unsigned)((n1 + 1) * (n2 + 1) * sizeof(float)),
         pFile);
  free(mem);

  memcnt = 0;
  w = net->hidden_weights;
  mem = (char *)malloc((unsigned)((n2 + 1) * (n3 + 1) * sizeof(float)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n2 + 1) * (n3 + 1) * sizeof(float));
  fwrite(mem, sizeof(float), (unsigned)((n2 + 1) * (n3 + 1) * sizeof(float)),
         pFile);
  free(mem);

  fclose(pFile);
  return;
}


BPNN *bpnn_read(char *filename) {
  char *mem;
  BPNN *new_t;
  int fd, n1, n2, n3, i, j, memcnt;

  if ((fd = open(filename, 0, 0644)) == -1) {
    return NULL;
  }

  printf("Reading '%s'\n", filename);
  //fflush(stdout);

  read(fd, (char *)&n1, sizeof(int));
  read(fd, (char *)&n2, sizeof(int));
  read(fd, (char *)&n3, sizeof(int));
  new_t = bpnn_internal_create(n1, n2, n3);

  printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  printf("Reading input weights...");
  //fflush(stdout);

  memcnt = 0;
  mem = (char *)malloc((unsigned)((n1 + 1) * (n2 + 1) * sizeof(float)));
  read(fd, mem, (n1 + 1) * (n2 + 1) * sizeof(float));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(new_t->input_weights[i][j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);

  printf("Done\nReading hidden weights...");

  memcnt = 0;
  mem = (char *)malloc((unsigned)((n2 + 1) * (n3 + 1) * sizeof(float)));
  read(fd, mem, (n2 + 1) * (n3 + 1) * sizeof(float));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(new_t->hidden_weights[i][j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);
  close(fd);

  printf("Done\n");

  bpnn_zero_weights(new_t->input_prev_weights, n1, n2);
  bpnn_zero_weights(new_t->hidden_prev_weights, n2, n3);

  return new_t;
}


BPNN *stochastic_backprop(BPNN *net, float **pattern, int n_p, int *label,int n_l, int nepoch) {
  int epoch, success = 0;
  float p_err, h_err, net_err;
  float sum = 0.0, total_p_err = 0.0, total_h_err = 0.0, total_net_err = 0.0, tp_acc = 0.0;

  for (epoch = 0; epoch < nepoch; epoch++) {
    for (int i = 0; i < n_l; i++) {
      bpnn_feedforward(net);  // Feed forward
      bpnn_train(net, &p_err, &h_err);  // Calculating the net error
      sum = 0.0;
      for (int j = 1; j <= net->output_n; j++)
        sum += ABS(net->output_delta[j]);  // Sum all the errors
      total_p_err += p_err;
      total_h_err += h_err;
      total_net_err += sum;
    }  // END: for (i = 0; i < NUM_TRAIN; i++)

    tp_acc = int(100.0 - (100.0 * total_net_err));
    printf("written\n");
    /*
    if (tp_acc > 90) success = 1;
    else success = 0;
    */

  }  // END: for (epoch = 0; epoch < NUM_EPOCH; epoch++)

  return net;
}

// Driver
int main() {
  int nepoch, n_in, n_l, n_h, success;
  BPNN *net;
  float **pattern, **label;
  int i, j;

  // Number of epochs
  nepoch = 50;
  printf("Initalizing pattern and label...\n");
  n_in = n_h = n_l = 5;

  pattern = (float **)malloc(n_in * sizeof(float *));
  label = (float **)malloc(n_l * sizeof(float *));

  for (i = 0; i < n_in; i++) {
    pattern[i] = (float *)malloc(5 * sizeof(float));
  }
  for (i = 0; i < n_l; i++) {
    label[i] = (float *)malloc(5 * sizeof(float));
  }

  pattern[0][0] = 0.2382815553434827;
  pattern[0][1] = 0.30468151692546126;
  pattern[0][2] = 0.7759569510898412;
  pattern[0][3] = 0.7932094753403818;
  pattern[0][4] = 0.37292972578394113;

  pattern[1][0] = 0.41007816587322934;
  pattern[1][1] = 0.017979590390179826;
  pattern[1][2] = 0.40862087379027755;
  pattern[1][3] = 0.57156660758823529;
  pattern[1][4] = 0.38656907464547016;

  pattern[2][0] = 0.91007816587322934;
  pattern[2][1] = 0.627979590390179826;
  pattern[2][2] = 0.51862087379027755;
  pattern[2][3] = 0.97156660758823529;
  pattern[2][4] = 0.68656907464547016;

  // Set the pattern label
  label[0][0] = 0.0;
  label[0][1] = 0.0;
  label[0][2] = 0.0;
  label[0][3] = 0.0;
  label[0][4] = 1.0;  // Label set to class 4

  label[1][0] = 1.0;
  label[1][1] = 0.0;
  label[1][2] = 0.0;
  label[1][3] = 0.0;
  label[1][4] = 0.0;  // Label set to class 1

  label[2][0] = 0.0;
  label[2][1] = 1.0;
  label[2][2] = 0.0;
  label[2][3] = 0.0;
  label[2][4] = 0.0;  // Label set to class 2

  // Creates a new network from scratch
  net = bpnn_create(n_in, n_h, n_l);

  progressive_bpp = (float *)malloc(nepoch * sizeof(float));
  // Perform the stochastic backpropagation
  stochastic_backprop(net, pattern, 5, label, 5, nepoch);
  bpnn_free(net);
  return 0;
}