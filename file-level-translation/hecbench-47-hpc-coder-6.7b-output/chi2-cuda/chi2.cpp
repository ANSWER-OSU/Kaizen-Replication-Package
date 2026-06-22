#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <random>
#include "reference.h"

void chi_kernel(
  const unsigned int rows,
  const unsigned int cols,
  const int cRows,
  const int contRows,
  const unsigned char *__restrict__ snpdata,
  float *__restrict__ results)
{  
  int tid;
  #pragma omp parallel shared(tid, results, rows, cols, cRows, contRows, snpdata) private(m, n, p, y, tot_cases, tot_controls, total, chisquare, exp, Cexpected, Conexpected, numerator1, numerator2, cases, controls)
  {
    #pragma omp for
    for( tid = 0 ; tid < cols ; tid++ ) {
      unsigned char y;
      int m, n;
      unsigned int p = 0;
      int tot_cases = 1;
      int tot_controls= 1;
      int total = 1;
      float chisquare = 0.0f;
      float exp[3];      
      float Conexpected[3];      
      float Cexpected[3];
      float numerator1;
      float numerator2;

      int cases[3] = {1,1,1};
      int controls[3] = {1,1,1};

      // read cases: each thread reads a column of snpdata matrix
      for ( m = 0 ; m < cRows ; m++ ) {
        y = snpdata[m * (size_t)cols + tid];
        if ( y == '0') { cases[0]++; }
        else if ( y == '1') { cases[1]++; }
        else if ( y == '2') { cases[2]++; }
      }

      // read controls: each thread reads a column of snpdata matrix
      for ( n = cRows ; n < cRows + contRows ; n++ ) {
        y = snpdata[n * cols + tid];
        if ( y == '0' ) { controls[0]++; }
        else if ( y == '1') { controls[1]++; }
        else if ( y == '2') { controls[2]++; }
      }
      
      #pragma omp atomic
      for( p = 0 ; p < 3; p++ ) {
        tot_cases += cases[p];
        tot_controls += controls[p];
      }
      #pragma omp atomic
      total = tot_cases + tot_controls;

      #pragma omp atomic
      for( p = 0 ; p < 3; p++ ) {
        exp[p] = (float)cases[p] + controls[p]; 
        Cexpected[p] = tot_cases * exp[p] / total;
        Conexpected[p] = tot_controls * exp[p] / total;
        numerator1 = (float)cases[p] - Cexpected[p];
        numerator2 = (float)controls[p] - Conexpected[p];
        chisquare += numerator1 * numerator1 / Cexpected[p] +
                     numerator2 * numerator2 / Conexpected[p];
      }
      
      #pragma omp atomic
      results[tid] = chisquare; 
    }
  }
}

int main(int argc, char* argv[]) {

  if (argc != 7) {
    printf("Usage: %s <rows> <cols> <cases> <controls> "
           "<threads> <repeat>\n", argv[0]);
    return 1;
  }

  unsigned int rows = atoi(argv[1]);
  unsigned int cols = atoi(argv[2]);
  int ncases = atoi(argv[3]);
  int ncontrols = atoi(argv[4]);
  int nthreads = atoi(argv[5]);
  int repeat = atoi(argv[6]);

  printf("Individuals=%d SNPs=%d cases=%d controls=%d nthreads=%d\n",
         rows,cols,ncases,ncontrols,nthreads);

  size_t size = (size_t)rows * (size_t)cols;
  printf("Size of the data = %lu\n",size);

  // allocate SNP host data 
  size_t snpdata_size = sizeof(unsigned char) * size;
  size_t result_size = sizeof(float) * cols;

  unsigned char *dataT = (unsigned char*)malloc(snpdata_size);
  float* h_results = (float*) malloc(result_size);
  float* cpu_results = (float*) malloc(result_size);

  if(dataT == NULL || h_results == NULL || cpu_results == NULL) {
    printf("ERROR: Memory for data not allocated.\n");
    if (dataT) free(dataT);
    if (h_results) free(h_results);
    return 1;
  }

  std::mt19937 gen(19937); // mersenne_twister_engin
  std::uniform_int_distribution<> distrib(0, 2);
  for (size_t i = 0; i < snpdata_size; i++) {
    dataT[i] = distrib(gen) + '0';
  }

  // allocate SNP device data
  unsigned char *d_data;
  float *d_results;

  #pragma omp target data map(to: dataT[0:snpdata_size]) map(from: d_results[0:cols])
  {
    d_data = dataT;
    d_results = h_results;
  }

  unsigned jobs = cols;

  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp master
    {
      #pragma omp taskloop
      for (int i = 0; i < repeat; ++i) {
        chi_kernel(rows,cols,ncases,ncontrols,d_data,d_results);
      }
    }
  }

  #pragma omp target update from(d_results[0:cols])

  #pragma omp parallel num_threads(1)
  {
    #pragma omp master
    {
      #pragma omp task
      cpu_kernel(rows,cols,ncases,ncontrols,dataT,cpu_results);
    }
  }

  #pragma omp parallel num_threads(1)
  {
    #pragma omp master
    {    
      #pragma omp taskwait
      for(unsigned int k = 0; k < jobs; k++) {
        if (fabs(cpu_results[k] - d_results[k]) > 1e-4) {
          printf("FAIL\n");
          exit(1);
        }
      }
      printf("PASS\n");
    }
  }

  free(dataT);
  free(h_results);
  free(cpu_results);

  return 0;
}