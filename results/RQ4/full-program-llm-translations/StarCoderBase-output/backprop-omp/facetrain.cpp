#include <iostream>

int layer_size = 0;

void backprop_face()
{
  // Use OpenMP to replace CUDA runtime semantics
  #pragma omp parallel
  {
    // Replace CUDA APIs with OpenMP counterparts
    #pragma omp sections
    {
      #pragma omp section
      {
        printf("Input layer size : %d\n", layer_size);
      }
      #pragma omp section
      {
        load();
      }
      #pragma omp section
      {
        printf("Starting training kernel\n");
      }
      #pragma omp section
      {
        train_kernel();
      }
      #pragma omp section
      {
        free();
      }
      #pragma omp section
      {
        printf("\nFinish the training for one iteration\n");
      }
    }
  }
}