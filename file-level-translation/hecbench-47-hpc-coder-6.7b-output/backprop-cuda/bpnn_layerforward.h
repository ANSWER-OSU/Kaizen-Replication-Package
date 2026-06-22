#include <omp.h>

void layerforward(
  const float* input,
        float* input_weights,
        float* hidden_partial_sum,
  const int hid) 
{
  float input_node[HEIGHT];
  float weight_matrix[HEIGHT * WIDTH];

  #pragma omp parallel for shared(input,input_weights, hidden_partial_sum, hid, input_node, weight_matrix) private(index,index_in,i,power_two,by,tx,ty)
  for(int by = 0; by < gridDim.y; by++) {
    for(int tx = 0; tx < blockDim.x; tx++) {
      for(int ty = 0; ty < blockDim.y; ty++) {
        int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

        int index_in = HEIGHT * by + ty + 1;

        if ( tx == 0 )
          input_node[ty] = input[index_in] ;

        weight_matrix[ty * WIDTH + tx] =  input_weights[index];

        weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] * input_node[ty];

        for ( int i = 1 ; i <= HEIGHT ; i=i*2){
          int power_two = i; 

          if( ty % power_two == 0 )
            weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] + weight_matrix[(ty + power_two/2)* WIDTH + tx];

        }

        input_weights[index] =  weight_matrix[ty * WIDTH + tx];

        if ( tx == 0 ) {
          hidden_partial_sum[by * hid + ty] = weight_matrix[tx* WIDTH + ty];
        }
      }
    }
  }
}