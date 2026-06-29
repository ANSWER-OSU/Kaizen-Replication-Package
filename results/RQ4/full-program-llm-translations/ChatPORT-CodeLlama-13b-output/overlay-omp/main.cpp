template<typename T>
__global__ void DetectionOverlayBox(
  const T*__restrict__ input,
        T*__restrict__  output,
  int imgWidth, int imgHeight,
  int x0, int y0, int boxWidth, int boxHeight,
  const float4 color) 
{
  const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int box_y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if( box_x >= boxWidth || box_y >= boxHeight ) return;
  
  const int x = box_x + x0;
  const int y = box_y + y0;
  
  if( x >= imgWidth || y >= imgHeight ) return;
  
  T px = input[ y * imgWidth + x ];
  
  const float alpha = color.w / 255.0f;
  const float ialph = 1.0f - alpha;
  
  px.x = alpha * color.x + ialph * px.x;
  px.y = alpha * color.y + ialph * px.y;
  px.z = alpha * color.z + ialph * px.z;
  
  output[y * imgWidth + x] = px;
}

template<typename T>
int DetectionOverlay(
  T* input, T* output, uint32_t width, uint32_t height, 
  Box *detections, int numDetections, float4 colors )
{
  if( !input || !output || width == 0 || height == 0 || !detections || numDetections == 0)
    return 1;
  		
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  for( int n=0; n < numDetections; n++ )
  {
    const int boxWidth = detections[n].width;
    const int boxHeight = detections[n].height;
    const int boxLeft = detections[n].left;
    const int boxTop = detections[n].top;
    
    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim((boxWidth+7)/8, (boxHeight+7)/8);
    DetectionOverlayBox<T><<<gridDim, blockDim>>>(
      input, output, width, height, boxLeft, boxTop, boxWidth, boxHeight, colors);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  return 0;
}