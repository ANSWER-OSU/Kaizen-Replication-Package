#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define XSTR(s) STR(s)
#define STR(s) #s

struct float3 {
  float x, y, z;
};

struct Box {
  int width, height, left, top;
};

void reference(
  float3* input, float3* output, uint32_t width, uint32_t height, 
  Box *detections, int numDetections, float4 colors )
{
  for( int n=0; n < numDetections; n++ )
  {
    const int boxWidth = detections[n].width;
    const int boxHeight = detections[n].height;
    const int boxLeft = detections[n].left;
    const int boxTop = detections[n].top;
    
    for (int box_y = 0; box_y < boxHeight; box_y++) 
    {
        for (int box_x = 0; box_x < boxWidth; box_x++) 
        {
            const int x = box_x + boxLeft;
            const int y = box_y + boxTop;

            if (x < width && y < height) 
            {
                float3 px = input[y * width + x];

                const float alpha = colors.w / 255.0f;
                const float ialph = 1.0f - alpha;
                
                px.x = alpha * colors.x + ialph * px.x;
                px.y = alpha * colors.y + ialph * px.y;
                px.z = alpha * colors.z + ialph * px.z;
                
                output[y * width + x] = px;
            }
        }
    }
  }
}

void DetectionOverlay(
  float3* input, float3* output, uint32_t width, uint32_t height,
  Box *detections, int numDetections, float4 colors )
{
  if( !input || !output || width == 0 || height == 0 || !detections || numDetections == 0)
    return;

  auto start = std::chrono::steady_clock::now();

  for( int n=0; n < numDetections; n++ )
  {
    const int boxWidth = detections[n].width;
    const int boxHeight = detections[n].height;
    const int boxLeft = detections[n].left;
    const int boxTop = detections[n].top;
 
    #pragma omp parallel for collapse(2)
    for (int box_y = 0; box_y < boxHeight; box_y++) 
    {
        for (int box_x = 0; box_x < boxWidth; box_x++) 
        {
            const int x = box_x + boxLeft;
            const int y = box_y + boxTop;

            if (x < width && y < height) 
            {
                float3 px = input[y * width + x];

                const float alpha = colors.w / 255.0f;
                const float ialph = 1.0f - alpha;
                
                px.x = alpha * colors.x + ialph * px.x;
                px.y = alpha * colors.y + ialph * px.y;
                px.z = alpha * colors.z + ialph * px.z;
                
                output[y * width + x] = px;
            }
        }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);
}

int main(int argc, char* argv[]) 
{
  if (argc != 3) {
    printf("Usage: %s <width> <height>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int img_size = width * height;
  const int img_size_byte = sizeof(float3) * width * height;
  const int numDetections = img_size * 0.8f;

  srand(123);
  float3 *input = (float3*) malloc (img_size_byte);
  float3 *output = (float3*) malloc (img_size_byte);
  float3 *ref_output = (float3*) malloc (img_size_byte);

  for (int i = 0; i < img_size; i++) 
  {
    ref_output[i].x = input[i].x = rand() % 256; 
    ref_output[i].y = input[i].y = rand() % 256; 
    ref_output[i].z = input[i].z = rand() % 256; 
  }

  Box* detections = (Box*) malloc (numDetections * sizeof(Box));
  for (int i = 0; i < numDetections; i++) 
  {
    detections[i].width = 64 + rand() % 128;
    detections[i].height = 64 + rand() % 128;
    detections[i].left = rand() % (width - 64);
    detections[i].top = rand() % (height - 64);
  }

  float4 colors = make_float4(255, 204, 203, 1); 

  DetectionOverlay(input, output, width, height, detections, numDetections, colors);

  reference(input, ref_output, width, height, detections, numDetections, colors);

  bool ok = true;
  for (int i = 0; i < img_size; i++) 
    if ((fabsf(ref_output[i].x - output[i].x) > 1e-3f) ||
       (fabsf(ref_output[i].y - output[i].y) > 1e-3f) ||
       (fabsf(ref_output[i].z - output[i].z) > 1e-3f)) {
      printf("Error at index %d\n", i);
      ok = false;
      break;
    }
    
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input);
  free(output);
  free(ref_output);
  free(detections);

  return 0;
}