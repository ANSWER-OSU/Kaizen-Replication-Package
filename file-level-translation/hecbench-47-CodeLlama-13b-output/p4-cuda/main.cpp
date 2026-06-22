#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <omp.h>

#include "params.h"
#include "reference.h"

#define THREADS_PER_BLOCK 256
#define ITERATIONS 1000

void postprocess (
          float *cls_input,
          float *box_input,
          float *dir_cls_input,
          float *anchors,
          float *anchor_bottom_heights,
          float *bndbox_output,
          float *score_output,
          int *object_counter,
          const float min_x_range,
          const float max_x_range,
          const float min_y_range,
          const float max_y_range,
          const int feature_x_size,
          const int feature_y_size,
          const int num_anchors,
          const int num_classes,
          const int num_box_values,
          const float score_thresh,
          const float dir_offset
        )
{
  int loc_index = omp_get_thread_num();
  int ith_anchor = omp_get_thread_num();
  if (ith_anchor >= num_anchors) return;

  int col = loc_index % feature_x_size;
  int row = loc_index / feature_x_size;
  float x_offset = min_x_range + col * (max_x_range - min_x_range) / (feature_x_size - 1);
  float y_offset = min_y_range + row * (max_y_range - min_y_range) / (feature_y_size - 1);
  int cls_offset = loc_index * num_anchors * num_classes + ith_anchor * num_classes;
  float dev_cls[2] = {-1.f, 0.f};

  const float *scores = cls_input + cls_offset;
  float max_score = 1.f / (1.f + expf(-scores[0]));
  int cls_id = 0;
  for (int i = 1; i < num_classes; i++) {
    float cls_score = 1.f / (1.f + expf(-scores[i]));
    if (cls_score > max_score) {
      max_score = cls_score;
      cls_id = i;
    }
  }
  dev_cls[0] = static_cast<float>(cls_id);
  dev_cls[1] = max_score;

  if (dev_cls[1] >= score_thresh)
  {
    const int box_offset = loc_index * num_anchors * num_box_values + ith_anchor * num_box_values;
    const int dir_cls_offset = loc_index * num_anchors * 2 + ith_anchor * 2;
    const float *anchor_ptr = anchors + ith_anchor * 4;
    const float z_offset = anchor_ptr[2] / 2 + anchor_bottom_heights[ith_anchor / 2];
    const float anchor[7] = {x_offset, y_offset, z_offset, anchor_ptr[0], anchor_ptr[1], anchor_ptr[2], anchor_ptr[3]};
    float *box_encodings = box_input + box_offset;

    const float xa = anchor[0];
    const float ya = anchor[1];
    const float za = anchor[2];
    const float dxa = anchor[3];
    const float dya = anchor[4];
    const float dza = anchor[5];
    const float ra = anchor[6];
    const float diagonal = sqrtf(dxa * dxa + dya * dya);
    box_encodings[0] = box_encodings[0] * diagonal + xa;
    box_encodings[1] = box_encodings[1] * diagonal + ya;
    box_encodings[2] = box_encodings[2] * dza + za;
    box_encodings[3] = expf(box_encodings[3]) * dxa;
    box_encodings[4] = expf(box_encodings[4]) * dya;
    box_encodings[5] = expf(box_encodings[5]) * dza;
    box_encodings[6] = box_encodings[6] + ra;

    const int dir_label = dir_cls_input[dir_cls_offset] > dir_cls_input[dir_cls_offset + 1] ? 0 : 1;
    const float period = (float)M_PI;
    const float val = box_input[box_offset + 6] - dir_offset;
    const float dir_rot = val - floorf(val / (period + 1e-8f)) * period;
    const float yaw = dir_rot + dir_offset + period * dir_label;

    int resCount = omp_fetch_add(object_counter, 1);
    bndbox_output[0] = resCount+1;
    float *data = bndbox_output + resCount * 9;
    data[0] = box_input[box_offset];
    data[1] = box_input[box_offset + 1];
    data[2] = box_input[box_offset + 2];
    data[3] = box_input[box_offset + 3];
    data[4] = box_input[box_offset + 4];
    data[5] = box_input[box_offset + 5];
    data[6] = yaw;
    data[7] = dev_cls[0];
    data[8] = box_offset;
    score_output[resCount] = max_score;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Params p;  // constant values defined in params.h
  const float min_x_range = p.min_x_range;
  const float max_x_range = p.max_x_range;
  const float min_y_range = p.min_y_range;
  const float max_y_range = p.max_y_range;
  const int feature_x_size = p.feature_x_size;
  const int feature_y_size = p.feature_y_size;
  const int num_anchors = p.num_anchors;
  const int num_classes = p.num_classes;
  const int num_box_values = p.num_box_values;
  const float score_thresh = p.score_thresh;
  const float dir_offset = p.dir_offset;
  const int len_per_anchor = p.len_per_anchor;
  const int num_dir_bins = p.num_dir_bins;
  
  const int feature_size = feature_x_size * feature_y_size;
  const int feature_anchor_size = feature_size * num_anchors;
  const int cls_size = feature_anchor_size * num_classes;
  const int box_size = feature_anchor_size * num_box_values;
  const int dir_cls_size = feature_anchor_size * num_dir_bins;
  const int bndbox_size = feature_anchor_size * 9;

  const int cls_size_byte = cls_size * sizeof(float);
  const int box_size_byte = box_size * sizeof(float);
  const int dir_cls_size_byte = dir_cls_size * sizeof(float);
  const int bndbox_size_byte = bndbox_size * sizeof(float);
  const int score_size_byte = feature_anchor_size * sizeof(float);

  // input of the post-process kernel
  float *h_cls_input = (float*) malloc (cls_size_byte);
  float *h_box_input = (float*) malloc (box_size_byte);
  float *h_dir_cls_input = (float*) malloc (dir_cls_size_byte);

  // output of the post-process kernel
  float *h_bndbox_output = (float*) malloc (bndbox_size_byte);
  float *h_score_output = (float*) malloc (score_size_byte);

  // random values
  srand(123);
  for (int i = 0; i < cls_size; i++)  h_cls_input[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < box_size; i++)  h_box_input[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < dir_cls_size; i++)  h_dir_cls_input[i] = rand() / (float)RAND_MAX;
  
  float *d_cls_input, *d_box_input, *d_dir_cls_input, *d_bndbox_output, *d_score_output;
  float *d_anchors, *d_anchor_bottom_heights;
  int *d_object_counter;

  omp_init_lock(&d_anchor_bottom_heights);
  d_cls_input = (float *) malloc (sizeof(float) * cls_size); 
  d_box_input = (float *) malloc (sizeof(float) * box_size); 
  d_dir_cls_input = (float *) malloc (sizeof(float) * dir_cls_size); 
  d_bndbox_output = (float *) malloc (sizeof(float) * bndbox_size); 
  d_score_output = (float *) malloc (sizeof(float) * score_size); 

  omp_memcpy(d_cls_input, h_cls_input, cls_size_byte, cudaMemcpyHostToDevice);
  omp_memcpy(d_dir_cls_input, h_dir_cls_input, dir_cls_size_byte, cudaMemcpyHostToDevice);

  d_anchors = (float *) malloc (sizeof(float) * num_anchors * len_per_anchor); 
  d_anchor_bottom_heights = (float *) malloc (sizeof(float) * num_classes); 
  d_object_counter = (int *) malloc (sizeof(int)); 

  omp_memcpy(d_anchors, p.anchors,
             num_anchors * len_per_anchor * sizeof(float), cudaMemcpyHostToDevice);

  omp_memcpy(d_anchor_bottom_heights, p.anchor_bottom_heights,
             num_classes * sizeof(float), cudaMemcpyHostToDevice);

  double time = 0.0;

  omp_set_dynamic(0);
  omp_set_num_threads(ITERATIONS);
#pragma omp parallel for default(none) shared(d_cls_input, d_box_input, d_dir_cls_input, d_anchors, \
                                              d_anchor_bottom_heights, d_bndbox_output, d_score_output, \
                                              d_object_counter, min_x_range, max_x_range, \
                                              min_y_range, max_y_range, feature_x_size, feature_y_size, \
                                              num_anchors, num_classes, num_box_values, score_thresh, \
                                              dir_offset)
  for (int i = 0; i < repeat; i++) {
    omp_memcpy(d_box_input, h_box_input, box_size_byte, cudaMemcpyHostToDevice);
    omp_memset(d_object_counter, 0, sizeof(int));

    omp_get_wtime();

    postprocess(d_cls_input,
                d_box_input,
                d_dir_cls_input,
                d_anchors,
                d_anchor_bottom_heights,
                d_bndbox_output,
                d_score_output,
                d_object_counter,
                min_x_range,
                max_x_range,
                min_y_range,
                max_y_range,
                feature_x_size,
                feature_y_size,
                num_anchors,
                num_classes,
                num_box_values,
                score_thresh,
                dir_offset);

    omp_get_wtime();
    time += omp_get_wtime() - start;
  }

  printf("Average execution time of postprocess kernel: %f (us)\n", (time * 1e-3f) / repeat);

  int bndbox_num;
  omp_memcpy(&bndbox_num, d_object_counter, sizeof(int), cudaMemcpyDeviceToHost);
  omp_memcpy(h_bndbox_output, d_bndbox_output, bndbox_size_byte, cudaMemcpyDeviceToHost);
  omp_memcpy(h_score_output, d_score_output, score_size_byte, cudaMemcpyDeviceToHost);

  verify (bndbox_num,
          h_cls_input,
          h_box_input,
          h_dir_cls_input,
          p.anchors,
          p.anchor_bottom_heights,
          h_bndbox_output,
          h_score_output,
          min_x_range,
          max_x_range,
          min_y_range,
          max_y_range,
          feature_size,
          feature_x_size,
          feature_y_size,
          num_anchors,
          num_classes,
          num_box_values,
          score_thresh,
          dir_offset);

  omp_destroy_lock(&d_anchor_bottom_heights);
  free(d_anchors);
  free(d_anchor_bottom_heights);
  free(d_object_counter);
  free(d_cls_input);
  free(d_box_input);
  free(d_dir_cls_input);
  free(d_bndbox_output);
  free(d_score_output);

  free(h_cls_input);
  free(h_box_input);
  free(h_dir_cls_input);
  free(h_score_output);
  free(h_bndbox_output);

  return 0;
}