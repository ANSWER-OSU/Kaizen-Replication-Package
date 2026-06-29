#include <omp.h>
void map(const int *pages,
         const float *page_ranks,
               float *maps,
         const unsigned int *noutlinks,
         const int n)
{
  #pragma omp parallel for
  for (int i = 0; i < n; ++i)
  {
    float outbound_rank = page_ranks[i] / (float)noutlinks[i];
    for (int j = 0; j < n; ++j)
    {
      maps[(size_t)i*n+j] = pages[(size_t)i*n+j] * outbound_rank;
    }
  }
}

void reduce(float *page_ranks, const float *maps, const int n, float *dif)
{
  #pragma omp parallel for
  for (int j = 0; j < n; ++j)
  {
    float new_rank = 0.0f;
    float old_rank = page_ranks[j];

    for (int i = 0; i < n; ++i)
    {
      new_rank += maps[(size_t)i*n + j];
    }

    new_rank = ((1.f-D_FACTOR)/n) + (D_FACTOR*new_rank);
    dif[j] = fmaxf(fabsf(new_rank - old_rank), dif[j]);
    page_ranks[j] = new_rank;
  }
}