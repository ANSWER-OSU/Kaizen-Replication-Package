#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define D_FACTOR (0.85f)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// default values
const int max_iter = 1000;
const float threshold = 1e-16f;

void map_omp(int *pages, float *page_ranks, float *maps, unsigned int *noutlinks, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (pages[i*n + j] != 0) {
                maps[i*n + j] = page_ranks[i] / (float) noutlinks[i];
            }
        }
    }
}

void reduce_omp(float *page_ranks, float *maps, int n, float *difs) {
    #pragma omp parallel for shared(page_ranks, maps)
    for (int j = 0; j < n; ++j) {
        float new_rank = 0.0f;
        for (int i = 0; i < n; ++i) {
            new_rank += maps[i*n + j];
        }
        new_rank = ((1.f - D_FACTOR) / n) + (D_FACTOR * new_rank);
        difs[j] = fmaxf(fabsf(new_rank - page_ranks[j]), difs[j]);
        page_ranks[j] = new_rank;
    }
}

int *random_pages(int n, unsigned int *noutlinks, int divisor) {
    int *pages = (int*) malloc((size_t)n * n * sizeof(int)); // matrix 1 means link from j->i

    if (divisor <= 0) {
        fprintf(stderr, "ERROR: Invalid divisor '%d' for random initialization, divisor should be greater or equal to 1\n", divisor);
        exit(1);
    }

    for (int i = 0; i < n; ++i) {
        noutlinks[i] = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j && (abs(rand()) % divisor == 0)) {
                pages[i*n + j] = 1;
                noutlinks[i] += 1;
            }
        }

        // the case with no outlinks is avoided
        if (noutlinks[i] == 0) {
            int k;
            do {
                k = abs(rand()) % n;
            } while (k == i);
            pages[i*n + k] = 1;
            noutlinks[i] = 1;
        }
    }

    return pages;
}

void init_array(float *a, int n, float val) {
    for (int i = 0; i < n; ++i) {
        a[i] = val;
    }
}

void usage(char *argv[]) {
    fprintf(stderr, "Usage: %s [-n number of pages] [-i max iterations]\n", argv[0]);
}

static struct option size_opts[] =
{
  /* name, has_tag, flag, val*/
  {"number of pages", 1, NULL, 'n'},
  {"max number of iterations", 1, NULL, 'i'},
  {"minimum threshold", 1, NULL, 't'},
  {"divisor for zero density", 1, NULL, 'q'},
  { 0, 0, 0}
};

float maximum_dif(float *difs, int n) {
    float max = 0.0f;
    for (int i = 0; i < n; ++i) {
        max = difs[i] > max ? difs[i] : max;
    }
    return max;
}

int main(int argc, char *argv[]) {
    int *pages;
    float *maps;
    float *page_ranks;
    unsigned int *noutlinks;
    int t;
    float max_diff, max_diff_ref;

    int i = 0;
    int j;//, k;
    int n = 1000;
    int iter = max_iter;
    float thresh = threshold;
    int divisor = 2;

    int opt, opt_index = 0;
    while ((opt = getopt_long(argc, argv, "::n:i:t:q:", size_opts, &opt_index)) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                break;
            case 'i':
                iter = atoi(optarg);
                break;
            case 't':
                thresh = atof(optarg);
                break;
            case 'q':
                divisor = atoi(optarg);
                break;
            default:
                usage(argv);
                exit(EXIT_FAILURE);
        }
    }

    size_t rank_size = (size_t)n * sizeof(float);
    size_t map_size = (size_t)n * n * sizeof(float);
    size_t page_size = (size_t)n * n * sizeof(int);
    size_t link_size = (size_t)n * sizeof(unsigned int);

    page_ranks = (float*)malloc(rank_size);
    maps = (float*)malloc(map_size);
    noutlinks = (unsigned int*)malloc(link_size);

    init_array(page_ranks, n, 1.0f / (float) n);

    float *diffs;
    diffs = (float*) malloc(rank_size);

    pages = random_pages(n, noutlinks, divisor);
    //cudaMemcpy(pages, random_pages, page_size, cudaMemcpyHostToDevice);

    max_diff = max_diff_ref = 99.0f;

    for (i = 0; i < n; ++i) {
        noutlinks[i] = 0;
    }

    init_array(diffs, n, 0.0f);

    int num_threads = std::thread::hardware_concurrency();
    omp_set_num_threads(num_threads);

    double ktime = 0.0;
    printf("Threshold: %f\n", thresh);

    for (t = 1; t <= iter && max_diff >= thresh; ++t) {
        auto start = std::chrono::high_resolution_clock::now();

        map_omp(pages, page_ranks, maps, noutlinks, n);
        reduce_omp(page_ranks, maps, n, diffs);

        auto end = std::chrono::high_resolution_clock::now();
        ktime += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        max_diff = maximum_dif(diffs, n);
    }
    //cudaMemcpy(maps, d_maps, page_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(page_ranks, d_page_ranks, rank_size, cudaMemcpyDeviceToHost);

    max_diff = maximum_dif(diffs, n);
    fprintf(stderr, "Max difference %.16f is reached at iteration %d\n", max_diff, t);
    printf("\"Options\": \"-n %d -i %d -t %f\". Total kernel execution time: %lf (s)\n", n, iter, thresh, ktime);

    memset(diffs, 0, rank_size);
    for (t = 1; t <= iter && max_diff_ref >= thresh; ++t) {
        max_diff_ref = maximum_dif(diffs, n);
    }
    int ok = fabsf(max_diff - max_diff_ref) < 1e-3f;
    printf("%s\n", ok == 0? "PASS" : "FAIL");

    free(pages);
    free(maps);
    free(page_ranks);
    free(diffs);

    return 0;
}