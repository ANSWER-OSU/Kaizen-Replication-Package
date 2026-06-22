/**
 * OpenMP Implementation of the above CUDA program
 */

#ifndef __OPENMP_MACRO_H__
#define __OPENMP_MACRO_H__

#define CHECK_OPENMP(call) {                                                \
    int err = call;                                                         \
    if( 0 != err) {                                                         \
        fprintf(stderr, "OpenMP error: %s = %d at (%s:%d)\n", #call,        \
                err, __FILE__, __LINE__);                                  \
        exit(EXIT_FAILURE);                                                 \
    }}

#define CHECK_ERROR(errorMessage) {                                         \
    int err = omp_get_error();                                              \
    if( 0 != err) {                                                         \
        fprintf(stderr, "OpenMP error: %s in file '%s' in line %i : %s.\n", \
                errorMessage, __FILE__, __LINE__, err );                   \
        exit(EXIT_FAILURE);                                                 \
    }}
#endif