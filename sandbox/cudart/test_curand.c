#include "cuew.h"
#include "cuew_cudart.h"
#include "cuew_curand.h"

#include <stdio.h>
#include <stdlib.h>


// Assume C11
#define CUDART_CHECK(expr) do { \
  cudaError_t err = (expr); \
  if (err != cudaSuccess) { \
    printf("cudaRT err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
  } \
} while (0)

#define CURAND_CHECK(expr) do { \
  curandResult err = (expr); \
  if (err != CUFFT_SUCCESS) { \
    printf("curand err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
  } \
} while (0)

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int ret = cuewInitCUDART(NULL);
  printf("cuewInitCUDART ret = %d\n", ret);

  if (ret != 0) {
    printf("cudart init failed.\n");
    exit(-1);
  }

  ret = cuewInitCURAND(NULL);
  printf("cuewInitCURAND ret = %d\n", ret);

  if (ret != 0) {
    printf("curand init failed.\n");
    exit(-1);
  }

  curandState *devstate;
  curandGenerator_t gen;

  printf("curand OK!\n");

  return 0;
}
