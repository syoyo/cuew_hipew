#include "cuew.h"
#include "cuew_cudart.h"
#include "cuew_cublas.h"

#include <stdio.h>
#include <stdlib.h>


// Assume C11
#define CUDART_CHECK(expr) do { \
  cudaError_t err = (expr); \
  if (err != cudaSuccess) { \
    printf("cudaRT err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
  } \
} while (0)

#define CUBLAS_CHECK(expr) do { \
  cufftResult err = (expr); \
  if (err != CUFFT_SUCCESS) { \
    printf("cublas err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
  } \
} while (0)

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int ret = cuewInitCUDART();
  printf("cuewInitCUDART ret = %d\n", ret);

  if (ret != 0) {
    printf("cudart init failed.\n");
    exit(-1);
  }

  ret = cuewInitCUBLAS();
  printf("cuewInitCUBLAS ret = %d\n", ret);

  if (ret != 0) {
    printf("cublas init failed.\n");
    exit(-1);
  }

  int major=-1,minor=-1,patch=-1;
  cublasGetProperty(0 /*MAJOR_VERSION*/, &major);
  cublasGetProperty(1 /*MINOR_VERSION*/, &minor);
  cublasGetProperty(2 /*PATCH_LEVEL*/, &patch);
  printf("CUBLAS Version (Major,Minor,PatchLevel): %d.%d.%d\n", major,minor,patch);

  printf("cublas OK!\n");

  return 0;
}
