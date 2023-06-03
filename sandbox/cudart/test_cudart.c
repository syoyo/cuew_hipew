#include "cuew.h"
#include "cuew_cudart.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int ret;
  char *ptr;
  int version;
  cudaError_t errcode;

  (void)argc;
  (void)argv;

  ret = cuewInitCUDART(NULL);
  printf("cuewInitCUDART ret = %d(0 = success)\n", ret);

  if (ret != 0) {
    printf("cudart init failed.\n");
    exit(-1);
  }

  errcode = cudaRuntimeGetVersion(&version);
  if (errcode != CUDA_SUCCESS) {
    printf("cudaRuntimeGetVersion failed.\n");
    exit(-1);
  }

  printf("CUDA Runtime version %d\n", version);


  cudaMalloc((void **)&ptr, 1024);
  cudaFree(ptr);

  return 0;
}
