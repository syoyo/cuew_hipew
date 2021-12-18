#include "cudart.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int ret = cuewInitCUDART();
  printf("ret = %d\n", ret);

  if (ret != 0) {
    printf("cudart init failed.\n");
    exit(-1);
  }

  char *ptr;

  cudaMalloc((void **)&ptr, 1024);
  cudaFree(ptr);

  return 0;
}
