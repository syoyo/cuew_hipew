#include "cuew_cufft.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int ret = cuewInitCUFFT();
  printf("ret = %d\n", ret);

  if (ret != 0) {
    printf("cufft init failed.\n");
    exit(-1);
  }

  char *ptr;

  cudaMalloc((void **)&ptr, 1024);
  cudaFree(ptr);

  return 0;
}
