#include "cudart.h"

#include <stdio.h>

int main(int argc, char **argv) {
  int ret = InitCUDART();
  printf("ret = %d\n", ret);

  if (ret == 0) {
    printf("addr %p\n", cudaMalloc);
  }

  char *ptr;

  cudaMalloc(&ptr, 1024);
  cudaFree(ptr);

  return 0;
}
