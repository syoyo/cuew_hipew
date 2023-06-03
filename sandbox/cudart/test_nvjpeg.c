#include "cuew.h"
#include "cuew_cudart.h"
#include "cuew_nvjpeg.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int ret = cuewInitCUDART(NULL);
  printf("ret = %d\n", ret);

  if (ret != 0) {
    printf("cudart init failed.\n");
    exit(-1);
  }

  /* TODO */
  printf("TODO: nvJPEG test.\n");
  exit(-1);

  return 0;
}
