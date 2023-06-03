#include "cuew.h"
#include "cuew_cudart.h"
#include "cuew_cusolver.h"

#include <stdio.h>
#include <stdlib.h>


// Assume C11
#define CUDART_CHECK(expr) do { \
  cudaError_t err = (expr); \
  if (err != cudaSuccess) { \
    printf("cudaRT err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
  } \
} while (0)

#define CUSOLVER_CHECK(expr) do { \
  cusolverStatus_t err = (expr); \
  if (err != CUSOLVER_STATUS_SUCCESS) { \
    printf("cusolver err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
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

  ret = cuewInitCUSOLVER(NULL);
  printf("cuewInitCUSOLVER ret = %d\n", ret);

  if (ret != 0) {
    printf("cusolver init failed.\n");
    exit(-1);
  }

  int major=-1,minor=-1,patch=-1;
  cusolverGetProperty(0 /*MAJOR_VERSION*/, &major);
  cusolverGetProperty(1 /*MINOR_VERSION*/, &minor);
  cusolverGetProperty(2 /*PATCH_LEVEL*/, &patch);
  printf("CUSOLVER Version (Major,Minor,PatchLevel): %d.%d.%d\n", major,minor,patch);

  cusolverRfHandle_t h;

  printf("cusolver OK!\n");

  return 0;
}
