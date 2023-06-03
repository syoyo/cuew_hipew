#include "cuew.h"
#include "cuew_cudart.h"
#include "cuew_cusparse.h"

#include <stdio.h>
#include <stdlib.h>


//#define NX 256
//#define BATCH 10
//#define RANK 1

// Assume C11
#define CUDART_CHECK(expr) do { \
  cudaError_t err = (expr); \
  if (err != cudaSuccess) { \
    printf("cudaRT err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
  } \
} while (0)

#define CUSPARSE_CHECK(expr) do { \
  cusparseStatus_t err = (expr); \
  if (err != CUSPARSE_STATUS_SUCCESS) { \
    printf("cusparse err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
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

  ret = cuewInitCUSPARSE(NULL);
  printf("cuewInitCUSPARSE ret = %d\n", ret);

  if (ret != 0) {
    printf("cusparse init failed.\n");
    exit(-1);
  }

  cusparseHandle_t h;
  CUSPARSE_CHECK(cusparseCreate(&h));
  CUSPARSE_CHECK(cusparseDestroy(h));

  /* TODO: call more cusparse APIs */

  printf("cusparse OK!\n");

  return 0;
}
