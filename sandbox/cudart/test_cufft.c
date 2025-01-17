#include "cuew.h"
#include "cuew_cudart.h"
#include "cuew_cufft.h"

#include <stdio.h>
#include <stdlib.h>


#define NX 256
#define BATCH 10
//#define RANK 1

// Assume C11
#define CUDART_CHECK(expr) do { \
  cudaError_t err = (expr); \
  if (err != cudaSuccess) { \
    printf("cudaRT err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
  } \
} while (0)

#define CUFFT_CHECK(expr) do { \
  cufftResult err = (expr); \
  if (err != CUFFT_SUCCESS) { \
    printf("cufft err at %s:%s:%d. code = %d\n", __FILE__, __func__, __LINE__, err); \
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

  ret = cuewInitCUFFT(NULL);
  printf("cuewInitCUFFT ret = %d\n", ret);

  if (ret != 0) {
    printf("cufft init failed.\n");
    exit(-1);
  }

  cufftHandle plan;

  int new_size = 1024; // FIXME
  CUFFT_CHECK(cufftPlan1d(&plan, new_size, CUFFT_C2C, 1));

  cufftComplex *data;

  CUDART_CHECK(cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH));

  /*
  cufftPlanMany(&plan, RANK, NX, &iembed, istride, idist,
      &oembed, ostride, odist, CUFFT_C2C, BATCH);
  */

  CUFFT_CHECK(cufftExecC2C(plan, data, data, CUFFT_FORWARD));

  CUDART_CHECK(cudaDeviceSynchronize());
  CUFFT_CHECK(cufftDestroy(plan));

  CUDART_CHECK(cudaFree(data));

  printf("cufft OK!\n");

  return 0;
}
