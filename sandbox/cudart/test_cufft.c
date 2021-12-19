#include "cuew.h"
#include "cuew_cudart.h"
#include "cuew_cufft.h"

#include <stdio.h>
#include <stdlib.h>


#define NX 256
#define BATCH 10
#define RANK 1

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int ret = cuewInitCUDART();
  printf("cuewInitCUDART ret = %d\n", ret);

  if (ret != 0) {
    printf("cudart init failed.\n");
    exit(-1);
  }

  ret = cuewInitCUFFT();
  printf("cuewInitCUFFT ret = %d\n", ret);

  if (ret != 0) {
    printf("cufft init failed.\n");
    exit(-1);
  }

  cufftHandle plan;
  cufftComplex *data;

  cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);

  /*
  cufftPlanMany(&plan, RANK, NX, &iembed, istride, idist,
      &oembed, ostride, odist, CUFFT_C2C, BATCH);
  */

  cufftExecC2C(plan, data, data, CUFFT_FORWARD);

  cudaDeviceSynchronize();
  cufftDestroy(plan);


  cudaFree(data);

  return 0;
}
