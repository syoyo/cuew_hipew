#include <stdio.h>
#include <stdlib.h>

#include "cuew.h"

#define M 6
#define N 5
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#if 1
static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-q+1, &alpha, &m[IDX2F(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p+1, &beta, &m[IDX2F(p,q,ldm)], 1);
}

int cublas_test (void){
    //cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    //float* devPtrA;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= M; i++) {
            a[IDX2F(i,j,M)] = (float)((i-1) * N + j);
        }
    }
    CUdeviceptr devPtrA;
    printf("len = %d\n", M*N*sizeof(*a));
    CUresult ret = cuMemAlloc (&devPtrA, M*N*sizeof(*a));
    if (ret != CUDA_SUCCESS) {
      printf("mem alloc failed. retcode = %d\n", ret);
      return EXIT_FAILURE;
    }

    //if (cudaStat != cudaSuccess) {
    //    printf ("device memory allocation failed");
    //    return EXIT_FAILURE;
    //}
    printf("malloced\n");
    stat = cublasCreate_v2(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    printf("cublasCreate = %p\n", cublasCreate);
    printf("cublasSetMatrix = %p\n", cublasSetMatrix);

    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, (float *)(uintptr_t)devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("stat= %d\n", stat);
        printf("data download failed");
        cuMemFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    modify (handle, (float *)devPtrA, M, N, 2, 3, 16.0f, 12.0f);
    stat = cublasGetMatrix (M, N, sizeof(*a), (float *)devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cuMemFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    //cudaFree (devPtrA);
    cuMemFree(devPtrA);
    cublasDestroy(handle);
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= M; i++) {
            printf ("%7.0f", a[IDX2F(i,j,M)]);
        }
        printf ("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}
#endif

int main(int argc, char **argv)
{
  if (cuewInit(CUEW_INIT_CUDA) == CUEW_SUCCESS) {
    printf("CUDA found\n");
    printf("NVCC path: %s\n", cuewCompilerPath());
    printf("NVCC version: %d\n", cuewCompilerVersion());
  } else {
    printf("CUDA not found\n");
    exit(-1);
  }

  CUresult ret = cuInit(0);
  if (ret != CUDA_SUCCESS) {
    printf("cuda init faled\n");
    exit(-1);
  }

  CUdevice device;
  ret = cuDeviceGet(&device, 0);
  if (ret != CUDA_SUCCESS) {
    printf("dev get faled\n");
    exit(-1);
  }

  CUcontext ctx;
  ret = cuCtxCreate(&ctx, 0, device);
  if (ret != CUDA_SUCCESS) {
    printf("ctx faled\n");
    exit(-1);
  }


  //cublasHandle_t handle;
  //cublasStatus_t ret = cublasCreate_v2(&handle);
  //printf("ret = %d\n", ret);
  //printf("hahndle = %p\n", handle);

  //printf("sz = %ld\n", sizeof(CUdeviceptr));

  return cublas_test();

}
