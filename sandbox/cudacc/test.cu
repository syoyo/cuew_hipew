extern "C" int printf(const char*, ...);
extern "C" int cudaGetLastError(void);

#define __global__ __attribute__((global))

__global__ void test_func() {}

int main() {

  test_func<<<1,1>>>();
  printf("CUDA Error: %d", cudaGetLastError());
}

