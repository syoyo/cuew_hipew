#include <cstdio>
#include <cstdlib>

//#include <hip/hip_runtime.h>
#include <hip/device_functions.h>

int main()
{
  int s = hipInit(0);
  printf("hipInit = %d\n", s);

  if (s != 0) {
    return -1;
  }

  int ver;
  hipDriverGetVersion(&ver);
  printf("version = %d\n",ver);
  int count=0;
  hipGetDeviceCount(&count);

  printf("# of devices = %d\n", count);
  
  return 0;
}

