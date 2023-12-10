#include <cstdio>
#include <cstdlib>

#include "hipew.h"

int main()
{
  int ret = hipewInit(0);
  printf("ret = %d\n", ret);

  if (ret == 0) {
    int s = hipInit(0);
    printf("hipInit = %d\n", s);

    if (s != 0) {
      return -1;
    }
  }

  int ver;
  hipDriverGetVersion(&ver);
  printf("version = %d\n",ver);
  int count=0;
  hipGetDeviceCount(&count);

  printf("# of devices = %d\n", count);
  
  return 0;
}

