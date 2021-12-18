

#ifdef _MSC_VER
#  if _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#  define popen _popen
#  define pclose _pclose
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include "cudart.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>


#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRALEAN
#  include <windows.h>


/* Utility macros. */

typedef HMODULE DynamicLibrary;

#  define dynamic_library_open(path)         LoadLibraryA(path)
#  define dynamic_library_close(lib)         FreeLibrary(lib)
#  define dynamic_library_find(lib, symbol)  GetProcAddress(lib, symbol)
#else
#  include <dlfcn.h>

typedef void* DynamicLibrary;

#  define dynamic_library_open(path)         dlopen(path, RTLD_NOW)
#  define dynamic_library_close(lib)         dlclose(lib)
#  define dynamic_library_find(lib, symbol)  dlsym(lib, symbol)
#endif

/*
#define CUEW_IMPL_LIBRARY_FIND_CHECKED(lib, name)         name = (t##name *)dynamic_library_find(lib, #name);         assert(name);
*/

#define CUEW_IMPL_LIBRARY_FIND(lib, name)         name = (t##name *)dynamic_library_find(lib, #name);


static DynamicLibrary dynamic_library_open_find(const char **paths) {
  int i = 0;
  while (paths[i] != NULL) {
      DynamicLibrary lib = dynamic_library_open(paths[i]);
      if (lib != NULL) {
        return lib;
      }
      ++i;
  }
  return NULL;
}

/*#define CUDART_LIBRARY_FIND_CHECKED(name) CUEW_IMPL_LIBRARY_FIND_CHECKED(cudart_lib, name)*/
#define CUDART_LIBRARY_FIND(name) CUEW_IMPL_LIBRARY_FIND(cudart_lib, name)
static DynamicLibrary cudart_lib;

static void cuewExitCUDART(void) {
  if (cudart_lib != NULL) {
    /* ignore errors */
    dynamic_library_close(cudart_lib);
    cudart_lib = NULL;
  }
}

tcufftPlan1d *cufftPlan1d;
tcufftPlan2d *cufftPlan2d;
tcufftPlan3d *cufftPlan3d;
tcufftPlanMany *cufftPlanMany;
tcufftMakePlan1d *cufftMakePlan1d;
tcufftMakePlan2d *cufftMakePlan2d;
tcufftMakePlan3d *cufftMakePlan3d;
tcufftMakePlanMany *cufftMakePlanMany;
tcufftMakePlanMany64 *cufftMakePlanMany64;
tcufftGetSizeMany64 *cufftGetSizeMany64;
tcufftEstimate1d *cufftEstimate1d;
tcufftEstimate2d *cufftEstimate2d;
tcufftEstimate3d *cufftEstimate3d;
tcufftEstimateMany *cufftEstimateMany;
tcufftCreate *cufftCreate;
tcufftGetSize1d *cufftGetSize1d;
tcufftGetSize2d *cufftGetSize2d;
tcufftGetSize3d *cufftGetSize3d;
tcufftGetSizeMany *cufftGetSizeMany;
tcufftGetSize *cufftGetSize;
tcufftSetWorkArea *cufftSetWorkArea;
tcufftSetAutoAllocation *cufftSetAutoAllocation;
tcufftExecC2C *cufftExecC2C;
tcufftExecR2C *cufftExecR2C;
tcufftExecC2R *cufftExecC2R;
tcufftExecZ2Z *cufftExecZ2Z;
tcufftExecD2Z *cufftExecD2Z;
tcufftExecZ2D *cufftExecZ2D;
tcufftSetStream *cufftSetStream;
tcufftDestroy *cufftDestroy;
tcufftGetVersion *cufftGetVersion;
tcufftGetProperty *cufftGetProperty;

int cuewInitCUDART() {

#ifdef _WIN32
  const char *paths[] = {   "cufft.dll",
NULL};
#else /* linux */
  const char *paths[] = {   "libcufft.so",
   "/usr/local/cuda/lib64/libcufft.so",
NULL};
#endif


  static int initialized = 0;
  static int result = 0;
  int error;

  if (initialized) {
    return result;
  }

  initialized = 1;
  error = atexit(cuewExitCUDART);

  if (error) {
    result = -2;
    return result;
  }
  cudart_lib = dynamic_library_open_find(paths);
  if (cudart_lib == NULL) { result = -1; return result; }

  CUDART_LIBRARY_FIND(cufftPlan1d)
  CUDART_LIBRARY_FIND(cufftPlan2d)
  CUDART_LIBRARY_FIND(cufftPlan3d)
  CUDART_LIBRARY_FIND(cufftPlanMany)
  CUDART_LIBRARY_FIND(cufftMakePlan1d)
  CUDART_LIBRARY_FIND(cufftMakePlan2d)
  CUDART_LIBRARY_FIND(cufftMakePlan3d)
  CUDART_LIBRARY_FIND(cufftMakePlanMany)
  CUDART_LIBRARY_FIND(cufftMakePlanMany64)
  CUDART_LIBRARY_FIND(cufftGetSizeMany64)
  CUDART_LIBRARY_FIND(cufftEstimate1d)
  CUDART_LIBRARY_FIND(cufftEstimate2d)
  CUDART_LIBRARY_FIND(cufftEstimate3d)
  CUDART_LIBRARY_FIND(cufftEstimateMany)
  CUDART_LIBRARY_FIND(cufftCreate)
  CUDART_LIBRARY_FIND(cufftGetSize1d)
  CUDART_LIBRARY_FIND(cufftGetSize2d)
  CUDART_LIBRARY_FIND(cufftGetSize3d)
  CUDART_LIBRARY_FIND(cufftGetSizeMany)
  CUDART_LIBRARY_FIND(cufftGetSize)
  CUDART_LIBRARY_FIND(cufftSetWorkArea)
  CUDART_LIBRARY_FIND(cufftSetAutoAllocation)
  CUDART_LIBRARY_FIND(cufftExecC2C)
  CUDART_LIBRARY_FIND(cufftExecR2C)
  CUDART_LIBRARY_FIND(cufftExecC2R)
  CUDART_LIBRARY_FIND(cufftExecZ2Z)
  CUDART_LIBRARY_FIND(cufftExecD2Z)
  CUDART_LIBRARY_FIND(cufftExecZ2D)
  CUDART_LIBRARY_FIND(cufftSetStream)
  CUDART_LIBRARY_FIND(cufftDestroy)
  CUDART_LIBRARY_FIND(cufftGetVersion)
  CUDART_LIBRARY_FIND(cufftGetProperty)
  result = 0; // success
  return result;
}
