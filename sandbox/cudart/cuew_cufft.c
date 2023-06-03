

#ifdef _MSC_VER
#  if _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#  define popen _popen
#  define pclose _pclose
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include "cuew.h"
#include "cuew_cufft.h"
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

/*#define CUFFT_LIBRARY_FIND_CHECKED(name) CUEW_IMPL_LIBRARY_FIND_CHECKED(cufft_lib, name)*/
#define CUFFT_LIBRARY_FIND(name) CUEW_IMPL_LIBRARY_FIND(cufft_lib, name)
static DynamicLibrary cufft_lib;

static void cuewExitCUFFT(void) {
  if (cufft_lib != NULL) {
    /* ignore errors */
    dynamic_library_close(cufft_lib);
    cufft_lib = NULL;
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

int cuewInitCUFFT(const char **extra_dll_search_paths) {

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
  error = atexit(cuewExitCUFFT);

  if (error) {
    result = -2;
    return result;
  }
  cufft_lib = dynamic_library_open_find(paths);
  if (cufft_lib == NULL) { 
    if (extra_dll_search_paths) { 
      cufft_lib = dynamic_library_open_find(extra_dll_search_paths);
    }
  }
  if (cufft_lib == NULL) { result = -1; return result; }

  CUFFT_LIBRARY_FIND(cufftPlan1d)
  CUFFT_LIBRARY_FIND(cufftPlan2d)
  CUFFT_LIBRARY_FIND(cufftPlan3d)
  CUFFT_LIBRARY_FIND(cufftPlanMany)
  CUFFT_LIBRARY_FIND(cufftMakePlan1d)
  CUFFT_LIBRARY_FIND(cufftMakePlan2d)
  CUFFT_LIBRARY_FIND(cufftMakePlan3d)
  CUFFT_LIBRARY_FIND(cufftMakePlanMany)
  CUFFT_LIBRARY_FIND(cufftMakePlanMany64)
  CUFFT_LIBRARY_FIND(cufftGetSizeMany64)
  CUFFT_LIBRARY_FIND(cufftEstimate1d)
  CUFFT_LIBRARY_FIND(cufftEstimate2d)
  CUFFT_LIBRARY_FIND(cufftEstimate3d)
  CUFFT_LIBRARY_FIND(cufftEstimateMany)
  CUFFT_LIBRARY_FIND(cufftCreate)
  CUFFT_LIBRARY_FIND(cufftGetSize1d)
  CUFFT_LIBRARY_FIND(cufftGetSize2d)
  CUFFT_LIBRARY_FIND(cufftGetSize3d)
  CUFFT_LIBRARY_FIND(cufftGetSizeMany)
  CUFFT_LIBRARY_FIND(cufftGetSize)
  CUFFT_LIBRARY_FIND(cufftSetWorkArea)
  CUFFT_LIBRARY_FIND(cufftSetAutoAllocation)
  CUFFT_LIBRARY_FIND(cufftExecC2C)
  CUFFT_LIBRARY_FIND(cufftExecR2C)
  CUFFT_LIBRARY_FIND(cufftExecC2R)
  CUFFT_LIBRARY_FIND(cufftExecZ2Z)
  CUFFT_LIBRARY_FIND(cufftExecD2Z)
  CUFFT_LIBRARY_FIND(cufftExecZ2D)
  CUFFT_LIBRARY_FIND(cufftSetStream)
  CUFFT_LIBRARY_FIND(cufftDestroy)
  CUFFT_LIBRARY_FIND(cufftGetVersion)
  CUFFT_LIBRARY_FIND(cufftGetProperty)
  result = 0; // success
  return result;
}
