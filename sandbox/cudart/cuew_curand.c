

#ifdef _MSC_VER
#  if _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#  define popen _popen
#  define pclose _pclose
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include "cuew.h"
#include "cuew_curand.h"
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

/*#define CURAND_LIBRARY_FIND_CHECKED(name) CUEW_IMPL_LIBRARY_FIND_CHECKED(curand_lib, name)*/
#define CURAND_LIBRARY_FIND(name) CUEW_IMPL_LIBRARY_FIND(curand_lib, name)
static DynamicLibrary curand_lib;

static void cuewExitCURAND(void) {
  if (curand_lib != NULL) {
    /* ignore errors */
    dynamic_library_close(curand_lib);
    curand_lib = NULL;
  }
}

tcurandCreateGenerator *curandCreateGenerator;
tcurandCreateGeneratorHost *curandCreateGeneratorHost;
tcurandDestroyGenerator *curandDestroyGenerator;
tcurandGetVersion *curandGetVersion;
tcurandGetProperty *curandGetProperty;
tcurandSetStream *curandSetStream;
tcurandSetPseudoRandomGeneratorSeed *curandSetPseudoRandomGeneratorSeed;
tcurandSetGeneratorOffset *curandSetGeneratorOffset;
tcurandSetGeneratorOrdering *curandSetGeneratorOrdering;
tcurandSetQuasiRandomGeneratorDimensions *curandSetQuasiRandomGeneratorDimensions;
tcurandGenerate *curandGenerate;
tcurandGenerateLongLong *curandGenerateLongLong;
tcurandGenerateUniform *curandGenerateUniform;
tcurandGenerateUniformDouble *curandGenerateUniformDouble;
tcurandGenerateNormal *curandGenerateNormal;
tcurandGenerateNormalDouble *curandGenerateNormalDouble;
tcurandGenerateLogNormal *curandGenerateLogNormal;
tcurandGenerateLogNormalDouble *curandGenerateLogNormalDouble;
tcurandCreatePoissonDistribution *curandCreatePoissonDistribution;
tcurandDestroyDistribution *curandDestroyDistribution;
tcurandGeneratePoisson *curandGeneratePoisson;
tcurandGeneratePoissonMethod *curandGeneratePoissonMethod;
tcurandGenerateBinomial *curandGenerateBinomial;
tcurandGenerateBinomialMethod *curandGenerateBinomialMethod;
tcurandGenerateSeeds *curandGenerateSeeds;
tcurandGetDirectionVectors32 *curandGetDirectionVectors32;
tcurandGetScrambleConstants32 *curandGetScrambleConstants32;
tcurandGetDirectionVectors64 *curandGetDirectionVectors64;
tcurandGetScrambleConstants64 *curandGetScrambleConstants64;

int cuewInitCURAND(const char **extra_dll_search_paths) {

#ifdef _WIN32
  const char *paths[] = {   "curand.dll",
NULL};
#else /* linux */
  const char *paths[] = {   "libcurand.so",
   "/usr/local/cuda/lib64/libcurand.so",
NULL};
#endif


  static int initialized = 0;
  static int result = 0;
  int error;

  if (initialized) {
    return result;
  }

  initialized = 1;
  error = atexit(cuewExitCURAND);

  if (error) {
    result = -2;
    return result;
  }
  curand_lib = dynamic_library_open_find(paths);
  if (curand_lib == NULL) { 
    if (extra_dll_search_paths) { 
      curand_lib = dynamic_library_open_find(extra_dll_search_paths);
    }
  }
  if (curand_lib == NULL) { result = -1; return result; }

  CURAND_LIBRARY_FIND(curandCreateGenerator)
  CURAND_LIBRARY_FIND(curandCreateGeneratorHost)
  CURAND_LIBRARY_FIND(curandDestroyGenerator)
  CURAND_LIBRARY_FIND(curandGetVersion)
  CURAND_LIBRARY_FIND(curandGetProperty)
  CURAND_LIBRARY_FIND(curandSetStream)
  CURAND_LIBRARY_FIND(curandSetPseudoRandomGeneratorSeed)
  CURAND_LIBRARY_FIND(curandSetGeneratorOffset)
  CURAND_LIBRARY_FIND(curandSetGeneratorOrdering)
  CURAND_LIBRARY_FIND(curandSetQuasiRandomGeneratorDimensions)
  CURAND_LIBRARY_FIND(curandGenerate)
  CURAND_LIBRARY_FIND(curandGenerateLongLong)
  CURAND_LIBRARY_FIND(curandGenerateUniform)
  CURAND_LIBRARY_FIND(curandGenerateUniformDouble)
  CURAND_LIBRARY_FIND(curandGenerateNormal)
  CURAND_LIBRARY_FIND(curandGenerateNormalDouble)
  CURAND_LIBRARY_FIND(curandGenerateLogNormal)
  CURAND_LIBRARY_FIND(curandGenerateLogNormalDouble)
  CURAND_LIBRARY_FIND(curandCreatePoissonDistribution)
  CURAND_LIBRARY_FIND(curandDestroyDistribution)
  CURAND_LIBRARY_FIND(curandGeneratePoisson)
  CURAND_LIBRARY_FIND(curandGeneratePoissonMethod)
  CURAND_LIBRARY_FIND(curandGenerateBinomial)
  CURAND_LIBRARY_FIND(curandGenerateBinomialMethod)
  CURAND_LIBRARY_FIND(curandGenerateSeeds)
  CURAND_LIBRARY_FIND(curandGetDirectionVectors32)
  CURAND_LIBRARY_FIND(curandGetScrambleConstants32)
  CURAND_LIBRARY_FIND(curandGetDirectionVectors64)
  CURAND_LIBRARY_FIND(curandGetScrambleConstants64)
  result = 0; // success
  return result;
}
