

#ifdef _MSC_VER
#  if _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#  define popen _popen
#  define pclose _pclose
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include "cuew.h"
#include "cuew_cublas.h"
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

/*#define CUBLAS_LIBRARY_FIND_CHECKED(name) CUEW_IMPL_LIBRARY_FIND_CHECKED(cublas_lib, name)*/
#define CUBLAS_LIBRARY_FIND(name) CUEW_IMPL_LIBRARY_FIND(cublas_lib, name)
static DynamicLibrary cublas_lib;

static void cuewExitCUBLAS(void) {
  if (cublas_lib != NULL) {
    /* ignore errors */
    dynamic_library_close(cublas_lib);
    cublas_lib = NULL;
  }
}

tcublasCreate_v2 *cublasCreate_v2;
tcublasDestroy_v2 *cublasDestroy_v2;
tcublasGetVersion_v2 *cublasGetVersion_v2;
tcublasGetProperty *cublasGetProperty;
tcublasGetCudartVersion *cublasGetCudartVersion;
tcublasSetWorkspace_v2 *cublasSetWorkspace_v2;
tcublasSetStream_v2 *cublasSetStream_v2;
tcublasGetStream_v2 *cublasGetStream_v2;
tcublasGetPointerMode_v2 *cublasGetPointerMode_v2;
tcublasSetPointerMode_v2 *cublasSetPointerMode_v2;
tcublasGetAtomicsMode *cublasGetAtomicsMode;
tcublasSetAtomicsMode *cublasSetAtomicsMode;
tcublasGetMathMode *cublasGetMathMode;
tcublasSetMathMode *cublasSetMathMode;
tcublasGetSmCountTarget *cublasGetSmCountTarget;
tcublasSetSmCountTarget *cublasSetSmCountTarget;
tcublasGetStatusName *cublasGetStatusName;
tcublasGetStatusString *cublasGetStatusString;
tcublasLoggerConfigure *cublasLoggerConfigure;
tcublasSetLoggerCallback *cublasSetLoggerCallback;
tcublasGetLoggerCallback *cublasGetLoggerCallback;
tcublasSetVector *cublasSetVector;
tcublasGetVector *cublasGetVector;
tcublasSetMatrix *cublasSetMatrix;
tcublasGetMatrix *cublasGetMatrix;
tcublasSetVectorAsync *cublasSetVectorAsync;
tcublasGetVectorAsync *cublasGetVectorAsync;
tcublasSetMatrixAsync *cublasSetMatrixAsync;
tcublasGetMatrixAsync *cublasGetMatrixAsync;
tcublasXerbla *cublasXerbla;
tcublasNrm2Ex *cublasNrm2Ex;
tcublasSnrm2_v2 *cublasSnrm2_v2;
tcublasDnrm2_v2 *cublasDnrm2_v2;
tcublasScnrm2_v2 *cublasScnrm2_v2;
tcublasDznrm2_v2 *cublasDznrm2_v2;
tcublasDotEx *cublasDotEx;
tcublasDotcEx *cublasDotcEx;
tcublasSdot_v2 *cublasSdot_v2;
tcublasDdot_v2 *cublasDdot_v2;
tcublasCdotu_v2 *cublasCdotu_v2;
tcublasCdotc_v2 *cublasCdotc_v2;
tcublasZdotu_v2 *cublasZdotu_v2;
tcublasZdotc_v2 *cublasZdotc_v2;
tcublasScalEx *cublasScalEx;
tcublasSscal_v2 *cublasSscal_v2;
tcublasDscal_v2 *cublasDscal_v2;
tcublasCscal_v2 *cublasCscal_v2;
tcublasCsscal_v2 *cublasCsscal_v2;
tcublasZscal_v2 *cublasZscal_v2;
tcublasZdscal_v2 *cublasZdscal_v2;
tcublasAxpyEx *cublasAxpyEx;
tcublasSaxpy_v2 *cublasSaxpy_v2;
tcublasDaxpy_v2 *cublasDaxpy_v2;
tcublasCaxpy_v2 *cublasCaxpy_v2;
tcublasZaxpy_v2 *cublasZaxpy_v2;
tcublasCopyEx *cublasCopyEx;
tcublasScopy_v2 *cublasScopy_v2;
tcublasDcopy_v2 *cublasDcopy_v2;
tcublasCcopy_v2 *cublasCcopy_v2;
tcublasZcopy_v2 *cublasZcopy_v2;
tcublasSswap_v2 *cublasSswap_v2;
tcublasDswap_v2 *cublasDswap_v2;
tcublasCswap_v2 *cublasCswap_v2;
tcublasZswap_v2 *cublasZswap_v2;
tcublasSwapEx *cublasSwapEx;
tcublasIsamax_v2 *cublasIsamax_v2;
tcublasIdamax_v2 *cublasIdamax_v2;
tcublasIcamax_v2 *cublasIcamax_v2;
tcublasIzamax_v2 *cublasIzamax_v2;
tcublasIamaxEx *cublasIamaxEx;
tcublasIsamin_v2 *cublasIsamin_v2;
tcublasIdamin_v2 *cublasIdamin_v2;
tcublasIcamin_v2 *cublasIcamin_v2;
tcublasIzamin_v2 *cublasIzamin_v2;
tcublasIaminEx *cublasIaminEx;
tcublasAsumEx *cublasAsumEx;
tcublasSasum_v2 *cublasSasum_v2;
tcublasDasum_v2 *cublasDasum_v2;
tcublasScasum_v2 *cublasScasum_v2;
tcublasDzasum_v2 *cublasDzasum_v2;
tcublasSrot_v2 *cublasSrot_v2;
tcublasDrot_v2 *cublasDrot_v2;
tcublasCrot_v2 *cublasCrot_v2;
tcublasCsrot_v2 *cublasCsrot_v2;
tcublasZrot_v2 *cublasZrot_v2;
tcublasZdrot_v2 *cublasZdrot_v2;
tcublasRotEx *cublasRotEx;
tcublasSrotg_v2 *cublasSrotg_v2;
tcublasDrotg_v2 *cublasDrotg_v2;
tcublasCrotg_v2 *cublasCrotg_v2;
tcublasZrotg_v2 *cublasZrotg_v2;
tcublasRotgEx *cublasRotgEx;
tcublasSrotm_v2 *cublasSrotm_v2;
tcublasDrotm_v2 *cublasDrotm_v2;
tcublasRotmEx *cublasRotmEx;
tcublasSrotmg_v2 *cublasSrotmg_v2;
tcublasDrotmg_v2 *cublasDrotmg_v2;
tcublasRotmgEx *cublasRotmgEx;
tcublasSgemv_v2 *cublasSgemv_v2;
tcublasDgemv_v2 *cublasDgemv_v2;
tcublasCgemv_v2 *cublasCgemv_v2;
tcublasZgemv_v2 *cublasZgemv_v2;
tcublasSgbmv_v2 *cublasSgbmv_v2;
tcublasDgbmv_v2 *cublasDgbmv_v2;
tcublasCgbmv_v2 *cublasCgbmv_v2;
tcublasZgbmv_v2 *cublasZgbmv_v2;
tcublasStrmv_v2 *cublasStrmv_v2;
tcublasDtrmv_v2 *cublasDtrmv_v2;
tcublasCtrmv_v2 *cublasCtrmv_v2;
tcublasZtrmv_v2 *cublasZtrmv_v2;
tcublasStbmv_v2 *cublasStbmv_v2;
tcublasDtbmv_v2 *cublasDtbmv_v2;
tcublasCtbmv_v2 *cublasCtbmv_v2;
tcublasZtbmv_v2 *cublasZtbmv_v2;
tcublasStpmv_v2 *cublasStpmv_v2;
tcublasDtpmv_v2 *cublasDtpmv_v2;
tcublasCtpmv_v2 *cublasCtpmv_v2;
tcublasZtpmv_v2 *cublasZtpmv_v2;
tcublasStrsv_v2 *cublasStrsv_v2;
tcublasDtrsv_v2 *cublasDtrsv_v2;
tcublasCtrsv_v2 *cublasCtrsv_v2;
tcublasZtrsv_v2 *cublasZtrsv_v2;
tcublasStpsv_v2 *cublasStpsv_v2;
tcublasDtpsv_v2 *cublasDtpsv_v2;
tcublasCtpsv_v2 *cublasCtpsv_v2;
tcublasZtpsv_v2 *cublasZtpsv_v2;
tcublasStbsv_v2 *cublasStbsv_v2;
tcublasDtbsv_v2 *cublasDtbsv_v2;
tcublasCtbsv_v2 *cublasCtbsv_v2;
tcublasZtbsv_v2 *cublasZtbsv_v2;
tcublasSsymv_v2 *cublasSsymv_v2;
tcublasDsymv_v2 *cublasDsymv_v2;
tcublasCsymv_v2 *cublasCsymv_v2;
tcublasZsymv_v2 *cublasZsymv_v2;
tcublasChemv_v2 *cublasChemv_v2;
tcublasZhemv_v2 *cublasZhemv_v2;
tcublasSsbmv_v2 *cublasSsbmv_v2;
tcublasDsbmv_v2 *cublasDsbmv_v2;
tcublasChbmv_v2 *cublasChbmv_v2;
tcublasZhbmv_v2 *cublasZhbmv_v2;
tcublasSspmv_v2 *cublasSspmv_v2;
tcublasDspmv_v2 *cublasDspmv_v2;
tcublasChpmv_v2 *cublasChpmv_v2;
tcublasZhpmv_v2 *cublasZhpmv_v2;
tcublasSger_v2 *cublasSger_v2;
tcublasDger_v2 *cublasDger_v2;
tcublasCgeru_v2 *cublasCgeru_v2;
tcublasCgerc_v2 *cublasCgerc_v2;
tcublasZgeru_v2 *cublasZgeru_v2;
tcublasZgerc_v2 *cublasZgerc_v2;
tcublasSsyr_v2 *cublasSsyr_v2;
tcublasDsyr_v2 *cublasDsyr_v2;
tcublasCsyr_v2 *cublasCsyr_v2;
tcublasZsyr_v2 *cublasZsyr_v2;
tcublasCher_v2 *cublasCher_v2;
tcublasZher_v2 *cublasZher_v2;
tcublasSspr_v2 *cublasSspr_v2;
tcublasDspr_v2 *cublasDspr_v2;
tcublasChpr_v2 *cublasChpr_v2;
tcublasZhpr_v2 *cublasZhpr_v2;
tcublasSsyr2_v2 *cublasSsyr2_v2;
tcublasDsyr2_v2 *cublasDsyr2_v2;
tcublasCsyr2_v2 *cublasCsyr2_v2;
tcublasZsyr2_v2 *cublasZsyr2_v2;
tcublasCher2_v2 *cublasCher2_v2;
tcublasZher2_v2 *cublasZher2_v2;
tcublasSspr2_v2 *cublasSspr2_v2;
tcublasDspr2_v2 *cublasDspr2_v2;
tcublasChpr2_v2 *cublasChpr2_v2;
tcublasZhpr2_v2 *cublasZhpr2_v2;
tcublasSgemm_v2 *cublasSgemm_v2;
tcublasDgemm_v2 *cublasDgemm_v2;
tcublasCgemm_v2 *cublasCgemm_v2;
tcublasCgemm3m *cublasCgemm3m;
tcublasCgemm3mEx *cublasCgemm3mEx;
tcublasZgemm_v2 *cublasZgemm_v2;
tcublasZgemm3m *cublasZgemm3m;
tcublasSgemmEx *cublasSgemmEx;
tcublasGemmEx *cublasGemmEx;
tcublasCgemmEx *cublasCgemmEx;
tcublasUint8gemmBias *cublasUint8gemmBias;
tcublasSsyrk_v2 *cublasSsyrk_v2;
tcublasDsyrk_v2 *cublasDsyrk_v2;
tcublasCsyrk_v2 *cublasCsyrk_v2;
tcublasZsyrk_v2 *cublasZsyrk_v2;
tcublasCsyrkEx *cublasCsyrkEx;
tcublasCsyrk3mEx *cublasCsyrk3mEx;
tcublasCherk_v2 *cublasCherk_v2;
tcublasZherk_v2 *cublasZherk_v2;
tcublasCherkEx *cublasCherkEx;
tcublasCherk3mEx *cublasCherk3mEx;
tcublasSsyr2k_v2 *cublasSsyr2k_v2;
tcublasDsyr2k_v2 *cublasDsyr2k_v2;
tcublasCsyr2k_v2 *cublasCsyr2k_v2;
tcublasZsyr2k_v2 *cublasZsyr2k_v2;
tcublasCher2k_v2 *cublasCher2k_v2;
tcublasZher2k_v2 *cublasZher2k_v2;
tcublasSsyrkx *cublasSsyrkx;
tcublasDsyrkx *cublasDsyrkx;
tcublasCsyrkx *cublasCsyrkx;
tcublasZsyrkx *cublasZsyrkx;
tcublasCherkx *cublasCherkx;
tcublasZherkx *cublasZherkx;
tcublasSsymm_v2 *cublasSsymm_v2;
tcublasDsymm_v2 *cublasDsymm_v2;
tcublasCsymm_v2 *cublasCsymm_v2;
tcublasZsymm_v2 *cublasZsymm_v2;
tcublasChemm_v2 *cublasChemm_v2;
tcublasZhemm_v2 *cublasZhemm_v2;
tcublasStrsm_v2 *cublasStrsm_v2;
tcublasDtrsm_v2 *cublasDtrsm_v2;
tcublasCtrsm_v2 *cublasCtrsm_v2;
tcublasZtrsm_v2 *cublasZtrsm_v2;
tcublasStrmm_v2 *cublasStrmm_v2;
tcublasDtrmm_v2 *cublasDtrmm_v2;
tcublasCtrmm_v2 *cublasCtrmm_v2;
tcublasZtrmm_v2 *cublasZtrmm_v2;
tcublasSgemmBatched *cublasSgemmBatched;
tcublasDgemmBatched *cublasDgemmBatched;
tcublasCgemmBatched *cublasCgemmBatched;
tcublasCgemm3mBatched *cublasCgemm3mBatched;
tcublasZgemmBatched *cublasZgemmBatched;
tcublasGemmBatchedEx *cublasGemmBatchedEx;
tcublasGemmStridedBatchedEx *cublasGemmStridedBatchedEx;
tcublasSgemmStridedBatched *cublasSgemmStridedBatched;
tcublasDgemmStridedBatched *cublasDgemmStridedBatched;
tcublasCgemmStridedBatched *cublasCgemmStridedBatched;
tcublasCgemm3mStridedBatched *cublasCgemm3mStridedBatched;
tcublasZgemmStridedBatched *cublasZgemmStridedBatched;
tcublasSgeam *cublasSgeam;
tcublasDgeam *cublasDgeam;
tcublasCgeam *cublasCgeam;
tcublasZgeam *cublasZgeam;
tcublasSgetrfBatched *cublasSgetrfBatched;
tcublasDgetrfBatched *cublasDgetrfBatched;
tcublasCgetrfBatched *cublasCgetrfBatched;
tcublasZgetrfBatched *cublasZgetrfBatched;
tcublasSgetriBatched *cublasSgetriBatched;
tcublasDgetriBatched *cublasDgetriBatched;
tcublasCgetriBatched *cublasCgetriBatched;
tcublasZgetriBatched *cublasZgetriBatched;
tcublasSgetrsBatched *cublasSgetrsBatched;
tcublasDgetrsBatched *cublasDgetrsBatched;
tcublasCgetrsBatched *cublasCgetrsBatched;
tcublasZgetrsBatched *cublasZgetrsBatched;
tcublasStrsmBatched *cublasStrsmBatched;
tcublasDtrsmBatched *cublasDtrsmBatched;
tcublasCtrsmBatched *cublasCtrsmBatched;
tcublasZtrsmBatched *cublasZtrsmBatched;
tcublasSmatinvBatched *cublasSmatinvBatched;
tcublasDmatinvBatched *cublasDmatinvBatched;
tcublasCmatinvBatched *cublasCmatinvBatched;
tcublasZmatinvBatched *cublasZmatinvBatched;
tcublasSgeqrfBatched *cublasSgeqrfBatched;
tcublasDgeqrfBatched *cublasDgeqrfBatched;
tcublasCgeqrfBatched *cublasCgeqrfBatched;
tcublasZgeqrfBatched *cublasZgeqrfBatched;
tcublasSgelsBatched *cublasSgelsBatched;
tcublasDgelsBatched *cublasDgelsBatched;
tcublasCgelsBatched *cublasCgelsBatched;
tcublasZgelsBatched *cublasZgelsBatched;
tcublasSdgmm *cublasSdgmm;
tcublasDdgmm *cublasDdgmm;
tcublasCdgmm *cublasCdgmm;
tcublasZdgmm *cublasZdgmm;
tcublasStpttr *cublasStpttr;
tcublasDtpttr *cublasDtpttr;
tcublasCtpttr *cublasCtpttr;
tcublasZtpttr *cublasZtpttr;
tcublasStrttp *cublasStrttp;
tcublasDtrttp *cublasDtrttp;
tcublasCtrttp *cublasCtrttp;
tcublasZtrttp *cublasZtrttp;

int cuewInitCUBLAS() {

#ifdef _WIN32
  const char *paths[] = {   "cublas.dll",
NULL};
#else /* linux */
  const char *paths[] = {   "libcublas.so",
   "/usr/local/cuda/lib64/libcublas.so",
NULL};
#endif


  static int initialized = 0;
  static int result = 0;
  int error;

  if (initialized) {
    return result;
  }

  initialized = 1;
  error = atexit(cuewExitCUBLAS);

  if (error) {
    result = -2;
    return result;
  }
  cublas_lib = dynamic_library_open_find(paths);
  if (cublas_lib == NULL) { result = -1; return result; }

  CUBLAS_LIBRARY_FIND(cublasCreate_v2)
  CUBLAS_LIBRARY_FIND(cublasDestroy_v2)
  CUBLAS_LIBRARY_FIND(cublasGetVersion_v2)
  CUBLAS_LIBRARY_FIND(cublasGetProperty)
  CUBLAS_LIBRARY_FIND(cublasGetCudartVersion)
  CUBLAS_LIBRARY_FIND(cublasSetWorkspace_v2)
  CUBLAS_LIBRARY_FIND(cublasSetStream_v2)
  CUBLAS_LIBRARY_FIND(cublasGetStream_v2)
  CUBLAS_LIBRARY_FIND(cublasGetPointerMode_v2)
  CUBLAS_LIBRARY_FIND(cublasSetPointerMode_v2)
  CUBLAS_LIBRARY_FIND(cublasGetAtomicsMode)
  CUBLAS_LIBRARY_FIND(cublasSetAtomicsMode)
  CUBLAS_LIBRARY_FIND(cublasGetMathMode)
  CUBLAS_LIBRARY_FIND(cublasSetMathMode)
  CUBLAS_LIBRARY_FIND(cublasGetSmCountTarget)
  CUBLAS_LIBRARY_FIND(cublasSetSmCountTarget)
  CUBLAS_LIBRARY_FIND(cublasGetStatusName)
  CUBLAS_LIBRARY_FIND(cublasGetStatusString)
  CUBLAS_LIBRARY_FIND(cublasLoggerConfigure)
  CUBLAS_LIBRARY_FIND(cublasSetLoggerCallback)
  CUBLAS_LIBRARY_FIND(cublasGetLoggerCallback)
  CUBLAS_LIBRARY_FIND(cublasSetVector)
  CUBLAS_LIBRARY_FIND(cublasGetVector)
  CUBLAS_LIBRARY_FIND(cublasSetMatrix)
  CUBLAS_LIBRARY_FIND(cublasGetMatrix)
  CUBLAS_LIBRARY_FIND(cublasSetVectorAsync)
  CUBLAS_LIBRARY_FIND(cublasGetVectorAsync)
  CUBLAS_LIBRARY_FIND(cublasSetMatrixAsync)
  CUBLAS_LIBRARY_FIND(cublasGetMatrixAsync)
  CUBLAS_LIBRARY_FIND(cublasXerbla)
  CUBLAS_LIBRARY_FIND(cublasNrm2Ex)
  CUBLAS_LIBRARY_FIND(cublasSnrm2_v2)
  CUBLAS_LIBRARY_FIND(cublasDnrm2_v2)
  CUBLAS_LIBRARY_FIND(cublasScnrm2_v2)
  CUBLAS_LIBRARY_FIND(cublasDznrm2_v2)
  CUBLAS_LIBRARY_FIND(cublasDotEx)
  CUBLAS_LIBRARY_FIND(cublasDotcEx)
  CUBLAS_LIBRARY_FIND(cublasSdot_v2)
  CUBLAS_LIBRARY_FIND(cublasDdot_v2)
  CUBLAS_LIBRARY_FIND(cublasCdotu_v2)
  CUBLAS_LIBRARY_FIND(cublasCdotc_v2)
  CUBLAS_LIBRARY_FIND(cublasZdotu_v2)
  CUBLAS_LIBRARY_FIND(cublasZdotc_v2)
  CUBLAS_LIBRARY_FIND(cublasScalEx)
  CUBLAS_LIBRARY_FIND(cublasSscal_v2)
  CUBLAS_LIBRARY_FIND(cublasDscal_v2)
  CUBLAS_LIBRARY_FIND(cublasCscal_v2)
  CUBLAS_LIBRARY_FIND(cublasCsscal_v2)
  CUBLAS_LIBRARY_FIND(cublasZscal_v2)
  CUBLAS_LIBRARY_FIND(cublasZdscal_v2)
  CUBLAS_LIBRARY_FIND(cublasAxpyEx)
  CUBLAS_LIBRARY_FIND(cublasSaxpy_v2)
  CUBLAS_LIBRARY_FIND(cublasDaxpy_v2)
  CUBLAS_LIBRARY_FIND(cublasCaxpy_v2)
  CUBLAS_LIBRARY_FIND(cublasZaxpy_v2)
  CUBLAS_LIBRARY_FIND(cublasCopyEx)
  CUBLAS_LIBRARY_FIND(cublasScopy_v2)
  CUBLAS_LIBRARY_FIND(cublasDcopy_v2)
  CUBLAS_LIBRARY_FIND(cublasCcopy_v2)
  CUBLAS_LIBRARY_FIND(cublasZcopy_v2)
  CUBLAS_LIBRARY_FIND(cublasSswap_v2)
  CUBLAS_LIBRARY_FIND(cublasDswap_v2)
  CUBLAS_LIBRARY_FIND(cublasCswap_v2)
  CUBLAS_LIBRARY_FIND(cublasZswap_v2)
  CUBLAS_LIBRARY_FIND(cublasSwapEx)
  CUBLAS_LIBRARY_FIND(cublasIsamax_v2)
  CUBLAS_LIBRARY_FIND(cublasIdamax_v2)
  CUBLAS_LIBRARY_FIND(cublasIcamax_v2)
  CUBLAS_LIBRARY_FIND(cublasIzamax_v2)
  CUBLAS_LIBRARY_FIND(cublasIamaxEx)
  CUBLAS_LIBRARY_FIND(cublasIsamin_v2)
  CUBLAS_LIBRARY_FIND(cublasIdamin_v2)
  CUBLAS_LIBRARY_FIND(cublasIcamin_v2)
  CUBLAS_LIBRARY_FIND(cublasIzamin_v2)
  CUBLAS_LIBRARY_FIND(cublasIaminEx)
  CUBLAS_LIBRARY_FIND(cublasAsumEx)
  CUBLAS_LIBRARY_FIND(cublasSasum_v2)
  CUBLAS_LIBRARY_FIND(cublasDasum_v2)
  CUBLAS_LIBRARY_FIND(cublasScasum_v2)
  CUBLAS_LIBRARY_FIND(cublasDzasum_v2)
  CUBLAS_LIBRARY_FIND(cublasSrot_v2)
  CUBLAS_LIBRARY_FIND(cublasDrot_v2)
  CUBLAS_LIBRARY_FIND(cublasCrot_v2)
  CUBLAS_LIBRARY_FIND(cublasCsrot_v2)
  CUBLAS_LIBRARY_FIND(cublasZrot_v2)
  CUBLAS_LIBRARY_FIND(cublasZdrot_v2)
  CUBLAS_LIBRARY_FIND(cublasRotEx)
  CUBLAS_LIBRARY_FIND(cublasSrotg_v2)
  CUBLAS_LIBRARY_FIND(cublasDrotg_v2)
  CUBLAS_LIBRARY_FIND(cublasCrotg_v2)
  CUBLAS_LIBRARY_FIND(cublasZrotg_v2)
  CUBLAS_LIBRARY_FIND(cublasRotgEx)
  CUBLAS_LIBRARY_FIND(cublasSrotm_v2)
  CUBLAS_LIBRARY_FIND(cublasDrotm_v2)
  CUBLAS_LIBRARY_FIND(cublasRotmEx)
  CUBLAS_LIBRARY_FIND(cublasSrotmg_v2)
  CUBLAS_LIBRARY_FIND(cublasDrotmg_v2)
  CUBLAS_LIBRARY_FIND(cublasRotmgEx)
  CUBLAS_LIBRARY_FIND(cublasSgemv_v2)
  CUBLAS_LIBRARY_FIND(cublasDgemv_v2)
  CUBLAS_LIBRARY_FIND(cublasCgemv_v2)
  CUBLAS_LIBRARY_FIND(cublasZgemv_v2)
  CUBLAS_LIBRARY_FIND(cublasSgbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasDgbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasCgbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasZgbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasStrmv_v2)
  CUBLAS_LIBRARY_FIND(cublasDtrmv_v2)
  CUBLAS_LIBRARY_FIND(cublasCtrmv_v2)
  CUBLAS_LIBRARY_FIND(cublasZtrmv_v2)
  CUBLAS_LIBRARY_FIND(cublasStbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasDtbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasCtbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasZtbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasStpmv_v2)
  CUBLAS_LIBRARY_FIND(cublasDtpmv_v2)
  CUBLAS_LIBRARY_FIND(cublasCtpmv_v2)
  CUBLAS_LIBRARY_FIND(cublasZtpmv_v2)
  CUBLAS_LIBRARY_FIND(cublasStrsv_v2)
  CUBLAS_LIBRARY_FIND(cublasDtrsv_v2)
  CUBLAS_LIBRARY_FIND(cublasCtrsv_v2)
  CUBLAS_LIBRARY_FIND(cublasZtrsv_v2)
  CUBLAS_LIBRARY_FIND(cublasStpsv_v2)
  CUBLAS_LIBRARY_FIND(cublasDtpsv_v2)
  CUBLAS_LIBRARY_FIND(cublasCtpsv_v2)
  CUBLAS_LIBRARY_FIND(cublasZtpsv_v2)
  CUBLAS_LIBRARY_FIND(cublasStbsv_v2)
  CUBLAS_LIBRARY_FIND(cublasDtbsv_v2)
  CUBLAS_LIBRARY_FIND(cublasCtbsv_v2)
  CUBLAS_LIBRARY_FIND(cublasZtbsv_v2)
  CUBLAS_LIBRARY_FIND(cublasSsymv_v2)
  CUBLAS_LIBRARY_FIND(cublasDsymv_v2)
  CUBLAS_LIBRARY_FIND(cublasCsymv_v2)
  CUBLAS_LIBRARY_FIND(cublasZsymv_v2)
  CUBLAS_LIBRARY_FIND(cublasChemv_v2)
  CUBLAS_LIBRARY_FIND(cublasZhemv_v2)
  CUBLAS_LIBRARY_FIND(cublasSsbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasDsbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasChbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasZhbmv_v2)
  CUBLAS_LIBRARY_FIND(cublasSspmv_v2)
  CUBLAS_LIBRARY_FIND(cublasDspmv_v2)
  CUBLAS_LIBRARY_FIND(cublasChpmv_v2)
  CUBLAS_LIBRARY_FIND(cublasZhpmv_v2)
  CUBLAS_LIBRARY_FIND(cublasSger_v2)
  CUBLAS_LIBRARY_FIND(cublasDger_v2)
  CUBLAS_LIBRARY_FIND(cublasCgeru_v2)
  CUBLAS_LIBRARY_FIND(cublasCgerc_v2)
  CUBLAS_LIBRARY_FIND(cublasZgeru_v2)
  CUBLAS_LIBRARY_FIND(cublasZgerc_v2)
  CUBLAS_LIBRARY_FIND(cublasSsyr_v2)
  CUBLAS_LIBRARY_FIND(cublasDsyr_v2)
  CUBLAS_LIBRARY_FIND(cublasCsyr_v2)
  CUBLAS_LIBRARY_FIND(cublasZsyr_v2)
  CUBLAS_LIBRARY_FIND(cublasCher_v2)
  CUBLAS_LIBRARY_FIND(cublasZher_v2)
  CUBLAS_LIBRARY_FIND(cublasSspr_v2)
  CUBLAS_LIBRARY_FIND(cublasDspr_v2)
  CUBLAS_LIBRARY_FIND(cublasChpr_v2)
  CUBLAS_LIBRARY_FIND(cublasZhpr_v2)
  CUBLAS_LIBRARY_FIND(cublasSsyr2_v2)
  CUBLAS_LIBRARY_FIND(cublasDsyr2_v2)
  CUBLAS_LIBRARY_FIND(cublasCsyr2_v2)
  CUBLAS_LIBRARY_FIND(cublasZsyr2_v2)
  CUBLAS_LIBRARY_FIND(cublasCher2_v2)
  CUBLAS_LIBRARY_FIND(cublasZher2_v2)
  CUBLAS_LIBRARY_FIND(cublasSspr2_v2)
  CUBLAS_LIBRARY_FIND(cublasDspr2_v2)
  CUBLAS_LIBRARY_FIND(cublasChpr2_v2)
  CUBLAS_LIBRARY_FIND(cublasZhpr2_v2)
  CUBLAS_LIBRARY_FIND(cublasSgemm_v2)
  CUBLAS_LIBRARY_FIND(cublasDgemm_v2)
  CUBLAS_LIBRARY_FIND(cublasCgemm_v2)
  CUBLAS_LIBRARY_FIND(cublasCgemm3m)
  CUBLAS_LIBRARY_FIND(cublasCgemm3mEx)
  CUBLAS_LIBRARY_FIND(cublasZgemm_v2)
  CUBLAS_LIBRARY_FIND(cublasZgemm3m)
  CUBLAS_LIBRARY_FIND(cublasSgemmEx)
  CUBLAS_LIBRARY_FIND(cublasGemmEx)
  CUBLAS_LIBRARY_FIND(cublasCgemmEx)
  CUBLAS_LIBRARY_FIND(cublasUint8gemmBias)
  CUBLAS_LIBRARY_FIND(cublasSsyrk_v2)
  CUBLAS_LIBRARY_FIND(cublasDsyrk_v2)
  CUBLAS_LIBRARY_FIND(cublasCsyrk_v2)
  CUBLAS_LIBRARY_FIND(cublasZsyrk_v2)
  CUBLAS_LIBRARY_FIND(cublasCsyrkEx)
  CUBLAS_LIBRARY_FIND(cublasCsyrk3mEx)
  CUBLAS_LIBRARY_FIND(cublasCherk_v2)
  CUBLAS_LIBRARY_FIND(cublasZherk_v2)
  CUBLAS_LIBRARY_FIND(cublasCherkEx)
  CUBLAS_LIBRARY_FIND(cublasCherk3mEx)
  CUBLAS_LIBRARY_FIND(cublasSsyr2k_v2)
  CUBLAS_LIBRARY_FIND(cublasDsyr2k_v2)
  CUBLAS_LIBRARY_FIND(cublasCsyr2k_v2)
  CUBLAS_LIBRARY_FIND(cublasZsyr2k_v2)
  CUBLAS_LIBRARY_FIND(cublasCher2k_v2)
  CUBLAS_LIBRARY_FIND(cublasZher2k_v2)
  CUBLAS_LIBRARY_FIND(cublasSsyrkx)
  CUBLAS_LIBRARY_FIND(cublasDsyrkx)
  CUBLAS_LIBRARY_FIND(cublasCsyrkx)
  CUBLAS_LIBRARY_FIND(cublasZsyrkx)
  CUBLAS_LIBRARY_FIND(cublasCherkx)
  CUBLAS_LIBRARY_FIND(cublasZherkx)
  CUBLAS_LIBRARY_FIND(cublasSsymm_v2)
  CUBLAS_LIBRARY_FIND(cublasDsymm_v2)
  CUBLAS_LIBRARY_FIND(cublasCsymm_v2)
  CUBLAS_LIBRARY_FIND(cublasZsymm_v2)
  CUBLAS_LIBRARY_FIND(cublasChemm_v2)
  CUBLAS_LIBRARY_FIND(cublasZhemm_v2)
  CUBLAS_LIBRARY_FIND(cublasStrsm_v2)
  CUBLAS_LIBRARY_FIND(cublasDtrsm_v2)
  CUBLAS_LIBRARY_FIND(cublasCtrsm_v2)
  CUBLAS_LIBRARY_FIND(cublasZtrsm_v2)
  CUBLAS_LIBRARY_FIND(cublasStrmm_v2)
  CUBLAS_LIBRARY_FIND(cublasDtrmm_v2)
  CUBLAS_LIBRARY_FIND(cublasCtrmm_v2)
  CUBLAS_LIBRARY_FIND(cublasZtrmm_v2)
  CUBLAS_LIBRARY_FIND(cublasSgemmBatched)
  CUBLAS_LIBRARY_FIND(cublasDgemmBatched)
  CUBLAS_LIBRARY_FIND(cublasCgemmBatched)
  CUBLAS_LIBRARY_FIND(cublasCgemm3mBatched)
  CUBLAS_LIBRARY_FIND(cublasZgemmBatched)
  CUBLAS_LIBRARY_FIND(cublasGemmBatchedEx)
  CUBLAS_LIBRARY_FIND(cublasGemmStridedBatchedEx)
  CUBLAS_LIBRARY_FIND(cublasSgemmStridedBatched)
  CUBLAS_LIBRARY_FIND(cublasDgemmStridedBatched)
  CUBLAS_LIBRARY_FIND(cublasCgemmStridedBatched)
  CUBLAS_LIBRARY_FIND(cublasCgemm3mStridedBatched)
  CUBLAS_LIBRARY_FIND(cublasZgemmStridedBatched)
  CUBLAS_LIBRARY_FIND(cublasSgeam)
  CUBLAS_LIBRARY_FIND(cublasDgeam)
  CUBLAS_LIBRARY_FIND(cublasCgeam)
  CUBLAS_LIBRARY_FIND(cublasZgeam)
  CUBLAS_LIBRARY_FIND(cublasSgetrfBatched)
  CUBLAS_LIBRARY_FIND(cublasDgetrfBatched)
  CUBLAS_LIBRARY_FIND(cublasCgetrfBatched)
  CUBLAS_LIBRARY_FIND(cublasZgetrfBatched)
  CUBLAS_LIBRARY_FIND(cublasSgetriBatched)
  CUBLAS_LIBRARY_FIND(cublasDgetriBatched)
  CUBLAS_LIBRARY_FIND(cublasCgetriBatched)
  CUBLAS_LIBRARY_FIND(cublasZgetriBatched)
  CUBLAS_LIBRARY_FIND(cublasSgetrsBatched)
  CUBLAS_LIBRARY_FIND(cublasDgetrsBatched)
  CUBLAS_LIBRARY_FIND(cublasCgetrsBatched)
  CUBLAS_LIBRARY_FIND(cublasZgetrsBatched)
  CUBLAS_LIBRARY_FIND(cublasStrsmBatched)
  CUBLAS_LIBRARY_FIND(cublasDtrsmBatched)
  CUBLAS_LIBRARY_FIND(cublasCtrsmBatched)
  CUBLAS_LIBRARY_FIND(cublasZtrsmBatched)
  CUBLAS_LIBRARY_FIND(cublasSmatinvBatched)
  CUBLAS_LIBRARY_FIND(cublasDmatinvBatched)
  CUBLAS_LIBRARY_FIND(cublasCmatinvBatched)
  CUBLAS_LIBRARY_FIND(cublasZmatinvBatched)
  CUBLAS_LIBRARY_FIND(cublasSgeqrfBatched)
  CUBLAS_LIBRARY_FIND(cublasDgeqrfBatched)
  CUBLAS_LIBRARY_FIND(cublasCgeqrfBatched)
  CUBLAS_LIBRARY_FIND(cublasZgeqrfBatched)
  CUBLAS_LIBRARY_FIND(cublasSgelsBatched)
  CUBLAS_LIBRARY_FIND(cublasDgelsBatched)
  CUBLAS_LIBRARY_FIND(cublasCgelsBatched)
  CUBLAS_LIBRARY_FIND(cublasZgelsBatched)
  CUBLAS_LIBRARY_FIND(cublasSdgmm)
  CUBLAS_LIBRARY_FIND(cublasDdgmm)
  CUBLAS_LIBRARY_FIND(cublasCdgmm)
  CUBLAS_LIBRARY_FIND(cublasZdgmm)
  CUBLAS_LIBRARY_FIND(cublasStpttr)
  CUBLAS_LIBRARY_FIND(cublasDtpttr)
  CUBLAS_LIBRARY_FIND(cublasCtpttr)
  CUBLAS_LIBRARY_FIND(cublasZtpttr)
  CUBLAS_LIBRARY_FIND(cublasStrttp)
  CUBLAS_LIBRARY_FIND(cublasDtrttp)
  CUBLAS_LIBRARY_FIND(cublasCtrttp)
  CUBLAS_LIBRARY_FIND(cublasZtrttp)
  result = 0; // success
  return result;
}
