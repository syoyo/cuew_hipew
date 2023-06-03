

#ifdef _MSC_VER
#  if _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#  define popen _popen
#  define pclose _pclose
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include "cuew.h"
#include "cuew_cusolver.h"
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

/*#define CUSOLVER_LIBRARY_FIND_CHECKED(name) CUEW_IMPL_LIBRARY_FIND_CHECKED(cusolver_lib, name)*/
#define CUSOLVER_LIBRARY_FIND(name) CUEW_IMPL_LIBRARY_FIND(cusolver_lib, name)
static DynamicLibrary cusolver_lib;

static void cuewExitCUSOLVER(void) {
  if (cusolver_lib != NULL) {
    /* ignore errors */
    dynamic_library_close(cusolver_lib);
    cusolver_lib = NULL;
  }
}

tcusolverGetProperty *cusolverGetProperty;
tcusolverGetVersion *cusolverGetVersion;
tcusolverRfCreate *cusolverRfCreate;
tcusolverRfDestroy *cusolverRfDestroy;
tcusolverRfGetMatrixFormat *cusolverRfGetMatrixFormat;
tcusolverRfSetMatrixFormat *cusolverRfSetMatrixFormat;
tcusolverRfSetNumericProperties *cusolverRfSetNumericProperties;
tcusolverRfGetNumericProperties *cusolverRfGetNumericProperties;
tcusolverRfGetNumericBoostReport *cusolverRfGetNumericBoostReport;
tcusolverRfSetAlgs *cusolverRfSetAlgs;
tcusolverRfGetAlgs *cusolverRfGetAlgs;
tcusolverRfGetResetValuesFastMode *cusolverRfGetResetValuesFastMode;
tcusolverRfSetResetValuesFastMode *cusolverRfSetResetValuesFastMode;
tcusolverRfSetupHost *cusolverRfSetupHost;
tcusolverRfSetupDevice *cusolverRfSetupDevice;
tcusolverRfResetValues *cusolverRfResetValues;
tcusolverRfAnalyze *cusolverRfAnalyze;
tcusolverRfRefactor *cusolverRfRefactor;
tcusolverRfAccessBundledFactorsDevice *cusolverRfAccessBundledFactorsDevice;
tcusolverRfExtractBundledFactorsHost *cusolverRfExtractBundledFactorsHost;
tcusolverRfExtractSplitFactorsHost *cusolverRfExtractSplitFactorsHost;
tcusolverRfSolve *cusolverRfSolve;
tcusolverRfBatchSetupHost *cusolverRfBatchSetupHost;
tcusolverRfBatchResetValues *cusolverRfBatchResetValues;
tcusolverRfBatchAnalyze *cusolverRfBatchAnalyze;
tcusolverRfBatchRefactor *cusolverRfBatchRefactor;
tcusolverRfBatchSolve *cusolverRfBatchSolve;
tcusolverRfBatchZeroPivot *cusolverRfBatchZeroPivot;
tcusolverSpCreate *cusolverSpCreate;
tcusolverSpDestroy *cusolverSpDestroy;
tcusolverSpSetStream *cusolverSpSetStream;
tcusolverSpGetStream *cusolverSpGetStream;
tcusolverSpXcsrissymHost *cusolverSpXcsrissymHost;
tcusolverSpScsrlsvluHost *cusolverSpScsrlsvluHost;
tcusolverSpDcsrlsvluHost *cusolverSpDcsrlsvluHost;
tcusolverSpCcsrlsvluHost *cusolverSpCcsrlsvluHost;
tcusolverSpZcsrlsvluHost *cusolverSpZcsrlsvluHost;
tcusolverSpScsrlsvqr *cusolverSpScsrlsvqr;
tcusolverSpDcsrlsvqr *cusolverSpDcsrlsvqr;
tcusolverSpCcsrlsvqr *cusolverSpCcsrlsvqr;
tcusolverSpZcsrlsvqr *cusolverSpZcsrlsvqr;
tcusolverSpScsrlsvqrHost *cusolverSpScsrlsvqrHost;
tcusolverSpDcsrlsvqrHost *cusolverSpDcsrlsvqrHost;
tcusolverSpCcsrlsvqrHost *cusolverSpCcsrlsvqrHost;
tcusolverSpZcsrlsvqrHost *cusolverSpZcsrlsvqrHost;
tcusolverSpScsrlsvcholHost *cusolverSpScsrlsvcholHost;
tcusolverSpDcsrlsvcholHost *cusolverSpDcsrlsvcholHost;
tcusolverSpCcsrlsvcholHost *cusolverSpCcsrlsvcholHost;
tcusolverSpZcsrlsvcholHost *cusolverSpZcsrlsvcholHost;
tcusolverSpScsrlsvchol *cusolverSpScsrlsvchol;
tcusolverSpDcsrlsvchol *cusolverSpDcsrlsvchol;
tcusolverSpCcsrlsvchol *cusolverSpCcsrlsvchol;
tcusolverSpZcsrlsvchol *cusolverSpZcsrlsvchol;
tcusolverSpScsrlsqvqrHost *cusolverSpScsrlsqvqrHost;
tcusolverSpDcsrlsqvqrHost *cusolverSpDcsrlsqvqrHost;
tcusolverSpCcsrlsqvqrHost *cusolverSpCcsrlsqvqrHost;
tcusolverSpZcsrlsqvqrHost *cusolverSpZcsrlsqvqrHost;
tcusolverSpScsreigvsiHost *cusolverSpScsreigvsiHost;
tcusolverSpDcsreigvsiHost *cusolverSpDcsreigvsiHost;
tcusolverSpCcsreigvsiHost *cusolverSpCcsreigvsiHost;
tcusolverSpZcsreigvsiHost *cusolverSpZcsreigvsiHost;
tcusolverSpScsreigvsi *cusolverSpScsreigvsi;
tcusolverSpDcsreigvsi *cusolverSpDcsreigvsi;
tcusolverSpCcsreigvsi *cusolverSpCcsreigvsi;
tcusolverSpZcsreigvsi *cusolverSpZcsreigvsi;
tcusolverSpScsreigsHost *cusolverSpScsreigsHost;
tcusolverSpDcsreigsHost *cusolverSpDcsreigsHost;
tcusolverSpCcsreigsHost *cusolverSpCcsreigsHost;
tcusolverSpZcsreigsHost *cusolverSpZcsreigsHost;
tcusolverSpXcsrsymrcmHost *cusolverSpXcsrsymrcmHost;
tcusolverSpXcsrsymmdqHost *cusolverSpXcsrsymmdqHost;
tcusolverSpXcsrsymamdHost *cusolverSpXcsrsymamdHost;
tcusolverSpXcsrmetisndHost *cusolverSpXcsrmetisndHost;
tcusolverSpScsrzfdHost *cusolverSpScsrzfdHost;
tcusolverSpDcsrzfdHost *cusolverSpDcsrzfdHost;
tcusolverSpCcsrzfdHost *cusolverSpCcsrzfdHost;
tcusolverSpZcsrzfdHost *cusolverSpZcsrzfdHost;
tcusolverSpXcsrperm_bufferSizeHost *cusolverSpXcsrperm_bufferSizeHost;
tcusolverSpXcsrpermHost *cusolverSpXcsrpermHost;
tcusolverSpCreateCsrqrInfo *cusolverSpCreateCsrqrInfo;
tcusolverSpDestroyCsrqrInfo *cusolverSpDestroyCsrqrInfo;
tcusolverSpXcsrqrAnalysisBatched *cusolverSpXcsrqrAnalysisBatched;
tcusolverSpScsrqrBufferInfoBatched *cusolverSpScsrqrBufferInfoBatched;
tcusolverSpDcsrqrBufferInfoBatched *cusolverSpDcsrqrBufferInfoBatched;
tcusolverSpCcsrqrBufferInfoBatched *cusolverSpCcsrqrBufferInfoBatched;
tcusolverSpZcsrqrBufferInfoBatched *cusolverSpZcsrqrBufferInfoBatched;
tcusolverSpScsrqrsvBatched *cusolverSpScsrqrsvBatched;
tcusolverSpDcsrqrsvBatched *cusolverSpDcsrqrsvBatched;
tcusolverSpCcsrqrsvBatched *cusolverSpCcsrqrsvBatched;
tcusolverSpZcsrqrsvBatched *cusolverSpZcsrqrsvBatched;
tcusolverDnCreate *cusolverDnCreate;
tcusolverDnDestroy *cusolverDnDestroy;
tcusolverDnSetStream *cusolverDnSetStream;
tcusolverDnGetStream *cusolverDnGetStream;
tcusolverDnIRSParamsCreate *cusolverDnIRSParamsCreate;
tcusolverDnIRSParamsDestroy *cusolverDnIRSParamsDestroy;
tcusolverDnIRSParamsSetRefinementSolver *cusolverDnIRSParamsSetRefinementSolver;
tcusolverDnIRSParamsSetSolverMainPrecision *cusolverDnIRSParamsSetSolverMainPrecision;
tcusolverDnIRSParamsSetSolverLowestPrecision *cusolverDnIRSParamsSetSolverLowestPrecision;
tcusolverDnIRSParamsSetSolverPrecisions *cusolverDnIRSParamsSetSolverPrecisions;
tcusolverDnIRSParamsSetTol *cusolverDnIRSParamsSetTol;
tcusolverDnIRSParamsSetTolInner *cusolverDnIRSParamsSetTolInner;
tcusolverDnIRSParamsSetMaxIters *cusolverDnIRSParamsSetMaxIters;
tcusolverDnIRSParamsSetMaxItersInner *cusolverDnIRSParamsSetMaxItersInner;
tcusolverDnIRSParamsGetMaxIters *cusolverDnIRSParamsGetMaxIters;
tcusolverDnIRSParamsEnableFallback *cusolverDnIRSParamsEnableFallback;
tcusolverDnIRSParamsDisableFallback *cusolverDnIRSParamsDisableFallback;
tcusolverDnIRSInfosDestroy *cusolverDnIRSInfosDestroy;
tcusolverDnIRSInfosCreate *cusolverDnIRSInfosCreate;
tcusolverDnIRSInfosGetNiters *cusolverDnIRSInfosGetNiters;
tcusolverDnIRSInfosGetOuterNiters *cusolverDnIRSInfosGetOuterNiters;
tcusolverDnIRSInfosRequestResidual *cusolverDnIRSInfosRequestResidual;
tcusolverDnIRSInfosGetResidualHistory *cusolverDnIRSInfosGetResidualHistory;
tcusolverDnIRSInfosGetMaxIters *cusolverDnIRSInfosGetMaxIters;
tcusolverDnZZgesv *cusolverDnZZgesv;
tcusolverDnZCgesv *cusolverDnZCgesv;
tcusolverDnZKgesv *cusolverDnZKgesv;
tcusolverDnZEgesv *cusolverDnZEgesv;
tcusolverDnZYgesv *cusolverDnZYgesv;
tcusolverDnCCgesv *cusolverDnCCgesv;
tcusolverDnCEgesv *cusolverDnCEgesv;
tcusolverDnCKgesv *cusolverDnCKgesv;
tcusolverDnCYgesv *cusolverDnCYgesv;
tcusolverDnDDgesv *cusolverDnDDgesv;
tcusolverDnDSgesv *cusolverDnDSgesv;
tcusolverDnDHgesv *cusolverDnDHgesv;
tcusolverDnDBgesv *cusolverDnDBgesv;
tcusolverDnDXgesv *cusolverDnDXgesv;
tcusolverDnSSgesv *cusolverDnSSgesv;
tcusolverDnSHgesv *cusolverDnSHgesv;
tcusolverDnSBgesv *cusolverDnSBgesv;
tcusolverDnSXgesv *cusolverDnSXgesv;
tcusolverDnZZgesv_bufferSize *cusolverDnZZgesv_bufferSize;
tcusolverDnZCgesv_bufferSize *cusolverDnZCgesv_bufferSize;
tcusolverDnZKgesv_bufferSize *cusolverDnZKgesv_bufferSize;
tcusolverDnZEgesv_bufferSize *cusolverDnZEgesv_bufferSize;
tcusolverDnZYgesv_bufferSize *cusolverDnZYgesv_bufferSize;
tcusolverDnCCgesv_bufferSize *cusolverDnCCgesv_bufferSize;
tcusolverDnCKgesv_bufferSize *cusolverDnCKgesv_bufferSize;
tcusolverDnCEgesv_bufferSize *cusolverDnCEgesv_bufferSize;
tcusolverDnCYgesv_bufferSize *cusolverDnCYgesv_bufferSize;
tcusolverDnDDgesv_bufferSize *cusolverDnDDgesv_bufferSize;
tcusolverDnDSgesv_bufferSize *cusolverDnDSgesv_bufferSize;
tcusolverDnDHgesv_bufferSize *cusolverDnDHgesv_bufferSize;
tcusolverDnDBgesv_bufferSize *cusolverDnDBgesv_bufferSize;
tcusolverDnDXgesv_bufferSize *cusolverDnDXgesv_bufferSize;
tcusolverDnSSgesv_bufferSize *cusolverDnSSgesv_bufferSize;
tcusolverDnSHgesv_bufferSize *cusolverDnSHgesv_bufferSize;
tcusolverDnSBgesv_bufferSize *cusolverDnSBgesv_bufferSize;
tcusolverDnSXgesv_bufferSize *cusolverDnSXgesv_bufferSize;
tcusolverDnZZgels *cusolverDnZZgels;
tcusolverDnZCgels *cusolverDnZCgels;
tcusolverDnZKgels *cusolverDnZKgels;
tcusolverDnZEgels *cusolverDnZEgels;
tcusolverDnZYgels *cusolverDnZYgels;
tcusolverDnCCgels *cusolverDnCCgels;
tcusolverDnCKgels *cusolverDnCKgels;
tcusolverDnCEgels *cusolverDnCEgels;
tcusolverDnCYgels *cusolverDnCYgels;
tcusolverDnDDgels *cusolverDnDDgels;
tcusolverDnDSgels *cusolverDnDSgels;
tcusolverDnDHgels *cusolverDnDHgels;
tcusolverDnDBgels *cusolverDnDBgels;
tcusolverDnDXgels *cusolverDnDXgels;
tcusolverDnSSgels *cusolverDnSSgels;
tcusolverDnSHgels *cusolverDnSHgels;
tcusolverDnSBgels *cusolverDnSBgels;
tcusolverDnSXgels *cusolverDnSXgels;
tcusolverDnZZgels_bufferSize *cusolverDnZZgels_bufferSize;
tcusolverDnZCgels_bufferSize *cusolverDnZCgels_bufferSize;
tcusolverDnZKgels_bufferSize *cusolverDnZKgels_bufferSize;
tcusolverDnZEgels_bufferSize *cusolverDnZEgels_bufferSize;
tcusolverDnZYgels_bufferSize *cusolverDnZYgels_bufferSize;
tcusolverDnCCgels_bufferSize *cusolverDnCCgels_bufferSize;
tcusolverDnCKgels_bufferSize *cusolverDnCKgels_bufferSize;
tcusolverDnCEgels_bufferSize *cusolverDnCEgels_bufferSize;
tcusolverDnCYgels_bufferSize *cusolverDnCYgels_bufferSize;
tcusolverDnDDgels_bufferSize *cusolverDnDDgels_bufferSize;
tcusolverDnDSgels_bufferSize *cusolverDnDSgels_bufferSize;
tcusolverDnDHgels_bufferSize *cusolverDnDHgels_bufferSize;
tcusolverDnDBgels_bufferSize *cusolverDnDBgels_bufferSize;
tcusolverDnDXgels_bufferSize *cusolverDnDXgels_bufferSize;
tcusolverDnSSgels_bufferSize *cusolverDnSSgels_bufferSize;
tcusolverDnSHgels_bufferSize *cusolverDnSHgels_bufferSize;
tcusolverDnSBgels_bufferSize *cusolverDnSBgels_bufferSize;
tcusolverDnSXgels_bufferSize *cusolverDnSXgels_bufferSize;
tcusolverDnIRSXgesv *cusolverDnIRSXgesv;
tcusolverDnIRSXgesv_bufferSize *cusolverDnIRSXgesv_bufferSize;
tcusolverDnIRSXgels *cusolverDnIRSXgels;
tcusolverDnIRSXgels_bufferSize *cusolverDnIRSXgels_bufferSize;
tcusolverDnSpotrf_bufferSize *cusolverDnSpotrf_bufferSize;
tcusolverDnDpotrf_bufferSize *cusolverDnDpotrf_bufferSize;
tcusolverDnCpotrf_bufferSize *cusolverDnCpotrf_bufferSize;
tcusolverDnZpotrf_bufferSize *cusolverDnZpotrf_bufferSize;
tcusolverDnSpotrf *cusolverDnSpotrf;
tcusolverDnDpotrf *cusolverDnDpotrf;
tcusolverDnCpotrf *cusolverDnCpotrf;
tcusolverDnZpotrf *cusolverDnZpotrf;
tcusolverDnSpotrs *cusolverDnSpotrs;
tcusolverDnDpotrs *cusolverDnDpotrs;
tcusolverDnCpotrs *cusolverDnCpotrs;
tcusolverDnZpotrs *cusolverDnZpotrs;
tcusolverDnSpotrfBatched *cusolverDnSpotrfBatched;
tcusolverDnDpotrfBatched *cusolverDnDpotrfBatched;
tcusolverDnCpotrfBatched *cusolverDnCpotrfBatched;
tcusolverDnZpotrfBatched *cusolverDnZpotrfBatched;
tcusolverDnSpotrsBatched *cusolverDnSpotrsBatched;
tcusolverDnDpotrsBatched *cusolverDnDpotrsBatched;
tcusolverDnCpotrsBatched *cusolverDnCpotrsBatched;
tcusolverDnZpotrsBatched *cusolverDnZpotrsBatched;
tcusolverDnSpotri_bufferSize *cusolverDnSpotri_bufferSize;
tcusolverDnDpotri_bufferSize *cusolverDnDpotri_bufferSize;
tcusolverDnCpotri_bufferSize *cusolverDnCpotri_bufferSize;
tcusolverDnZpotri_bufferSize *cusolverDnZpotri_bufferSize;
tcusolverDnSpotri *cusolverDnSpotri;
tcusolverDnDpotri *cusolverDnDpotri;
tcusolverDnCpotri *cusolverDnCpotri;
tcusolverDnZpotri *cusolverDnZpotri;
tcusolverDnXtrtri_bufferSize *cusolverDnXtrtri_bufferSize;
tcusolverDnXtrtri *cusolverDnXtrtri;
tcusolverDnSlauum_bufferSize *cusolverDnSlauum_bufferSize;
tcusolverDnDlauum_bufferSize *cusolverDnDlauum_bufferSize;
tcusolverDnClauum_bufferSize *cusolverDnClauum_bufferSize;
tcusolverDnZlauum_bufferSize *cusolverDnZlauum_bufferSize;
tcusolverDnSlauum *cusolverDnSlauum;
tcusolverDnDlauum *cusolverDnDlauum;
tcusolverDnClauum *cusolverDnClauum;
tcusolverDnZlauum *cusolverDnZlauum;
tcusolverDnSgetrf_bufferSize *cusolverDnSgetrf_bufferSize;
tcusolverDnDgetrf_bufferSize *cusolverDnDgetrf_bufferSize;
tcusolverDnCgetrf_bufferSize *cusolverDnCgetrf_bufferSize;
tcusolverDnZgetrf_bufferSize *cusolverDnZgetrf_bufferSize;
tcusolverDnSgetrf *cusolverDnSgetrf;
tcusolverDnDgetrf *cusolverDnDgetrf;
tcusolverDnCgetrf *cusolverDnCgetrf;
tcusolverDnZgetrf *cusolverDnZgetrf;
tcusolverDnSlaswp *cusolverDnSlaswp;
tcusolverDnDlaswp *cusolverDnDlaswp;
tcusolverDnClaswp *cusolverDnClaswp;
tcusolverDnZlaswp *cusolverDnZlaswp;
tcusolverDnSgetrs *cusolverDnSgetrs;
tcusolverDnDgetrs *cusolverDnDgetrs;
tcusolverDnCgetrs *cusolverDnCgetrs;
tcusolverDnZgetrs *cusolverDnZgetrs;
tcusolverDnSgeqrf_bufferSize *cusolverDnSgeqrf_bufferSize;
tcusolverDnDgeqrf_bufferSize *cusolverDnDgeqrf_bufferSize;
tcusolverDnCgeqrf_bufferSize *cusolverDnCgeqrf_bufferSize;
tcusolverDnZgeqrf_bufferSize *cusolverDnZgeqrf_bufferSize;
tcusolverDnSgeqrf *cusolverDnSgeqrf;
tcusolverDnDgeqrf *cusolverDnDgeqrf;
tcusolverDnCgeqrf *cusolverDnCgeqrf;
tcusolverDnZgeqrf *cusolverDnZgeqrf;
tcusolverDnSorgqr_bufferSize *cusolverDnSorgqr_bufferSize;
tcusolverDnDorgqr_bufferSize *cusolverDnDorgqr_bufferSize;
tcusolverDnCungqr_bufferSize *cusolverDnCungqr_bufferSize;
tcusolverDnZungqr_bufferSize *cusolverDnZungqr_bufferSize;
tcusolverDnSorgqr *cusolverDnSorgqr;
tcusolverDnDorgqr *cusolverDnDorgqr;
tcusolverDnCungqr *cusolverDnCungqr;
tcusolverDnZungqr *cusolverDnZungqr;
tcusolverDnSormqr_bufferSize *cusolverDnSormqr_bufferSize;
tcusolverDnDormqr_bufferSize *cusolverDnDormqr_bufferSize;
tcusolverDnCunmqr_bufferSize *cusolverDnCunmqr_bufferSize;
tcusolverDnZunmqr_bufferSize *cusolverDnZunmqr_bufferSize;
tcusolverDnSormqr *cusolverDnSormqr;
tcusolverDnDormqr *cusolverDnDormqr;
tcusolverDnCunmqr *cusolverDnCunmqr;
tcusolverDnZunmqr *cusolverDnZunmqr;
tcusolverDnSsytrf_bufferSize *cusolverDnSsytrf_bufferSize;
tcusolverDnDsytrf_bufferSize *cusolverDnDsytrf_bufferSize;
tcusolverDnCsytrf_bufferSize *cusolverDnCsytrf_bufferSize;
tcusolverDnZsytrf_bufferSize *cusolverDnZsytrf_bufferSize;
tcusolverDnSsytrf *cusolverDnSsytrf;
tcusolverDnDsytrf *cusolverDnDsytrf;
tcusolverDnCsytrf *cusolverDnCsytrf;
tcusolverDnZsytrf *cusolverDnZsytrf;
tcusolverDnXsytrs_bufferSize *cusolverDnXsytrs_bufferSize;
tcusolverDnXsytrs *cusolverDnXsytrs;
tcusolverDnSsytri_bufferSize *cusolverDnSsytri_bufferSize;
tcusolverDnDsytri_bufferSize *cusolverDnDsytri_bufferSize;
tcusolverDnCsytri_bufferSize *cusolverDnCsytri_bufferSize;
tcusolverDnZsytri_bufferSize *cusolverDnZsytri_bufferSize;
tcusolverDnSsytri *cusolverDnSsytri;
tcusolverDnDsytri *cusolverDnDsytri;
tcusolverDnCsytri *cusolverDnCsytri;
tcusolverDnZsytri *cusolverDnZsytri;
tcusolverDnSgebrd_bufferSize *cusolverDnSgebrd_bufferSize;
tcusolverDnDgebrd_bufferSize *cusolverDnDgebrd_bufferSize;
tcusolverDnCgebrd_bufferSize *cusolverDnCgebrd_bufferSize;
tcusolverDnZgebrd_bufferSize *cusolverDnZgebrd_bufferSize;
tcusolverDnSgebrd *cusolverDnSgebrd;
tcusolverDnDgebrd *cusolverDnDgebrd;
tcusolverDnCgebrd *cusolverDnCgebrd;
tcusolverDnZgebrd *cusolverDnZgebrd;
tcusolverDnSorgbr_bufferSize *cusolverDnSorgbr_bufferSize;
tcusolverDnDorgbr_bufferSize *cusolverDnDorgbr_bufferSize;
tcusolverDnCungbr_bufferSize *cusolverDnCungbr_bufferSize;
tcusolverDnZungbr_bufferSize *cusolverDnZungbr_bufferSize;
tcusolverDnSorgbr *cusolverDnSorgbr;
tcusolverDnDorgbr *cusolverDnDorgbr;
tcusolverDnCungbr *cusolverDnCungbr;
tcusolverDnZungbr *cusolverDnZungbr;
tcusolverDnSsytrd_bufferSize *cusolverDnSsytrd_bufferSize;
tcusolverDnDsytrd_bufferSize *cusolverDnDsytrd_bufferSize;
tcusolverDnChetrd_bufferSize *cusolverDnChetrd_bufferSize;
tcusolverDnZhetrd_bufferSize *cusolverDnZhetrd_bufferSize;
tcusolverDnSsytrd *cusolverDnSsytrd;
tcusolverDnDsytrd *cusolverDnDsytrd;
tcusolverDnChetrd *cusolverDnChetrd;
tcusolverDnZhetrd *cusolverDnZhetrd;
tcusolverDnSorgtr_bufferSize *cusolverDnSorgtr_bufferSize;
tcusolverDnDorgtr_bufferSize *cusolverDnDorgtr_bufferSize;
tcusolverDnCungtr_bufferSize *cusolverDnCungtr_bufferSize;
tcusolverDnZungtr_bufferSize *cusolverDnZungtr_bufferSize;
tcusolverDnSorgtr *cusolverDnSorgtr;
tcusolverDnDorgtr *cusolverDnDorgtr;
tcusolverDnCungtr *cusolverDnCungtr;
tcusolverDnZungtr *cusolverDnZungtr;
tcusolverDnSormtr_bufferSize *cusolverDnSormtr_bufferSize;
tcusolverDnDormtr_bufferSize *cusolverDnDormtr_bufferSize;
tcusolverDnCunmtr_bufferSize *cusolverDnCunmtr_bufferSize;
tcusolverDnZunmtr_bufferSize *cusolverDnZunmtr_bufferSize;
tcusolverDnSormtr *cusolverDnSormtr;
tcusolverDnDormtr *cusolverDnDormtr;
tcusolverDnCunmtr *cusolverDnCunmtr;
tcusolverDnZunmtr *cusolverDnZunmtr;
tcusolverDnSgesvd_bufferSize *cusolverDnSgesvd_bufferSize;
tcusolverDnDgesvd_bufferSize *cusolverDnDgesvd_bufferSize;
tcusolverDnCgesvd_bufferSize *cusolverDnCgesvd_bufferSize;
tcusolverDnZgesvd_bufferSize *cusolverDnZgesvd_bufferSize;
tcusolverDnSgesvd *cusolverDnSgesvd;
tcusolverDnDgesvd *cusolverDnDgesvd;
tcusolverDnCgesvd *cusolverDnCgesvd;
tcusolverDnZgesvd *cusolverDnZgesvd;
tcusolverDnSsyevd_bufferSize *cusolverDnSsyevd_bufferSize;
tcusolverDnDsyevd_bufferSize *cusolverDnDsyevd_bufferSize;
tcusolverDnCheevd_bufferSize *cusolverDnCheevd_bufferSize;
tcusolverDnZheevd_bufferSize *cusolverDnZheevd_bufferSize;
tcusolverDnSsyevd *cusolverDnSsyevd;
tcusolverDnDsyevd *cusolverDnDsyevd;
tcusolverDnCheevd *cusolverDnCheevd;
tcusolverDnZheevd *cusolverDnZheevd;
tcusolverDnSsyevdx_bufferSize *cusolverDnSsyevdx_bufferSize;
tcusolverDnDsyevdx_bufferSize *cusolverDnDsyevdx_bufferSize;
tcusolverDnCheevdx_bufferSize *cusolverDnCheevdx_bufferSize;
tcusolverDnZheevdx_bufferSize *cusolverDnZheevdx_bufferSize;
tcusolverDnSsyevdx *cusolverDnSsyevdx;
tcusolverDnDsyevdx *cusolverDnDsyevdx;
tcusolverDnCheevdx *cusolverDnCheevdx;
tcusolverDnZheevdx *cusolverDnZheevdx;
tcusolverDnSsygvdx_bufferSize *cusolverDnSsygvdx_bufferSize;
tcusolverDnDsygvdx_bufferSize *cusolverDnDsygvdx_bufferSize;
tcusolverDnChegvdx_bufferSize *cusolverDnChegvdx_bufferSize;
tcusolverDnZhegvdx_bufferSize *cusolverDnZhegvdx_bufferSize;
tcusolverDnSsygvdx *cusolverDnSsygvdx;
tcusolverDnDsygvdx *cusolverDnDsygvdx;
tcusolverDnChegvdx *cusolverDnChegvdx;
tcusolverDnZhegvdx *cusolverDnZhegvdx;
tcusolverDnSsygvd_bufferSize *cusolverDnSsygvd_bufferSize;
tcusolverDnDsygvd_bufferSize *cusolverDnDsygvd_bufferSize;
tcusolverDnChegvd_bufferSize *cusolverDnChegvd_bufferSize;
tcusolverDnZhegvd_bufferSize *cusolverDnZhegvd_bufferSize;
tcusolverDnSsygvd *cusolverDnSsygvd;
tcusolverDnDsygvd *cusolverDnDsygvd;
tcusolverDnChegvd *cusolverDnChegvd;
tcusolverDnZhegvd *cusolverDnZhegvd;
tcusolverDnCreateSyevjInfo *cusolverDnCreateSyevjInfo;
tcusolverDnDestroySyevjInfo *cusolverDnDestroySyevjInfo;
tcusolverDnXsyevjSetTolerance *cusolverDnXsyevjSetTolerance;
tcusolverDnXsyevjSetMaxSweeps *cusolverDnXsyevjSetMaxSweeps;
tcusolverDnXsyevjSetSortEig *cusolverDnXsyevjSetSortEig;
tcusolverDnXsyevjGetResidual *cusolverDnXsyevjGetResidual;
tcusolverDnXsyevjGetSweeps *cusolverDnXsyevjGetSweeps;
tcusolverDnSsyevjBatched_bufferSize *cusolverDnSsyevjBatched_bufferSize;
tcusolverDnDsyevjBatched_bufferSize *cusolverDnDsyevjBatched_bufferSize;
tcusolverDnCheevjBatched_bufferSize *cusolverDnCheevjBatched_bufferSize;
tcusolverDnZheevjBatched_bufferSize *cusolverDnZheevjBatched_bufferSize;
tcusolverDnSsyevjBatched *cusolverDnSsyevjBatched;
tcusolverDnDsyevjBatched *cusolverDnDsyevjBatched;
tcusolverDnCheevjBatched *cusolverDnCheevjBatched;
tcusolverDnZheevjBatched *cusolverDnZheevjBatched;
tcusolverDnSsyevj_bufferSize *cusolverDnSsyevj_bufferSize;
tcusolverDnDsyevj_bufferSize *cusolverDnDsyevj_bufferSize;
tcusolverDnCheevj_bufferSize *cusolverDnCheevj_bufferSize;
tcusolverDnZheevj_bufferSize *cusolverDnZheevj_bufferSize;
tcusolverDnSsyevj *cusolverDnSsyevj;
tcusolverDnDsyevj *cusolverDnDsyevj;
tcusolverDnCheevj *cusolverDnCheevj;
tcusolverDnZheevj *cusolverDnZheevj;
tcusolverDnSsygvj_bufferSize *cusolverDnSsygvj_bufferSize;
tcusolverDnDsygvj_bufferSize *cusolverDnDsygvj_bufferSize;
tcusolverDnChegvj_bufferSize *cusolverDnChegvj_bufferSize;
tcusolverDnZhegvj_bufferSize *cusolverDnZhegvj_bufferSize;
tcusolverDnSsygvj *cusolverDnSsygvj;
tcusolverDnDsygvj *cusolverDnDsygvj;
tcusolverDnChegvj *cusolverDnChegvj;
tcusolverDnZhegvj *cusolverDnZhegvj;
tcusolverDnCreateGesvdjInfo *cusolverDnCreateGesvdjInfo;
tcusolverDnDestroyGesvdjInfo *cusolverDnDestroyGesvdjInfo;
tcusolverDnXgesvdjSetTolerance *cusolverDnXgesvdjSetTolerance;
tcusolverDnXgesvdjSetMaxSweeps *cusolverDnXgesvdjSetMaxSweeps;
tcusolverDnXgesvdjSetSortEig *cusolverDnXgesvdjSetSortEig;
tcusolverDnXgesvdjGetResidual *cusolverDnXgesvdjGetResidual;
tcusolverDnXgesvdjGetSweeps *cusolverDnXgesvdjGetSweeps;
tcusolverDnSgesvdjBatched_bufferSize *cusolverDnSgesvdjBatched_bufferSize;
tcusolverDnDgesvdjBatched_bufferSize *cusolverDnDgesvdjBatched_bufferSize;
tcusolverDnCgesvdjBatched_bufferSize *cusolverDnCgesvdjBatched_bufferSize;
tcusolverDnZgesvdjBatched_bufferSize *cusolverDnZgesvdjBatched_bufferSize;
tcusolverDnSgesvdjBatched *cusolverDnSgesvdjBatched;
tcusolverDnDgesvdjBatched *cusolverDnDgesvdjBatched;
tcusolverDnCgesvdjBatched *cusolverDnCgesvdjBatched;
tcusolverDnZgesvdjBatched *cusolverDnZgesvdjBatched;
tcusolverDnSgesvdj_bufferSize *cusolverDnSgesvdj_bufferSize;
tcusolverDnDgesvdj_bufferSize *cusolverDnDgesvdj_bufferSize;
tcusolverDnCgesvdj_bufferSize *cusolverDnCgesvdj_bufferSize;
tcusolverDnZgesvdj_bufferSize *cusolverDnZgesvdj_bufferSize;
tcusolverDnSgesvdj *cusolverDnSgesvdj;
tcusolverDnDgesvdj *cusolverDnDgesvdj;
tcusolverDnCgesvdj *cusolverDnCgesvdj;
tcusolverDnZgesvdj *cusolverDnZgesvdj;
tcusolverDnSgesvdaStridedBatched_bufferSize *cusolverDnSgesvdaStridedBatched_bufferSize;
tcusolverDnDgesvdaStridedBatched_bufferSize *cusolverDnDgesvdaStridedBatched_bufferSize;
tcusolverDnCgesvdaStridedBatched_bufferSize *cusolverDnCgesvdaStridedBatched_bufferSize;
tcusolverDnZgesvdaStridedBatched_bufferSize *cusolverDnZgesvdaStridedBatched_bufferSize;
tcusolverDnSgesvdaStridedBatched *cusolverDnSgesvdaStridedBatched;
tcusolverDnDgesvdaStridedBatched *cusolverDnDgesvdaStridedBatched;
tcusolverDnCgesvdaStridedBatched *cusolverDnCgesvdaStridedBatched;
tcusolverDnZgesvdaStridedBatched *cusolverDnZgesvdaStridedBatched;
tcusolverDnCreateParams *cusolverDnCreateParams;
tcusolverDnDestroyParams *cusolverDnDestroyParams;
tcusolverDnSetAdvOptions *cusolverDnSetAdvOptions;
tcusolverDnPotrf_bufferSize *cusolverDnPotrf_bufferSize;
tcusolverDnPotrf *cusolverDnPotrf;
tcusolverDnPotrs *cusolverDnPotrs;
tcusolverDnGeqrf_bufferSize *cusolverDnGeqrf_bufferSize;
tcusolverDnGeqrf *cusolverDnGeqrf;
tcusolverDnGetrf_bufferSize *cusolverDnGetrf_bufferSize;
tcusolverDnGetrf *cusolverDnGetrf;
tcusolverDnGetrs *cusolverDnGetrs;
tcusolverDnSyevd_bufferSize *cusolverDnSyevd_bufferSize;
tcusolverDnSyevd *cusolverDnSyevd;
tcusolverDnSyevdx_bufferSize *cusolverDnSyevdx_bufferSize;
tcusolverDnSyevdx *cusolverDnSyevdx;
tcusolverDnGesvd_bufferSize *cusolverDnGesvd_bufferSize;
tcusolverDnGesvd *cusolverDnGesvd;
tcusolverDnXpotrf_bufferSize *cusolverDnXpotrf_bufferSize;
tcusolverDnXpotrf *cusolverDnXpotrf;
tcusolverDnXpotrs *cusolverDnXpotrs;
tcusolverDnXgeqrf_bufferSize *cusolverDnXgeqrf_bufferSize;
tcusolverDnXgeqrf *cusolverDnXgeqrf;
tcusolverDnXgetrf_bufferSize *cusolverDnXgetrf_bufferSize;
tcusolverDnXgetrf *cusolverDnXgetrf;
tcusolverDnXgetrs *cusolverDnXgetrs;
tcusolverDnXsyevd_bufferSize *cusolverDnXsyevd_bufferSize;
tcusolverDnXsyevd *cusolverDnXsyevd;
tcusolverDnXsyevdx_bufferSize *cusolverDnXsyevdx_bufferSize;
tcusolverDnXsyevdx *cusolverDnXsyevdx;
tcusolverDnXgesvd_bufferSize *cusolverDnXgesvd_bufferSize;
tcusolverDnXgesvd *cusolverDnXgesvd;
tcusolverDnXgesvdp_bufferSize *cusolverDnXgesvdp_bufferSize;
tcusolverDnXgesvdp *cusolverDnXgesvdp;
tcusolverDnXgesvdr_bufferSize *cusolverDnXgesvdr_bufferSize;
tcusolverDnXgesvdr *cusolverDnXgesvdr;
tcusolverDnLoggerSetCallback *cusolverDnLoggerSetCallback;
tcusolverDnLoggerSetFile *cusolverDnLoggerSetFile;
tcusolverDnLoggerOpenFile *cusolverDnLoggerOpenFile;
tcusolverDnLoggerSetLevel *cusolverDnLoggerSetLevel;
tcusolverDnLoggerSetMask *cusolverDnLoggerSetMask;
tcusolverDnLoggerForceDisable *cusolverDnLoggerForceDisable;

int cuewInitCUSOLVER(const char **extra_dll_search_paths) {

#ifdef _WIN32
  const char *paths[] = {   "cusolver.dll",
NULL};
#else /* linux */
  const char *paths[] = {   "libcusolver.so",
   "/usr/local/cuda/lib64/libcusolver.so",
NULL};
#endif


  static int initialized = 0;
  static int result = 0;
  int error;

  if (initialized) {
    return result;
  }

  initialized = 1;
  error = atexit(cuewExitCUSOLVER);

  if (error) {
    result = -2;
    return result;
  }
  cusolver_lib = dynamic_library_open_find(paths);
  if (cusolver_lib == NULL) { 
    if (extra_dll_search_paths) { 
      cusolver_lib = dynamic_library_open_find(extra_dll_search_paths);
    }
  }
  if (cusolver_lib == NULL) { result = -1; return result; }

  CUSOLVER_LIBRARY_FIND(cusolverGetProperty)
  CUSOLVER_LIBRARY_FIND(cusolverGetVersion)
  CUSOLVER_LIBRARY_FIND(cusolverRfCreate)
  CUSOLVER_LIBRARY_FIND(cusolverRfDestroy)
  CUSOLVER_LIBRARY_FIND(cusolverRfGetMatrixFormat)
  CUSOLVER_LIBRARY_FIND(cusolverRfSetMatrixFormat)
  CUSOLVER_LIBRARY_FIND(cusolverRfSetNumericProperties)
  CUSOLVER_LIBRARY_FIND(cusolverRfGetNumericProperties)
  CUSOLVER_LIBRARY_FIND(cusolverRfGetNumericBoostReport)
  CUSOLVER_LIBRARY_FIND(cusolverRfSetAlgs)
  CUSOLVER_LIBRARY_FIND(cusolverRfGetAlgs)
  CUSOLVER_LIBRARY_FIND(cusolverRfGetResetValuesFastMode)
  CUSOLVER_LIBRARY_FIND(cusolverRfSetResetValuesFastMode)
  CUSOLVER_LIBRARY_FIND(cusolverRfSetupHost)
  CUSOLVER_LIBRARY_FIND(cusolverRfSetupDevice)
  CUSOLVER_LIBRARY_FIND(cusolverRfResetValues)
  CUSOLVER_LIBRARY_FIND(cusolverRfAnalyze)
  CUSOLVER_LIBRARY_FIND(cusolverRfRefactor)
  CUSOLVER_LIBRARY_FIND(cusolverRfAccessBundledFactorsDevice)
  CUSOLVER_LIBRARY_FIND(cusolverRfExtractBundledFactorsHost)
  CUSOLVER_LIBRARY_FIND(cusolverRfExtractSplitFactorsHost)
  CUSOLVER_LIBRARY_FIND(cusolverRfSolve)
  CUSOLVER_LIBRARY_FIND(cusolverRfBatchSetupHost)
  CUSOLVER_LIBRARY_FIND(cusolverRfBatchResetValues)
  CUSOLVER_LIBRARY_FIND(cusolverRfBatchAnalyze)
  CUSOLVER_LIBRARY_FIND(cusolverRfBatchRefactor)
  CUSOLVER_LIBRARY_FIND(cusolverRfBatchSolve)
  CUSOLVER_LIBRARY_FIND(cusolverRfBatchZeroPivot)
  CUSOLVER_LIBRARY_FIND(cusolverSpCreate)
  CUSOLVER_LIBRARY_FIND(cusolverSpDestroy)
  CUSOLVER_LIBRARY_FIND(cusolverSpSetStream)
  CUSOLVER_LIBRARY_FIND(cusolverSpGetStream)
  CUSOLVER_LIBRARY_FIND(cusolverSpXcsrissymHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrlsvluHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrlsvluHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrlsvluHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrlsvluHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrlsvqr)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrlsvqr)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrlsvqr)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrlsvqr)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrlsvqrHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrlsvqrHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrlsvqrHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrlsvqrHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrlsvcholHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrlsvcholHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrlsvcholHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrlsvcholHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrlsvchol)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrlsvchol)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrlsvchol)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrlsvchol)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrlsqvqrHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrlsqvqrHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrlsqvqrHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrlsqvqrHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsreigvsiHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsreigvsiHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsreigvsiHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsreigvsiHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsreigvsi)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsreigvsi)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsreigvsi)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsreigvsi)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsreigsHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsreigsHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsreigsHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsreigsHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpXcsrsymrcmHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpXcsrsymmdqHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpXcsrsymamdHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpXcsrmetisndHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrzfdHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrzfdHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrzfdHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrzfdHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpXcsrperm_bufferSizeHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpXcsrpermHost)
  CUSOLVER_LIBRARY_FIND(cusolverSpCreateCsrqrInfo)
  CUSOLVER_LIBRARY_FIND(cusolverSpDestroyCsrqrInfo)
  CUSOLVER_LIBRARY_FIND(cusolverSpXcsrqrAnalysisBatched)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrqrBufferInfoBatched)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrqrBufferInfoBatched)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrqrBufferInfoBatched)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrqrBufferInfoBatched)
  CUSOLVER_LIBRARY_FIND(cusolverSpScsrqrsvBatched)
  CUSOLVER_LIBRARY_FIND(cusolverSpDcsrqrsvBatched)
  CUSOLVER_LIBRARY_FIND(cusolverSpCcsrqrsvBatched)
  CUSOLVER_LIBRARY_FIND(cusolverSpZcsrqrsvBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnCreate)
  CUSOLVER_LIBRARY_FIND(cusolverDnDestroy)
  CUSOLVER_LIBRARY_FIND(cusolverDnSetStream)
  CUSOLVER_LIBRARY_FIND(cusolverDnGetStream)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsCreate)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsDestroy)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsSetRefinementSolver)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsSetSolverMainPrecision)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsSetSolverLowestPrecision)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsSetSolverPrecisions)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsSetTol)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsSetTolInner)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsSetMaxIters)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsSetMaxItersInner)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsGetMaxIters)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsEnableFallback)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSParamsDisableFallback)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSInfosDestroy)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSInfosCreate)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSInfosGetNiters)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSInfosGetOuterNiters)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSInfosRequestResidual)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSInfosGetResidualHistory)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSInfosGetMaxIters)
  CUSOLVER_LIBRARY_FIND(cusolverDnZZgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnZCgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnZKgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnZEgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnZYgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnCCgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnCEgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnCKgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnCYgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnDDgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnDSgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnDHgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnDBgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnDXgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnSSgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnSHgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnSBgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnSXgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnZZgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZCgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZKgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZEgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZYgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCCgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCKgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCEgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCYgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDDgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDSgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDHgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDBgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDXgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSSgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSHgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSBgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSXgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZZgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnZCgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnZKgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnZEgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnZYgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnCCgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnCKgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnCEgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnCYgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnDDgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnDSgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnDHgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnDBgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnDXgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnSSgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnSHgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnSBgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnSXgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnZZgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZCgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZKgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZEgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZYgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCCgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCKgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCEgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCYgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDDgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDSgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDHgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDBgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDXgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSSgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSHgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSBgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSXgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSXgesv)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSXgesv_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSXgels)
  CUSOLVER_LIBRARY_FIND(cusolverDnIRSXgels_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSpotrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDpotrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCpotrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZpotrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSpotrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnDpotrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnCpotrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnZpotrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnSpotrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnDpotrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnCpotrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnZpotrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnSpotrfBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnDpotrfBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnCpotrfBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnZpotrfBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnSpotrsBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnDpotrsBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnCpotrsBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnZpotrsBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnSpotri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDpotri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCpotri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZpotri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSpotri)
  CUSOLVER_LIBRARY_FIND(cusolverDnDpotri)
  CUSOLVER_LIBRARY_FIND(cusolverDnCpotri)
  CUSOLVER_LIBRARY_FIND(cusolverDnZpotri)
  CUSOLVER_LIBRARY_FIND(cusolverDnXtrtri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXtrtri)
  CUSOLVER_LIBRARY_FIND(cusolverDnSlauum_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDlauum_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnClauum_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZlauum_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSlauum)
  CUSOLVER_LIBRARY_FIND(cusolverDnDlauum)
  CUSOLVER_LIBRARY_FIND(cusolverDnClauum)
  CUSOLVER_LIBRARY_FIND(cusolverDnZlauum)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgetrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgetrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgetrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgetrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgetrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgetrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgetrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgetrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnSlaswp)
  CUSOLVER_LIBRARY_FIND(cusolverDnDlaswp)
  CUSOLVER_LIBRARY_FIND(cusolverDnClaswp)
  CUSOLVER_LIBRARY_FIND(cusolverDnZlaswp)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgetrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgetrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgetrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgetrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgeqrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgeqrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgeqrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgeqrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgeqrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgeqrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgeqrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgeqrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnSorgqr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDorgqr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCungqr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZungqr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSorgqr)
  CUSOLVER_LIBRARY_FIND(cusolverDnDorgqr)
  CUSOLVER_LIBRARY_FIND(cusolverDnCungqr)
  CUSOLVER_LIBRARY_FIND(cusolverDnZungqr)
  CUSOLVER_LIBRARY_FIND(cusolverDnSormqr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDormqr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCunmqr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZunmqr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSormqr)
  CUSOLVER_LIBRARY_FIND(cusolverDnDormqr)
  CUSOLVER_LIBRARY_FIND(cusolverDnCunmqr)
  CUSOLVER_LIBRARY_FIND(cusolverDnZunmqr)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsytrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsytrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCsytrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZsytrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsytrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsytrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnCsytrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnZsytrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsytrs_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsytrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsytri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsytri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCsytri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZsytri_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsytri)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsytri)
  CUSOLVER_LIBRARY_FIND(cusolverDnCsytri)
  CUSOLVER_LIBRARY_FIND(cusolverDnZsytri)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgebrd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgebrd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgebrd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgebrd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgebrd)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgebrd)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgebrd)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgebrd)
  CUSOLVER_LIBRARY_FIND(cusolverDnSorgbr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDorgbr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCungbr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZungbr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSorgbr)
  CUSOLVER_LIBRARY_FIND(cusolverDnDorgbr)
  CUSOLVER_LIBRARY_FIND(cusolverDnCungbr)
  CUSOLVER_LIBRARY_FIND(cusolverDnZungbr)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsytrd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsytrd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnChetrd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZhetrd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsytrd)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsytrd)
  CUSOLVER_LIBRARY_FIND(cusolverDnChetrd)
  CUSOLVER_LIBRARY_FIND(cusolverDnZhetrd)
  CUSOLVER_LIBRARY_FIND(cusolverDnSorgtr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDorgtr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCungtr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZungtr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSorgtr)
  CUSOLVER_LIBRARY_FIND(cusolverDnDorgtr)
  CUSOLVER_LIBRARY_FIND(cusolverDnCungtr)
  CUSOLVER_LIBRARY_FIND(cusolverDnZungtr)
  CUSOLVER_LIBRARY_FIND(cusolverDnSormtr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDormtr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCunmtr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZunmtr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSormtr)
  CUSOLVER_LIBRARY_FIND(cusolverDnDormtr)
  CUSOLVER_LIBRARY_FIND(cusolverDnCunmtr)
  CUSOLVER_LIBRARY_FIND(cusolverDnZunmtr)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgesvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgesvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgesvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgesvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgesvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgesvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgesvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgesvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsyevd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsyevd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCheevd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZheevd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsyevd)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsyevd)
  CUSOLVER_LIBRARY_FIND(cusolverDnCheevd)
  CUSOLVER_LIBRARY_FIND(cusolverDnZheevd)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsyevdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsyevdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCheevdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZheevdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsyevdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsyevdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnCheevdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnZheevdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsygvdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsygvdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnChegvdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZhegvdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsygvdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsygvdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnChegvdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnZhegvdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsygvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsygvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnChegvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZhegvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsygvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsygvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnChegvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnZhegvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnCreateSyevjInfo)
  CUSOLVER_LIBRARY_FIND(cusolverDnDestroySyevjInfo)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevjSetTolerance)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevjSetMaxSweeps)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevjSetSortEig)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevjGetResidual)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevjGetSweeps)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsyevjBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsyevjBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCheevjBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZheevjBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsyevjBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsyevjBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnCheevjBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnZheevjBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsyevj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsyevj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCheevj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZheevj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsyevj)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsyevj)
  CUSOLVER_LIBRARY_FIND(cusolverDnCheevj)
  CUSOLVER_LIBRARY_FIND(cusolverDnZheevj)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsygvj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsygvj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnChegvj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZhegvj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSsygvj)
  CUSOLVER_LIBRARY_FIND(cusolverDnDsygvj)
  CUSOLVER_LIBRARY_FIND(cusolverDnChegvj)
  CUSOLVER_LIBRARY_FIND(cusolverDnZhegvj)
  CUSOLVER_LIBRARY_FIND(cusolverDnCreateGesvdjInfo)
  CUSOLVER_LIBRARY_FIND(cusolverDnDestroyGesvdjInfo)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdjSetTolerance)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdjSetMaxSweeps)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdjSetSortEig)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdjGetResidual)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdjGetSweeps)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgesvdjBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgesvdjBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgesvdjBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgesvdjBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgesvdjBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgesvdjBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgesvdjBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgesvdjBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgesvdj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgesvdj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgesvdj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgesvdj_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgesvdj)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgesvdj)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgesvdj)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgesvdj)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgesvdaStridedBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgesvdaStridedBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgesvdaStridedBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgesvdaStridedBatched_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSgesvdaStridedBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnDgesvdaStridedBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnCgesvdaStridedBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnZgesvdaStridedBatched)
  CUSOLVER_LIBRARY_FIND(cusolverDnCreateParams)
  CUSOLVER_LIBRARY_FIND(cusolverDnDestroyParams)
  CUSOLVER_LIBRARY_FIND(cusolverDnSetAdvOptions)
  CUSOLVER_LIBRARY_FIND(cusolverDnPotrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnPotrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnPotrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnGeqrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnGeqrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnGetrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnGetrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnGetrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnSyevd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSyevd)
  CUSOLVER_LIBRARY_FIND(cusolverDnSyevdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnSyevdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnGesvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnGesvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnXpotrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXpotrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnXpotrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgeqrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgeqrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgetrf_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgetrf)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgetrs)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevd)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevdx_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXsyevdx)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvd_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvd)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdp_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdp)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdr_bufferSize)
  CUSOLVER_LIBRARY_FIND(cusolverDnXgesvdr)
  CUSOLVER_LIBRARY_FIND(cusolverDnLoggerSetCallback)
  CUSOLVER_LIBRARY_FIND(cusolverDnLoggerSetFile)
  CUSOLVER_LIBRARY_FIND(cusolverDnLoggerOpenFile)
  CUSOLVER_LIBRARY_FIND(cusolverDnLoggerSetLevel)
  CUSOLVER_LIBRARY_FIND(cusolverDnLoggerSetMask)
  CUSOLVER_LIBRARY_FIND(cusolverDnLoggerForceDisable)
  result = 0; // success
  return result;
}
