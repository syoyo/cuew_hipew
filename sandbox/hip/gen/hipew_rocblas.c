

#ifdef _MSC_VER
#  if _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#  define popen _popen
#  define pclose _pclose
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include "hipew.h"
#include "hipew_rocblas.h"
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
#define HIPEW_IMPL_LIBRARY_FIND_CHECKED(lib, name)         name = (t##name *)dynamic_library_find(lib, #name);         assert(name);
*/

#define HIPEW_IMPL_LIBRARY_FIND(lib, name)         name = (t##name *)dynamic_library_find(lib, #name);


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

/*#define ROCBLAS_LIBRARY_FIND_CHECKED(name) HIPEW_IMPL_LIBRARY_FIND_CHECKED(rocblas_lib, name)*/
#define ROCBLAS_LIBRARY_FIND(name) HIPEW_IMPL_LIBRARY_FIND(rocblas_lib, name)
static DynamicLibrary rocblas_lib;

static void hipewExitROCBLAS(void) {
  if (rocblas_lib != NULL) {
    /* ignore errors */
    dynamic_library_close(rocblas_lib);
    rocblas_lib = NULL;
  }
}

trocblas_create_handle *rocblas_create_handle;
trocblas_destroy_handle *rocblas_destroy_handle;
trocblas_set_stream *rocblas_set_stream;
trocblas_get_stream *rocblas_get_stream;
trocblas_set_pointer_mode *rocblas_set_pointer_mode;
trocblas_get_pointer_mode *rocblas_get_pointer_mode;
trocblas_set_int8_type_for_hipblas *rocblas_set_int8_type_for_hipblas;
trocblas_get_int8_type_for_hipblas *rocblas_get_int8_type_for_hipblas;
trocblas_set_atomics_mode *rocblas_set_atomics_mode;
trocblas_get_atomics_mode *rocblas_get_atomics_mode;
trocblas_query_int8_layout_flag *rocblas_query_int8_layout_flag;
trocblas_pointer_to_mode *rocblas_pointer_to_mode;
trocblas_set_vector *rocblas_set_vector;
trocblas_get_vector *rocblas_get_vector;
trocblas_set_matrix *rocblas_set_matrix;
trocblas_get_matrix *rocblas_get_matrix;
trocblas_set_vector_async *rocblas_set_vector_async;
trocblas_get_vector_async *rocblas_get_vector_async;
trocblas_set_matrix_async *rocblas_set_matrix_async;
trocblas_get_matrix_async *rocblas_get_matrix_async;
trocblas_set_start_stop_events *rocblas_set_start_stop_events;
trocblas_set_solution_fitness_query *rocblas_set_solution_fitness_query;
trocblas_set_performance_metric *rocblas_set_performance_metric;
trocblas_get_performance_metric *rocblas_get_performance_metric;
trocblas_sscal *rocblas_sscal;
trocblas_dscal *rocblas_dscal;
trocblas_cscal *rocblas_cscal;
trocblas_zscal *rocblas_zscal;
trocblas_csscal *rocblas_csscal;
trocblas_zdscal *rocblas_zdscal;
trocblas_sscal_batched *rocblas_sscal_batched;
trocblas_dscal_batched *rocblas_dscal_batched;
trocblas_cscal_batched *rocblas_cscal_batched;
trocblas_zscal_batched *rocblas_zscal_batched;
trocblas_csscal_batched *rocblas_csscal_batched;
trocblas_zdscal_batched *rocblas_zdscal_batched;
trocblas_sscal_strided_batched *rocblas_sscal_strided_batched;
trocblas_dscal_strided_batched *rocblas_dscal_strided_batched;
trocblas_cscal_strided_batched *rocblas_cscal_strided_batched;
trocblas_zscal_strided_batched *rocblas_zscal_strided_batched;
trocblas_csscal_strided_batched *rocblas_csscal_strided_batched;
trocblas_zdscal_strided_batched *rocblas_zdscal_strided_batched;
trocblas_scopy *rocblas_scopy;
trocblas_dcopy *rocblas_dcopy;
trocblas_ccopy *rocblas_ccopy;
trocblas_zcopy *rocblas_zcopy;
trocblas_scopy_batched *rocblas_scopy_batched;
trocblas_dcopy_batched *rocblas_dcopy_batched;
trocblas_ccopy_batched *rocblas_ccopy_batched;
trocblas_zcopy_batched *rocblas_zcopy_batched;
trocblas_scopy_strided_batched *rocblas_scopy_strided_batched;
trocblas_dcopy_strided_batched *rocblas_dcopy_strided_batched;
trocblas_ccopy_strided_batched *rocblas_ccopy_strided_batched;
trocblas_zcopy_strided_batched *rocblas_zcopy_strided_batched;
trocblas_sdot *rocblas_sdot;
trocblas_ddot *rocblas_ddot;
trocblas_hdot *rocblas_hdot;
trocblas_bfdot *rocblas_bfdot;
trocblas_cdotu *rocblas_cdotu;
trocblas_zdotu *rocblas_zdotu;
trocblas_cdotc *rocblas_cdotc;
trocblas_zdotc *rocblas_zdotc;
trocblas_sdot_batched *rocblas_sdot_batched;
trocblas_ddot_batched *rocblas_ddot_batched;
trocblas_hdot_batched *rocblas_hdot_batched;
trocblas_bfdot_batched *rocblas_bfdot_batched;
trocblas_cdotu_batched *rocblas_cdotu_batched;
trocblas_zdotu_batched *rocblas_zdotu_batched;
trocblas_cdotc_batched *rocblas_cdotc_batched;
trocblas_zdotc_batched *rocblas_zdotc_batched;
trocblas_sdot_strided_batched *rocblas_sdot_strided_batched;
trocblas_ddot_strided_batched *rocblas_ddot_strided_batched;
trocblas_hdot_strided_batched *rocblas_hdot_strided_batched;
trocblas_bfdot_strided_batched *rocblas_bfdot_strided_batched;
trocblas_cdotu_strided_batched *rocblas_cdotu_strided_batched;
trocblas_zdotu_strided_batched *rocblas_zdotu_strided_batched;
trocblas_cdotc_strided_batched *rocblas_cdotc_strided_batched;
trocblas_zdotc_strided_batched *rocblas_zdotc_strided_batched;
trocblas_sswap *rocblas_sswap;
trocblas_dswap *rocblas_dswap;
trocblas_cswap *rocblas_cswap;
trocblas_zswap *rocblas_zswap;
trocblas_sswap_batched *rocblas_sswap_batched;
trocblas_dswap_batched *rocblas_dswap_batched;
trocblas_cswap_batched *rocblas_cswap_batched;
trocblas_zswap_batched *rocblas_zswap_batched;
trocblas_sswap_strided_batched *rocblas_sswap_strided_batched;
trocblas_dswap_strided_batched *rocblas_dswap_strided_batched;
trocblas_cswap_strided_batched *rocblas_cswap_strided_batched;
trocblas_zswap_strided_batched *rocblas_zswap_strided_batched;
trocblas_saxpy *rocblas_saxpy;
trocblas_daxpy *rocblas_daxpy;
trocblas_haxpy *rocblas_haxpy;
trocblas_caxpy *rocblas_caxpy;
trocblas_zaxpy *rocblas_zaxpy;
trocblas_haxpy_batched *rocblas_haxpy_batched;
trocblas_saxpy_batched *rocblas_saxpy_batched;
trocblas_daxpy_batched *rocblas_daxpy_batched;
trocblas_caxpy_batched *rocblas_caxpy_batched;
trocblas_zaxpy_batched *rocblas_zaxpy_batched;
trocblas_haxpy_strided_batched *rocblas_haxpy_strided_batched;
trocblas_saxpy_strided_batched *rocblas_saxpy_strided_batched;
trocblas_daxpy_strided_batched *rocblas_daxpy_strided_batched;
trocblas_caxpy_strided_batched *rocblas_caxpy_strided_batched;
trocblas_zaxpy_strided_batched *rocblas_zaxpy_strided_batched;
trocblas_sasum *rocblas_sasum;
trocblas_dasum *rocblas_dasum;
trocblas_scasum *rocblas_scasum;
trocblas_dzasum *rocblas_dzasum;
trocblas_sasum_batched *rocblas_sasum_batched;
trocblas_dasum_batched *rocblas_dasum_batched;
trocblas_scasum_batched *rocblas_scasum_batched;
trocblas_dzasum_batched *rocblas_dzasum_batched;
trocblas_sasum_strided_batched *rocblas_sasum_strided_batched;
trocblas_dasum_strided_batched *rocblas_dasum_strided_batched;
trocblas_scasum_strided_batched *rocblas_scasum_strided_batched;
trocblas_dzasum_strided_batched *rocblas_dzasum_strided_batched;
trocblas_snrm2 *rocblas_snrm2;
trocblas_dnrm2 *rocblas_dnrm2;
trocblas_scnrm2 *rocblas_scnrm2;
trocblas_dznrm2 *rocblas_dznrm2;
trocblas_snrm2_batched *rocblas_snrm2_batched;
trocblas_dnrm2_batched *rocblas_dnrm2_batched;
trocblas_scnrm2_batched *rocblas_scnrm2_batched;
trocblas_dznrm2_batched *rocblas_dznrm2_batched;
trocblas_snrm2_strided_batched *rocblas_snrm2_strided_batched;
trocblas_dnrm2_strided_batched *rocblas_dnrm2_strided_batched;
trocblas_scnrm2_strided_batched *rocblas_scnrm2_strided_batched;
trocblas_dznrm2_strided_batched *rocblas_dznrm2_strided_batched;
trocblas_isamax *rocblas_isamax;
trocblas_idamax *rocblas_idamax;
trocblas_icamax *rocblas_icamax;
trocblas_izamax *rocblas_izamax;
trocblas_isamax_batched *rocblas_isamax_batched;
trocblas_idamax_batched *rocblas_idamax_batched;
trocblas_icamax_batched *rocblas_icamax_batched;
trocblas_izamax_batched *rocblas_izamax_batched;
trocblas_isamax_strided_batched *rocblas_isamax_strided_batched;
trocblas_idamax_strided_batched *rocblas_idamax_strided_batched;
trocblas_icamax_strided_batched *rocblas_icamax_strided_batched;
trocblas_izamax_strided_batched *rocblas_izamax_strided_batched;
trocblas_isamin *rocblas_isamin;
trocblas_idamin *rocblas_idamin;
trocblas_icamin *rocblas_icamin;
trocblas_izamin *rocblas_izamin;
trocblas_isamin_batched *rocblas_isamin_batched;
trocblas_idamin_batched *rocblas_idamin_batched;
trocblas_icamin_batched *rocblas_icamin_batched;
trocblas_izamin_batched *rocblas_izamin_batched;
trocblas_isamin_strided_batched *rocblas_isamin_strided_batched;
trocblas_idamin_strided_batched *rocblas_idamin_strided_batched;
trocblas_icamin_strided_batched *rocblas_icamin_strided_batched;
trocblas_izamin_strided_batched *rocblas_izamin_strided_batched;
trocblas_srot *rocblas_srot;
trocblas_drot *rocblas_drot;
trocblas_crot *rocblas_crot;
trocblas_csrot *rocblas_csrot;
trocblas_zrot *rocblas_zrot;
trocblas_zdrot *rocblas_zdrot;
trocblas_srot_batched *rocblas_srot_batched;
trocblas_drot_batched *rocblas_drot_batched;
trocblas_crot_batched *rocblas_crot_batched;
trocblas_csrot_batched *rocblas_csrot_batched;
trocblas_zrot_batched *rocblas_zrot_batched;
trocblas_zdrot_batched *rocblas_zdrot_batched;
trocblas_srot_strided_batched *rocblas_srot_strided_batched;
trocblas_drot_strided_batched *rocblas_drot_strided_batched;
trocblas_crot_strided_batched *rocblas_crot_strided_batched;
trocblas_csrot_strided_batched *rocblas_csrot_strided_batched;
trocblas_zrot_strided_batched *rocblas_zrot_strided_batched;
trocblas_zdrot_strided_batched *rocblas_zdrot_strided_batched;
trocblas_srotg *rocblas_srotg;
trocblas_drotg *rocblas_drotg;
trocblas_crotg *rocblas_crotg;
trocblas_zrotg *rocblas_zrotg;
trocblas_srotg_batched *rocblas_srotg_batched;
trocblas_drotg_batched *rocblas_drotg_batched;
trocblas_crotg_batched *rocblas_crotg_batched;
trocblas_zrotg_batched *rocblas_zrotg_batched;
trocblas_srotg_strided_batched *rocblas_srotg_strided_batched;
trocblas_drotg_strided_batched *rocblas_drotg_strided_batched;
trocblas_crotg_strided_batched *rocblas_crotg_strided_batched;
trocblas_zrotg_strided_batched *rocblas_zrotg_strided_batched;
trocblas_srotm *rocblas_srotm;
trocblas_drotm *rocblas_drotm;
trocblas_srotm_batched *rocblas_srotm_batched;
trocblas_drotm_batched *rocblas_drotm_batched;
trocblas_srotm_strided_batched *rocblas_srotm_strided_batched;
trocblas_drotm_strided_batched *rocblas_drotm_strided_batched;
trocblas_srotmg *rocblas_srotmg;
trocblas_drotmg *rocblas_drotmg;
trocblas_srotmg_batched *rocblas_srotmg_batched;
trocblas_drotmg_batched *rocblas_drotmg_batched;
trocblas_srotmg_strided_batched *rocblas_srotmg_strided_batched;
trocblas_drotmg_strided_batched *rocblas_drotmg_strided_batched;
trocblas_sgbmv *rocblas_sgbmv;
trocblas_dgbmv *rocblas_dgbmv;
trocblas_cgbmv *rocblas_cgbmv;
trocblas_zgbmv *rocblas_zgbmv;
trocblas_sgbmv_batched *rocblas_sgbmv_batched;
trocblas_dgbmv_batched *rocblas_dgbmv_batched;
trocblas_cgbmv_batched *rocblas_cgbmv_batched;
trocblas_zgbmv_batched *rocblas_zgbmv_batched;
trocblas_sgbmv_strided_batched *rocblas_sgbmv_strided_batched;
trocblas_dgbmv_strided_batched *rocblas_dgbmv_strided_batched;
trocblas_cgbmv_strided_batched *rocblas_cgbmv_strided_batched;
trocblas_zgbmv_strided_batched *rocblas_zgbmv_strided_batched;
trocblas_sgemv *rocblas_sgemv;
trocblas_dgemv *rocblas_dgemv;
trocblas_cgemv *rocblas_cgemv;
trocblas_zgemv *rocblas_zgemv;
trocblas_sgemv_batched *rocblas_sgemv_batched;
trocblas_dgemv_batched *rocblas_dgemv_batched;
trocblas_cgemv_batched *rocblas_cgemv_batched;
trocblas_zgemv_batched *rocblas_zgemv_batched;
trocblas_sgemv_strided_batched *rocblas_sgemv_strided_batched;
trocblas_dgemv_strided_batched *rocblas_dgemv_strided_batched;
trocblas_cgemv_strided_batched *rocblas_cgemv_strided_batched;
trocblas_zgemv_strided_batched *rocblas_zgemv_strided_batched;
trocblas_chbmv *rocblas_chbmv;
trocblas_zhbmv *rocblas_zhbmv;
trocblas_chbmv_batched *rocblas_chbmv_batched;
trocblas_zhbmv_batched *rocblas_zhbmv_batched;
trocblas_chbmv_strided_batched *rocblas_chbmv_strided_batched;
trocblas_zhbmv_strided_batched *rocblas_zhbmv_strided_batched;
trocblas_chemv *rocblas_chemv;
trocblas_zhemv *rocblas_zhemv;
trocblas_chemv_batched *rocblas_chemv_batched;
trocblas_zhemv_batched *rocblas_zhemv_batched;
trocblas_chemv_strided_batched *rocblas_chemv_strided_batched;
trocblas_zhemv_strided_batched *rocblas_zhemv_strided_batched;
trocblas_cher *rocblas_cher;
trocblas_zher *rocblas_zher;
trocblas_cher_batched *rocblas_cher_batched;
trocblas_zher_batched *rocblas_zher_batched;
trocblas_cher_strided_batched *rocblas_cher_strided_batched;
trocblas_zher_strided_batched *rocblas_zher_strided_batched;
trocblas_cher2 *rocblas_cher2;
trocblas_zher2 *rocblas_zher2;
trocblas_cher2_batched *rocblas_cher2_batched;
trocblas_zher2_batched *rocblas_zher2_batched;
trocblas_cher2_strided_batched *rocblas_cher2_strided_batched;
trocblas_zher2_strided_batched *rocblas_zher2_strided_batched;
trocblas_chpmv *rocblas_chpmv;
trocblas_zhpmv *rocblas_zhpmv;
trocblas_chpmv_batched *rocblas_chpmv_batched;
trocblas_zhpmv_batched *rocblas_zhpmv_batched;
trocblas_chpmv_strided_batched *rocblas_chpmv_strided_batched;
trocblas_zhpmv_strided_batched *rocblas_zhpmv_strided_batched;
trocblas_chpr *rocblas_chpr;
trocblas_zhpr *rocblas_zhpr;
trocblas_chpr_batched *rocblas_chpr_batched;
trocblas_zhpr_batched *rocblas_zhpr_batched;
trocblas_chpr_strided_batched *rocblas_chpr_strided_batched;
trocblas_zhpr_strided_batched *rocblas_zhpr_strided_batched;
trocblas_chpr2 *rocblas_chpr2;
trocblas_zhpr2 *rocblas_zhpr2;
trocblas_chpr2_batched *rocblas_chpr2_batched;
trocblas_zhpr2_batched *rocblas_zhpr2_batched;
trocblas_chpr2_strided_batched *rocblas_chpr2_strided_batched;
trocblas_zhpr2_strided_batched *rocblas_zhpr2_strided_batched;
trocblas_strmv *rocblas_strmv;
trocblas_dtrmv *rocblas_dtrmv;
trocblas_ctrmv *rocblas_ctrmv;
trocblas_ztrmv *rocblas_ztrmv;
trocblas_strmv_batched *rocblas_strmv_batched;
trocblas_dtrmv_batched *rocblas_dtrmv_batched;
trocblas_ctrmv_batched *rocblas_ctrmv_batched;
trocblas_ztrmv_batched *rocblas_ztrmv_batched;
trocblas_strmv_strided_batched *rocblas_strmv_strided_batched;
trocblas_dtrmv_strided_batched *rocblas_dtrmv_strided_batched;
trocblas_ctrmv_strided_batched *rocblas_ctrmv_strided_batched;
trocblas_ztrmv_strided_batched *rocblas_ztrmv_strided_batched;
trocblas_stpmv *rocblas_stpmv;
trocblas_dtpmv *rocblas_dtpmv;
trocblas_ctpmv *rocblas_ctpmv;
trocblas_ztpmv *rocblas_ztpmv;
trocblas_stpmv_batched *rocblas_stpmv_batched;
trocblas_dtpmv_batched *rocblas_dtpmv_batched;
trocblas_ctpmv_batched *rocblas_ctpmv_batched;
trocblas_ztpmv_batched *rocblas_ztpmv_batched;
trocblas_stpmv_strided_batched *rocblas_stpmv_strided_batched;
trocblas_dtpmv_strided_batched *rocblas_dtpmv_strided_batched;
trocblas_ctpmv_strided_batched *rocblas_ctpmv_strided_batched;
trocblas_ztpmv_strided_batched *rocblas_ztpmv_strided_batched;
trocblas_stbmv *rocblas_stbmv;
trocblas_dtbmv *rocblas_dtbmv;
trocblas_ctbmv *rocblas_ctbmv;
trocblas_ztbmv *rocblas_ztbmv;
trocblas_stbmv_batched *rocblas_stbmv_batched;
trocblas_dtbmv_batched *rocblas_dtbmv_batched;
trocblas_ctbmv_batched *rocblas_ctbmv_batched;
trocblas_ztbmv_batched *rocblas_ztbmv_batched;
trocblas_stbmv_strided_batched *rocblas_stbmv_strided_batched;
trocblas_dtbmv_strided_batched *rocblas_dtbmv_strided_batched;
trocblas_ctbmv_strided_batched *rocblas_ctbmv_strided_batched;
trocblas_ztbmv_strided_batched *rocblas_ztbmv_strided_batched;
trocblas_stbsv *rocblas_stbsv;
trocblas_dtbsv *rocblas_dtbsv;
trocblas_ctbsv *rocblas_ctbsv;
trocblas_ztbsv *rocblas_ztbsv;
trocblas_stbsv_batched *rocblas_stbsv_batched;
trocblas_dtbsv_batched *rocblas_dtbsv_batched;
trocblas_ctbsv_batched *rocblas_ctbsv_batched;
trocblas_ztbsv_batched *rocblas_ztbsv_batched;
trocblas_stbsv_strided_batched *rocblas_stbsv_strided_batched;
trocblas_dtbsv_strided_batched *rocblas_dtbsv_strided_batched;
trocblas_ctbsv_strided_batched *rocblas_ctbsv_strided_batched;
trocblas_ztbsv_strided_batched *rocblas_ztbsv_strided_batched;
trocblas_strsv *rocblas_strsv;
trocblas_dtrsv *rocblas_dtrsv;
trocblas_ctrsv *rocblas_ctrsv;
trocblas_ztrsv *rocblas_ztrsv;
trocblas_strsv_batched *rocblas_strsv_batched;
trocblas_dtrsv_batched *rocblas_dtrsv_batched;
trocblas_ctrsv_batched *rocblas_ctrsv_batched;
trocblas_ztrsv_batched *rocblas_ztrsv_batched;
trocblas_strsv_strided_batched *rocblas_strsv_strided_batched;
trocblas_dtrsv_strided_batched *rocblas_dtrsv_strided_batched;
trocblas_ctrsv_strided_batched *rocblas_ctrsv_strided_batched;
trocblas_ztrsv_strided_batched *rocblas_ztrsv_strided_batched;
trocblas_stpsv *rocblas_stpsv;
trocblas_dtpsv *rocblas_dtpsv;
trocblas_ctpsv *rocblas_ctpsv;
trocblas_ztpsv *rocblas_ztpsv;
trocblas_stpsv_batched *rocblas_stpsv_batched;
trocblas_dtpsv_batched *rocblas_dtpsv_batched;
trocblas_ctpsv_batched *rocblas_ctpsv_batched;
trocblas_ztpsv_batched *rocblas_ztpsv_batched;
trocblas_stpsv_strided_batched *rocblas_stpsv_strided_batched;
trocblas_dtpsv_strided_batched *rocblas_dtpsv_strided_batched;
trocblas_ctpsv_strided_batched *rocblas_ctpsv_strided_batched;
trocblas_ztpsv_strided_batched *rocblas_ztpsv_strided_batched;
trocblas_ssymv *rocblas_ssymv;
trocblas_dsymv *rocblas_dsymv;
trocblas_csymv *rocblas_csymv;
trocblas_zsymv *rocblas_zsymv;
trocblas_ssymv_batched *rocblas_ssymv_batched;
trocblas_dsymv_batched *rocblas_dsymv_batched;
trocblas_csymv_batched *rocblas_csymv_batched;
trocblas_zsymv_batched *rocblas_zsymv_batched;
trocblas_ssymv_strided_batched *rocblas_ssymv_strided_batched;
trocblas_dsymv_strided_batched *rocblas_dsymv_strided_batched;
trocblas_csymv_strided_batched *rocblas_csymv_strided_batched;
trocblas_zsymv_strided_batched *rocblas_zsymv_strided_batched;
trocblas_sspmv *rocblas_sspmv;
trocblas_dspmv *rocblas_dspmv;
trocblas_sspmv_batched *rocblas_sspmv_batched;
trocblas_dspmv_batched *rocblas_dspmv_batched;
trocblas_sspmv_strided_batched *rocblas_sspmv_strided_batched;
trocblas_dspmv_strided_batched *rocblas_dspmv_strided_batched;
trocblas_ssbmv *rocblas_ssbmv;
trocblas_dsbmv *rocblas_dsbmv;
trocblas_dsbmv_batched *rocblas_dsbmv_batched;
trocblas_ssbmv_batched *rocblas_ssbmv_batched;
trocblas_ssbmv_strided_batched *rocblas_ssbmv_strided_batched;
trocblas_dsbmv_strided_batched *rocblas_dsbmv_strided_batched;
trocblas_sger *rocblas_sger;
trocblas_dger *rocblas_dger;
trocblas_cgeru *rocblas_cgeru;
trocblas_zgeru *rocblas_zgeru;
trocblas_cgerc *rocblas_cgerc;
trocblas_zgerc *rocblas_zgerc;
trocblas_sger_batched *rocblas_sger_batched;
trocblas_dger_batched *rocblas_dger_batched;
trocblas_cgeru_batched *rocblas_cgeru_batched;
trocblas_zgeru_batched *rocblas_zgeru_batched;
trocblas_cgerc_batched *rocblas_cgerc_batched;
trocblas_zgerc_batched *rocblas_zgerc_batched;
trocblas_sger_strided_batched *rocblas_sger_strided_batched;
trocblas_dger_strided_batched *rocblas_dger_strided_batched;
trocblas_cgeru_strided_batched *rocblas_cgeru_strided_batched;
trocblas_zgeru_strided_batched *rocblas_zgeru_strided_batched;
trocblas_cgerc_strided_batched *rocblas_cgerc_strided_batched;
trocblas_zgerc_strided_batched *rocblas_zgerc_strided_batched;
trocblas_sspr *rocblas_sspr;
trocblas_dspr *rocblas_dspr;
trocblas_cspr *rocblas_cspr;
trocblas_zspr *rocblas_zspr;
trocblas_sspr_batched *rocblas_sspr_batched;
trocblas_dspr_batched *rocblas_dspr_batched;
trocblas_cspr_batched *rocblas_cspr_batched;
trocblas_zspr_batched *rocblas_zspr_batched;
trocblas_sspr_strided_batched *rocblas_sspr_strided_batched;
trocblas_dspr_strided_batched *rocblas_dspr_strided_batched;
trocblas_cspr_strided_batched *rocblas_cspr_strided_batched;
trocblas_zspr_strided_batched *rocblas_zspr_strided_batched;
trocblas_sspr2 *rocblas_sspr2;
trocblas_dspr2 *rocblas_dspr2;
trocblas_sspr2_batched *rocblas_sspr2_batched;
trocblas_dspr2_batched *rocblas_dspr2_batched;
trocblas_sspr2_strided_batched *rocblas_sspr2_strided_batched;
trocblas_dspr2_strided_batched *rocblas_dspr2_strided_batched;
trocblas_ssyr *rocblas_ssyr;
trocblas_dsyr *rocblas_dsyr;
trocblas_csyr *rocblas_csyr;
trocblas_zsyr *rocblas_zsyr;
trocblas_ssyr_batched *rocblas_ssyr_batched;
trocblas_dsyr_batched *rocblas_dsyr_batched;
trocblas_csyr_batched *rocblas_csyr_batched;
trocblas_zsyr_batched *rocblas_zsyr_batched;
trocblas_ssyr_strided_batched *rocblas_ssyr_strided_batched;
trocblas_dsyr_strided_batched *rocblas_dsyr_strided_batched;
trocblas_csyr_strided_batched *rocblas_csyr_strided_batched;
trocblas_zsyr_strided_batched *rocblas_zsyr_strided_batched;
trocblas_ssyr2 *rocblas_ssyr2;
trocblas_dsyr2 *rocblas_dsyr2;
trocblas_csyr2 *rocblas_csyr2;
trocblas_zsyr2 *rocblas_zsyr2;
trocblas_ssyr2_batched *rocblas_ssyr2_batched;
trocblas_dsyr2_batched *rocblas_dsyr2_batched;
trocblas_csyr2_batched *rocblas_csyr2_batched;
trocblas_zsyr2_batched *rocblas_zsyr2_batched;
trocblas_ssyr2_strided_batched *rocblas_ssyr2_strided_batched;
trocblas_dsyr2_strided_batched *rocblas_dsyr2_strided_batched;
trocblas_csyr2_strided_batched *rocblas_csyr2_strided_batched;
trocblas_zsyr2_strided_batched *rocblas_zsyr2_strided_batched;
trocblas_chemm *rocblas_chemm;
trocblas_zhemm *rocblas_zhemm;
trocblas_chemm_batched *rocblas_chemm_batched;
trocblas_zhemm_batched *rocblas_zhemm_batched;
trocblas_chemm_strided_batched *rocblas_chemm_strided_batched;
trocblas_zhemm_strided_batched *rocblas_zhemm_strided_batched;
trocblas_cherk *rocblas_cherk;
trocblas_zherk *rocblas_zherk;
trocblas_cherk_batched *rocblas_cherk_batched;
trocblas_zherk_batched *rocblas_zherk_batched;
trocblas_cherk_strided_batched *rocblas_cherk_strided_batched;
trocblas_zherk_strided_batched *rocblas_zherk_strided_batched;
trocblas_cher2k *rocblas_cher2k;
trocblas_zher2k *rocblas_zher2k;
trocblas_cher2k_batched *rocblas_cher2k_batched;
trocblas_zher2k_batched *rocblas_zher2k_batched;
trocblas_cher2k_strided_batched *rocblas_cher2k_strided_batched;
trocblas_zher2k_strided_batched *rocblas_zher2k_strided_batched;
trocblas_cherkx *rocblas_cherkx;
trocblas_zherkx *rocblas_zherkx;
trocblas_cherkx_batched *rocblas_cherkx_batched;
trocblas_zherkx_batched *rocblas_zherkx_batched;
trocblas_cherkx_strided_batched *rocblas_cherkx_strided_batched;
trocblas_zherkx_strided_batched *rocblas_zherkx_strided_batched;
trocblas_ssymm *rocblas_ssymm;
trocblas_dsymm *rocblas_dsymm;
trocblas_csymm *rocblas_csymm;
trocblas_zsymm *rocblas_zsymm;
trocblas_ssymm_batched *rocblas_ssymm_batched;
trocblas_dsymm_batched *rocblas_dsymm_batched;
trocblas_csymm_batched *rocblas_csymm_batched;
trocblas_zsymm_batched *rocblas_zsymm_batched;
trocblas_ssymm_strided_batched *rocblas_ssymm_strided_batched;
trocblas_dsymm_strided_batched *rocblas_dsymm_strided_batched;
trocblas_csymm_strided_batched *rocblas_csymm_strided_batched;
trocblas_zsymm_strided_batched *rocblas_zsymm_strided_batched;
trocblas_ssyrk *rocblas_ssyrk;
trocblas_dsyrk *rocblas_dsyrk;
trocblas_csyrk *rocblas_csyrk;
trocblas_zsyrk *rocblas_zsyrk;
trocblas_ssyrk_batched *rocblas_ssyrk_batched;
trocblas_dsyrk_batched *rocblas_dsyrk_batched;
trocblas_csyrk_batched *rocblas_csyrk_batched;
trocblas_zsyrk_batched *rocblas_zsyrk_batched;
trocblas_ssyrk_strided_batched *rocblas_ssyrk_strided_batched;
trocblas_dsyrk_strided_batched *rocblas_dsyrk_strided_batched;
trocblas_csyrk_strided_batched *rocblas_csyrk_strided_batched;
trocblas_zsyrk_strided_batched *rocblas_zsyrk_strided_batched;
trocblas_ssyr2k *rocblas_ssyr2k;
trocblas_dsyr2k *rocblas_dsyr2k;
trocblas_csyr2k *rocblas_csyr2k;
trocblas_zsyr2k *rocblas_zsyr2k;
trocblas_ssyr2k_batched *rocblas_ssyr2k_batched;
trocblas_dsyr2k_batched *rocblas_dsyr2k_batched;
trocblas_csyr2k_batched *rocblas_csyr2k_batched;
trocblas_zsyr2k_batched *rocblas_zsyr2k_batched;
trocblas_ssyr2k_strided_batched *rocblas_ssyr2k_strided_batched;
trocblas_dsyr2k_strided_batched *rocblas_dsyr2k_strided_batched;
trocblas_csyr2k_strided_batched *rocblas_csyr2k_strided_batched;
trocblas_zsyr2k_strided_batched *rocblas_zsyr2k_strided_batched;
trocblas_ssyrkx *rocblas_ssyrkx;
trocblas_dsyrkx *rocblas_dsyrkx;
trocblas_csyrkx *rocblas_csyrkx;
trocblas_zsyrkx *rocblas_zsyrkx;
trocblas_ssyrkx_batched *rocblas_ssyrkx_batched;
trocblas_dsyrkx_batched *rocblas_dsyrkx_batched;
trocblas_csyrkx_batched *rocblas_csyrkx_batched;
trocblas_zsyrkx_batched *rocblas_zsyrkx_batched;
trocblas_ssyrkx_strided_batched *rocblas_ssyrkx_strided_batched;
trocblas_dsyrkx_strided_batched *rocblas_dsyrkx_strided_batched;
trocblas_csyrkx_strided_batched *rocblas_csyrkx_strided_batched;
trocblas_zsyrkx_strided_batched *rocblas_zsyrkx_strided_batched;
trocblas_strmm *rocblas_strmm;
trocblas_dtrmm *rocblas_dtrmm;
trocblas_ctrmm *rocblas_ctrmm;
trocblas_ztrmm *rocblas_ztrmm;
trocblas_strmm_batched *rocblas_strmm_batched;
trocblas_dtrmm_batched *rocblas_dtrmm_batched;
trocblas_ctrmm_batched *rocblas_ctrmm_batched;
trocblas_ztrmm_batched *rocblas_ztrmm_batched;
trocblas_strmm_strided_batched *rocblas_strmm_strided_batched;
trocblas_dtrmm_strided_batched *rocblas_dtrmm_strided_batched;
trocblas_ctrmm_strided_batched *rocblas_ctrmm_strided_batched;
trocblas_ztrmm_strided_batched *rocblas_ztrmm_strided_batched;
trocblas_strmm_outofplace *rocblas_strmm_outofplace;
trocblas_dtrmm_outofplace *rocblas_dtrmm_outofplace;
trocblas_ctrmm_outofplace *rocblas_ctrmm_outofplace;
trocblas_ztrmm_outofplace *rocblas_ztrmm_outofplace;
trocblas_strmm_outofplace_batched *rocblas_strmm_outofplace_batched;
trocblas_dtrmm_outofplace_batched *rocblas_dtrmm_outofplace_batched;
trocblas_ctrmm_outofplace_batched *rocblas_ctrmm_outofplace_batched;
trocblas_ztrmm_outofplace_batched *rocblas_ztrmm_outofplace_batched;
trocblas_strmm_outofplace_strided_batched *rocblas_strmm_outofplace_strided_batched;
trocblas_dtrmm_outofplace_strided_batched *rocblas_dtrmm_outofplace_strided_batched;
trocblas_ctrmm_outofplace_strided_batched *rocblas_ctrmm_outofplace_strided_batched;
trocblas_ztrmm_outofplace_strided_batched *rocblas_ztrmm_outofplace_strided_batched;
trocblas_strtri *rocblas_strtri;
trocblas_dtrtri *rocblas_dtrtri;
trocblas_ctrtri *rocblas_ctrtri;
trocblas_ztrtri *rocblas_ztrtri;
trocblas_strtri_batched *rocblas_strtri_batched;
trocblas_dtrtri_batched *rocblas_dtrtri_batched;
trocblas_ctrtri_batched *rocblas_ctrtri_batched;
trocblas_ztrtri_batched *rocblas_ztrtri_batched;
trocblas_strtri_strided_batched *rocblas_strtri_strided_batched;
trocblas_dtrtri_strided_batched *rocblas_dtrtri_strided_batched;
trocblas_ctrtri_strided_batched *rocblas_ctrtri_strided_batched;
trocblas_ztrtri_strided_batched *rocblas_ztrtri_strided_batched;
trocblas_strsm *rocblas_strsm;
trocblas_dtrsm *rocblas_dtrsm;
trocblas_ctrsm *rocblas_ctrsm;
trocblas_ztrsm *rocblas_ztrsm;
trocblas_strsm_batched *rocblas_strsm_batched;
trocblas_dtrsm_batched *rocblas_dtrsm_batched;
trocblas_ctrsm_batched *rocblas_ctrsm_batched;
trocblas_ztrsm_batched *rocblas_ztrsm_batched;
trocblas_strsm_strided_batched *rocblas_strsm_strided_batched;
trocblas_dtrsm_strided_batched *rocblas_dtrsm_strided_batched;
trocblas_ctrsm_strided_batched *rocblas_ctrsm_strided_batched;
trocblas_ztrsm_strided_batched *rocblas_ztrsm_strided_batched;
trocblas_sgemm *rocblas_sgemm;
trocblas_dgemm *rocblas_dgemm;
trocblas_hgemm *rocblas_hgemm;
trocblas_cgemm *rocblas_cgemm;
trocblas_zgemm *rocblas_zgemm;
trocblas_sgemm_batched *rocblas_sgemm_batched;
trocblas_dgemm_batched *rocblas_dgemm_batched;
trocblas_hgemm_batched *rocblas_hgemm_batched;
trocblas_cgemm_batched *rocblas_cgemm_batched;
trocblas_zgemm_batched *rocblas_zgemm_batched;
trocblas_sgemm_strided_batched *rocblas_sgemm_strided_batched;
trocblas_dgemm_strided_batched *rocblas_dgemm_strided_batched;
trocblas_hgemm_strided_batched *rocblas_hgemm_strided_batched;
trocblas_hgemm_kernel_name *rocblas_hgemm_kernel_name;
trocblas_sgemm_kernel_name *rocblas_sgemm_kernel_name;
trocblas_dgemm_kernel_name *rocblas_dgemm_kernel_name;
trocblas_cgemm_strided_batched *rocblas_cgemm_strided_batched;
trocblas_zgemm_strided_batched *rocblas_zgemm_strided_batched;
trocblas_sdgmm *rocblas_sdgmm;
trocblas_ddgmm *rocblas_ddgmm;
trocblas_cdgmm *rocblas_cdgmm;
trocblas_zdgmm *rocblas_zdgmm;
trocblas_sdgmm_batched *rocblas_sdgmm_batched;
trocblas_ddgmm_batched *rocblas_ddgmm_batched;
trocblas_cdgmm_batched *rocblas_cdgmm_batched;
trocblas_zdgmm_batched *rocblas_zdgmm_batched;
trocblas_sdgmm_strided_batched *rocblas_sdgmm_strided_batched;
trocblas_ddgmm_strided_batched *rocblas_ddgmm_strided_batched;
trocblas_cdgmm_strided_batched *rocblas_cdgmm_strided_batched;
trocblas_zdgmm_strided_batched *rocblas_zdgmm_strided_batched;
trocblas_sgeam *rocblas_sgeam;
trocblas_dgeam *rocblas_dgeam;
trocblas_cgeam *rocblas_cgeam;
trocblas_zgeam *rocblas_zgeam;
trocblas_sgeam_batched *rocblas_sgeam_batched;
trocblas_dgeam_batched *rocblas_dgeam_batched;
trocblas_cgeam_batched *rocblas_cgeam_batched;
trocblas_zgeam_batched *rocblas_zgeam_batched;
trocblas_sgeam_strided_batched *rocblas_sgeam_strided_batched;
trocblas_dgeam_strided_batched *rocblas_dgeam_strided_batched;
trocblas_cgeam_strided_batched *rocblas_cgeam_strided_batched;
trocblas_zgeam_strided_batched *rocblas_zgeam_strided_batched;
trocblas_gemm_ex *rocblas_gemm_ex;
trocblas_gemm_batched_ex *rocblas_gemm_batched_ex;
trocblas_gemm_strided_batched_ex *rocblas_gemm_strided_batched_ex;
trocblas_gemm_ext2 *rocblas_gemm_ext2;
trocblas_geam_ex *rocblas_geam_ex;
trocblas_trsm_ex *rocblas_trsm_ex;
trocblas_trsm_batched_ex *rocblas_trsm_batched_ex;
trocblas_trsm_strided_batched_ex *rocblas_trsm_strided_batched_ex;
trocblas_axpy_ex *rocblas_axpy_ex;
trocblas_axpy_batched_ex *rocblas_axpy_batched_ex;
trocblas_axpy_strided_batched_ex *rocblas_axpy_strided_batched_ex;
trocblas_dot_ex *rocblas_dot_ex;
trocblas_dotc_ex *rocblas_dotc_ex;
trocblas_dot_batched_ex *rocblas_dot_batched_ex;
trocblas_dotc_batched_ex *rocblas_dotc_batched_ex;
trocblas_dot_strided_batched_ex *rocblas_dot_strided_batched_ex;
trocblas_dotc_strided_batched_ex *rocblas_dotc_strided_batched_ex;
trocblas_nrm2_ex *rocblas_nrm2_ex;
trocblas_nrm2_batched_ex *rocblas_nrm2_batched_ex;
trocblas_nrm2_strided_batched_ex *rocblas_nrm2_strided_batched_ex;
trocblas_rot_ex *rocblas_rot_ex;
trocblas_rot_batched_ex *rocblas_rot_batched_ex;
trocblas_rot_strided_batched_ex *rocblas_rot_strided_batched_ex;
trocblas_scal_ex *rocblas_scal_ex;
trocblas_scal_batched_ex *rocblas_scal_batched_ex;
trocblas_scal_strided_batched_ex *rocblas_scal_strided_batched_ex;
trocblas_status_to_string *rocblas_status_to_string;
trocblas_initialize *rocblas_initialize;
trocblas_get_version_string *rocblas_get_version_string;
trocblas_get_version_string_size *rocblas_get_version_string_size;
trocblas_start_device_memory_size_query *rocblas_start_device_memory_size_query;
trocblas_stop_device_memory_size_query *rocblas_stop_device_memory_size_query;
trocblas_is_device_memory_size_query *rocblas_is_device_memory_size_query;
trocblas_set_optimal_device_memory_size_impl *rocblas_set_optimal_device_memory_size_impl;
trocblas_device_malloc_alloc *rocblas_device_malloc_alloc;
trocblas_device_malloc_success *rocblas_device_malloc_success;
trocblas_device_malloc_ptr *rocblas_device_malloc_ptr;
trocblas_device_malloc_get *rocblas_device_malloc_get;
trocblas_device_malloc_free *rocblas_device_malloc_free;
trocblas_device_malloc_set_default_memory_size *rocblas_device_malloc_set_default_memory_size;
trocblas_get_device_memory_size *rocblas_get_device_memory_size;
trocblas_set_device_memory_size *rocblas_set_device_memory_size;
trocblas_set_workspace *rocblas_set_workspace;
trocblas_is_managing_device_memory *rocblas_is_managing_device_memory;
trocblas_is_user_managing_device_memory *rocblas_is_user_managing_device_memory;
trocblas_abort *rocblas_abort;

int hipewInitROCBLAS(const char **extra_dll_search_paths) {

#ifdef _WIN32
  const char *paths[] = {   "rocblas.dll",
NULL};
#else /* linux */
  const char *paths[] = {   "librocblas.so",
   "/opt/rocm/lib/librocblas.so",
NULL};
#endif


  static int initialized = 0;
  static int result = 0;
  int error;

  if (initialized) {
    return result;
  }

  initialized = 1;
  error = atexit(hipewExitROCBLAS);

  if (error) {
    result = -2;
    return result;
  }
  rocblas_lib = dynamic_library_open_find(paths);
  if (rocblas_lib == NULL) { 
    if (extra_dll_search_paths) { 
      rocblas_lib = dynamic_library_open_find(extra_dll_search_paths);
    }
  }
  if (rocblas_lib == NULL) { result = -1; return result; }

  ROCBLAS_LIBRARY_FIND(rocblas_create_handle)
  ROCBLAS_LIBRARY_FIND(rocblas_destroy_handle)
  ROCBLAS_LIBRARY_FIND(rocblas_set_stream)
  ROCBLAS_LIBRARY_FIND(rocblas_get_stream)
  ROCBLAS_LIBRARY_FIND(rocblas_set_pointer_mode)
  ROCBLAS_LIBRARY_FIND(rocblas_get_pointer_mode)
  ROCBLAS_LIBRARY_FIND(rocblas_set_int8_type_for_hipblas)
  ROCBLAS_LIBRARY_FIND(rocblas_get_int8_type_for_hipblas)
  ROCBLAS_LIBRARY_FIND(rocblas_set_atomics_mode)
  ROCBLAS_LIBRARY_FIND(rocblas_get_atomics_mode)
  ROCBLAS_LIBRARY_FIND(rocblas_query_int8_layout_flag)
  ROCBLAS_LIBRARY_FIND(rocblas_pointer_to_mode)
  ROCBLAS_LIBRARY_FIND(rocblas_set_vector)
  ROCBLAS_LIBRARY_FIND(rocblas_get_vector)
  ROCBLAS_LIBRARY_FIND(rocblas_set_matrix)
  ROCBLAS_LIBRARY_FIND(rocblas_get_matrix)
  ROCBLAS_LIBRARY_FIND(rocblas_set_vector_async)
  ROCBLAS_LIBRARY_FIND(rocblas_get_vector_async)
  ROCBLAS_LIBRARY_FIND(rocblas_set_matrix_async)
  ROCBLAS_LIBRARY_FIND(rocblas_get_matrix_async)
  ROCBLAS_LIBRARY_FIND(rocblas_set_start_stop_events)
  ROCBLAS_LIBRARY_FIND(rocblas_set_solution_fitness_query)
  ROCBLAS_LIBRARY_FIND(rocblas_set_performance_metric)
  ROCBLAS_LIBRARY_FIND(rocblas_get_performance_metric)
  ROCBLAS_LIBRARY_FIND(rocblas_sscal)
  ROCBLAS_LIBRARY_FIND(rocblas_dscal)
  ROCBLAS_LIBRARY_FIND(rocblas_cscal)
  ROCBLAS_LIBRARY_FIND(rocblas_zscal)
  ROCBLAS_LIBRARY_FIND(rocblas_csscal)
  ROCBLAS_LIBRARY_FIND(rocblas_zdscal)
  ROCBLAS_LIBRARY_FIND(rocblas_sscal_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dscal_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cscal_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zscal_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csscal_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdscal_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sscal_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dscal_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cscal_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zscal_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csscal_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdscal_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_scopy)
  ROCBLAS_LIBRARY_FIND(rocblas_dcopy)
  ROCBLAS_LIBRARY_FIND(rocblas_ccopy)
  ROCBLAS_LIBRARY_FIND(rocblas_zcopy)
  ROCBLAS_LIBRARY_FIND(rocblas_scopy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dcopy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ccopy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zcopy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_scopy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dcopy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ccopy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zcopy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sdot)
  ROCBLAS_LIBRARY_FIND(rocblas_ddot)
  ROCBLAS_LIBRARY_FIND(rocblas_hdot)
  ROCBLAS_LIBRARY_FIND(rocblas_bfdot)
  ROCBLAS_LIBRARY_FIND(rocblas_cdotu)
  ROCBLAS_LIBRARY_FIND(rocblas_zdotu)
  ROCBLAS_LIBRARY_FIND(rocblas_cdotc)
  ROCBLAS_LIBRARY_FIND(rocblas_zdotc)
  ROCBLAS_LIBRARY_FIND(rocblas_sdot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ddot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_hdot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_bfdot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cdotu_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdotu_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cdotc_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdotc_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sdot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ddot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_hdot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_bfdot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cdotu_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdotu_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cdotc_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdotc_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sswap)
  ROCBLAS_LIBRARY_FIND(rocblas_dswap)
  ROCBLAS_LIBRARY_FIND(rocblas_cswap)
  ROCBLAS_LIBRARY_FIND(rocblas_zswap)
  ROCBLAS_LIBRARY_FIND(rocblas_sswap_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dswap_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cswap_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zswap_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sswap_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dswap_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cswap_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zswap_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_saxpy)
  ROCBLAS_LIBRARY_FIND(rocblas_daxpy)
  ROCBLAS_LIBRARY_FIND(rocblas_haxpy)
  ROCBLAS_LIBRARY_FIND(rocblas_caxpy)
  ROCBLAS_LIBRARY_FIND(rocblas_zaxpy)
  ROCBLAS_LIBRARY_FIND(rocblas_haxpy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_saxpy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_daxpy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_caxpy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zaxpy_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_haxpy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_saxpy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_daxpy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_caxpy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zaxpy_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sasum)
  ROCBLAS_LIBRARY_FIND(rocblas_dasum)
  ROCBLAS_LIBRARY_FIND(rocblas_scasum)
  ROCBLAS_LIBRARY_FIND(rocblas_dzasum)
  ROCBLAS_LIBRARY_FIND(rocblas_sasum_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dasum_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_scasum_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dzasum_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sasum_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dasum_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_scasum_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dzasum_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_snrm2)
  ROCBLAS_LIBRARY_FIND(rocblas_dnrm2)
  ROCBLAS_LIBRARY_FIND(rocblas_scnrm2)
  ROCBLAS_LIBRARY_FIND(rocblas_dznrm2)
  ROCBLAS_LIBRARY_FIND(rocblas_snrm2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dnrm2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_scnrm2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dznrm2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_snrm2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dnrm2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_scnrm2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dznrm2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_isamax)
  ROCBLAS_LIBRARY_FIND(rocblas_idamax)
  ROCBLAS_LIBRARY_FIND(rocblas_icamax)
  ROCBLAS_LIBRARY_FIND(rocblas_izamax)
  ROCBLAS_LIBRARY_FIND(rocblas_isamax_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_idamax_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_icamax_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_izamax_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_isamax_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_idamax_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_icamax_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_izamax_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_isamin)
  ROCBLAS_LIBRARY_FIND(rocblas_idamin)
  ROCBLAS_LIBRARY_FIND(rocblas_icamin)
  ROCBLAS_LIBRARY_FIND(rocblas_izamin)
  ROCBLAS_LIBRARY_FIND(rocblas_isamin_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_idamin_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_icamin_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_izamin_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_isamin_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_idamin_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_icamin_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_izamin_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_srot)
  ROCBLAS_LIBRARY_FIND(rocblas_drot)
  ROCBLAS_LIBRARY_FIND(rocblas_crot)
  ROCBLAS_LIBRARY_FIND(rocblas_csrot)
  ROCBLAS_LIBRARY_FIND(rocblas_zrot)
  ROCBLAS_LIBRARY_FIND(rocblas_zdrot)
  ROCBLAS_LIBRARY_FIND(rocblas_srot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_drot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_crot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csrot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zrot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdrot_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_srot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_drot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_crot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csrot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zrot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdrot_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_srotg)
  ROCBLAS_LIBRARY_FIND(rocblas_drotg)
  ROCBLAS_LIBRARY_FIND(rocblas_crotg)
  ROCBLAS_LIBRARY_FIND(rocblas_zrotg)
  ROCBLAS_LIBRARY_FIND(rocblas_srotg_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_drotg_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_crotg_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zrotg_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_srotg_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_drotg_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_crotg_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zrotg_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_srotm)
  ROCBLAS_LIBRARY_FIND(rocblas_drotm)
  ROCBLAS_LIBRARY_FIND(rocblas_srotm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_drotm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_srotm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_drotm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_srotmg)
  ROCBLAS_LIBRARY_FIND(rocblas_drotmg)
  ROCBLAS_LIBRARY_FIND(rocblas_srotmg_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_drotmg_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_srotmg_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_drotmg_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sgbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_dgbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_cgbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_zgbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_sgbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dgbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sgbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dgbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sgemv)
  ROCBLAS_LIBRARY_FIND(rocblas_dgemv)
  ROCBLAS_LIBRARY_FIND(rocblas_cgemv)
  ROCBLAS_LIBRARY_FIND(rocblas_zgemv)
  ROCBLAS_LIBRARY_FIND(rocblas_sgemv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dgemv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgemv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgemv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sgemv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dgemv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgemv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgemv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_zhbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_chbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chemv)
  ROCBLAS_LIBRARY_FIND(rocblas_zhemv)
  ROCBLAS_LIBRARY_FIND(rocblas_chemv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhemv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chemv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhemv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cher)
  ROCBLAS_LIBRARY_FIND(rocblas_zher)
  ROCBLAS_LIBRARY_FIND(rocblas_cher_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zher_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cher_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zher_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cher2)
  ROCBLAS_LIBRARY_FIND(rocblas_zher2)
  ROCBLAS_LIBRARY_FIND(rocblas_cher2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zher2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cher2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zher2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chpmv)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpmv)
  ROCBLAS_LIBRARY_FIND(rocblas_chpmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chpmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chpr)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpr)
  ROCBLAS_LIBRARY_FIND(rocblas_chpr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chpr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chpr2)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpr2)
  ROCBLAS_LIBRARY_FIND(rocblas_chpr2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpr2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chpr2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhpr2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strmv)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmv)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmv)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmv)
  ROCBLAS_LIBRARY_FIND(rocblas_strmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_stpmv)
  ROCBLAS_LIBRARY_FIND(rocblas_dtpmv)
  ROCBLAS_LIBRARY_FIND(rocblas_ctpmv)
  ROCBLAS_LIBRARY_FIND(rocblas_ztpmv)
  ROCBLAS_LIBRARY_FIND(rocblas_stpmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtpmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctpmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztpmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_stpmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtpmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctpmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztpmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_stbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_dtbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_ctbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_ztbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_stbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_stbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_stbsv)
  ROCBLAS_LIBRARY_FIND(rocblas_dtbsv)
  ROCBLAS_LIBRARY_FIND(rocblas_ctbsv)
  ROCBLAS_LIBRARY_FIND(rocblas_ztbsv)
  ROCBLAS_LIBRARY_FIND(rocblas_stbsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtbsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctbsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztbsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_stbsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtbsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctbsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztbsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strsv)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrsv)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrsv)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrsv)
  ROCBLAS_LIBRARY_FIND(rocblas_strsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_stpsv)
  ROCBLAS_LIBRARY_FIND(rocblas_dtpsv)
  ROCBLAS_LIBRARY_FIND(rocblas_ctpsv)
  ROCBLAS_LIBRARY_FIND(rocblas_ztpsv)
  ROCBLAS_LIBRARY_FIND(rocblas_stpsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtpsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctpsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztpsv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_stpsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtpsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctpsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztpsv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssymv)
  ROCBLAS_LIBRARY_FIND(rocblas_dsymv)
  ROCBLAS_LIBRARY_FIND(rocblas_csymv)
  ROCBLAS_LIBRARY_FIND(rocblas_zsymv)
  ROCBLAS_LIBRARY_FIND(rocblas_ssymv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsymv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csymv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsymv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssymv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsymv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csymv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsymv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sspmv)
  ROCBLAS_LIBRARY_FIND(rocblas_dspmv)
  ROCBLAS_LIBRARY_FIND(rocblas_sspmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dspmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sspmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dspmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_dsbmv)
  ROCBLAS_LIBRARY_FIND(rocblas_dsbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssbmv_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsbmv_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sger)
  ROCBLAS_LIBRARY_FIND(rocblas_dger)
  ROCBLAS_LIBRARY_FIND(rocblas_cgeru)
  ROCBLAS_LIBRARY_FIND(rocblas_zgeru)
  ROCBLAS_LIBRARY_FIND(rocblas_cgerc)
  ROCBLAS_LIBRARY_FIND(rocblas_zgerc)
  ROCBLAS_LIBRARY_FIND(rocblas_sger_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dger_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgeru_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgeru_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgerc_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgerc_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sger_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dger_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgeru_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgeru_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgerc_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgerc_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sspr)
  ROCBLAS_LIBRARY_FIND(rocblas_dspr)
  ROCBLAS_LIBRARY_FIND(rocblas_cspr)
  ROCBLAS_LIBRARY_FIND(rocblas_zspr)
  ROCBLAS_LIBRARY_FIND(rocblas_sspr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dspr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cspr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zspr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sspr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dspr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cspr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zspr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sspr2)
  ROCBLAS_LIBRARY_FIND(rocblas_dspr2)
  ROCBLAS_LIBRARY_FIND(rocblas_sspr2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dspr2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sspr2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dspr2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr2)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr2)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr2)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr2)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr2_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr2_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chemm)
  ROCBLAS_LIBRARY_FIND(rocblas_zhemm)
  ROCBLAS_LIBRARY_FIND(rocblas_chemm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhemm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_chemm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zhemm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cherk)
  ROCBLAS_LIBRARY_FIND(rocblas_zherk)
  ROCBLAS_LIBRARY_FIND(rocblas_cherk_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zherk_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cherk_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zherk_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cher2k)
  ROCBLAS_LIBRARY_FIND(rocblas_zher2k)
  ROCBLAS_LIBRARY_FIND(rocblas_cher2k_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zher2k_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cher2k_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zher2k_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cherkx)
  ROCBLAS_LIBRARY_FIND(rocblas_zherkx)
  ROCBLAS_LIBRARY_FIND(rocblas_cherkx_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zherkx_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cherkx_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zherkx_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssymm)
  ROCBLAS_LIBRARY_FIND(rocblas_dsymm)
  ROCBLAS_LIBRARY_FIND(rocblas_csymm)
  ROCBLAS_LIBRARY_FIND(rocblas_zsymm)
  ROCBLAS_LIBRARY_FIND(rocblas_ssymm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsymm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csymm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsymm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssymm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsymm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csymm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsymm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyrk)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyrk)
  ROCBLAS_LIBRARY_FIND(rocblas_csyrk)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyrk)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyrk_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyrk_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyrk_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyrk_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyrk_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyrk_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyrk_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyrk_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr2k)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr2k)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr2k)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr2k)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr2k_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr2k_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr2k_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr2k_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyr2k_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyr2k_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyr2k_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyr2k_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyrkx)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyrkx)
  ROCBLAS_LIBRARY_FIND(rocblas_csyrkx)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyrkx)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyrkx_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyrkx_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyrkx_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyrkx_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ssyrkx_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dsyrkx_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_csyrkx_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zsyrkx_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strmm)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmm)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmm)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmm)
  ROCBLAS_LIBRARY_FIND(rocblas_strmm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strmm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strmm_outofplace)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmm_outofplace)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmm_outofplace)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmm_outofplace)
  ROCBLAS_LIBRARY_FIND(rocblas_strmm_outofplace_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmm_outofplace_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmm_outofplace_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmm_outofplace_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strmm_outofplace_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrmm_outofplace_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrmm_outofplace_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrmm_outofplace_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strtri)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrtri)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrtri)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrtri)
  ROCBLAS_LIBRARY_FIND(rocblas_strtri_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrtri_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrtri_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrtri_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strtri_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrtri_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrtri_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrtri_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strsm)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrsm)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrsm)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrsm)
  ROCBLAS_LIBRARY_FIND(rocblas_strsm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrsm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrsm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrsm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_strsm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dtrsm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ctrsm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ztrsm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sgemm)
  ROCBLAS_LIBRARY_FIND(rocblas_dgemm)
  ROCBLAS_LIBRARY_FIND(rocblas_hgemm)
  ROCBLAS_LIBRARY_FIND(rocblas_cgemm)
  ROCBLAS_LIBRARY_FIND(rocblas_zgemm)
  ROCBLAS_LIBRARY_FIND(rocblas_sgemm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dgemm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_hgemm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgemm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgemm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sgemm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dgemm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_hgemm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_hgemm_kernel_name)
  ROCBLAS_LIBRARY_FIND(rocblas_sgemm_kernel_name)
  ROCBLAS_LIBRARY_FIND(rocblas_dgemm_kernel_name)
  ROCBLAS_LIBRARY_FIND(rocblas_cgemm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgemm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sdgmm)
  ROCBLAS_LIBRARY_FIND(rocblas_ddgmm)
  ROCBLAS_LIBRARY_FIND(rocblas_cdgmm)
  ROCBLAS_LIBRARY_FIND(rocblas_zdgmm)
  ROCBLAS_LIBRARY_FIND(rocblas_sdgmm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ddgmm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cdgmm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdgmm_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sdgmm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_ddgmm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cdgmm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zdgmm_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sgeam)
  ROCBLAS_LIBRARY_FIND(rocblas_dgeam)
  ROCBLAS_LIBRARY_FIND(rocblas_cgeam)
  ROCBLAS_LIBRARY_FIND(rocblas_zgeam)
  ROCBLAS_LIBRARY_FIND(rocblas_sgeam_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dgeam_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgeam_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgeam_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_sgeam_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_dgeam_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_cgeam_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_zgeam_strided_batched)
  ROCBLAS_LIBRARY_FIND(rocblas_gemm_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_gemm_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_gemm_strided_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_gemm_ext2)
  ROCBLAS_LIBRARY_FIND(rocblas_geam_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_trsm_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_trsm_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_trsm_strided_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_axpy_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_axpy_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_axpy_strided_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_dot_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_dotc_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_dot_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_dotc_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_dot_strided_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_dotc_strided_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_nrm2_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_nrm2_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_nrm2_strided_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_rot_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_rot_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_rot_strided_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_scal_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_scal_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_scal_strided_batched_ex)
  ROCBLAS_LIBRARY_FIND(rocblas_status_to_string)
  ROCBLAS_LIBRARY_FIND(rocblas_initialize)
  ROCBLAS_LIBRARY_FIND(rocblas_get_version_string)
  ROCBLAS_LIBRARY_FIND(rocblas_get_version_string_size)
  ROCBLAS_LIBRARY_FIND(rocblas_start_device_memory_size_query)
  ROCBLAS_LIBRARY_FIND(rocblas_stop_device_memory_size_query)
  ROCBLAS_LIBRARY_FIND(rocblas_is_device_memory_size_query)
  ROCBLAS_LIBRARY_FIND(rocblas_set_optimal_device_memory_size_impl)
  ROCBLAS_LIBRARY_FIND(rocblas_device_malloc_alloc)
  ROCBLAS_LIBRARY_FIND(rocblas_device_malloc_success)
  ROCBLAS_LIBRARY_FIND(rocblas_device_malloc_ptr)
  ROCBLAS_LIBRARY_FIND(rocblas_device_malloc_get)
  ROCBLAS_LIBRARY_FIND(rocblas_device_malloc_free)
  ROCBLAS_LIBRARY_FIND(rocblas_device_malloc_set_default_memory_size)
  ROCBLAS_LIBRARY_FIND(rocblas_get_device_memory_size)
  ROCBLAS_LIBRARY_FIND(rocblas_set_device_memory_size)
  ROCBLAS_LIBRARY_FIND(rocblas_set_workspace)
  ROCBLAS_LIBRARY_FIND(rocblas_is_managing_device_memory)
  ROCBLAS_LIBRARY_FIND(rocblas_is_user_managing_device_memory)
  ROCBLAS_LIBRARY_FIND(rocblas_abort)
  result = 0; // success
  return result;
}
