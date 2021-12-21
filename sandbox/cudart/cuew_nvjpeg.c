

#ifdef _MSC_VER
#  if _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#  define popen _popen
#  define pclose _pclose
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include "cuew.h"
#include "cuew_nvjpeg.h"
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

/*#define NVJPEG_LIBRARY_FIND_CHECKED(name) CUEW_IMPL_LIBRARY_FIND_CHECKED(nvjpeg_lib, name)*/
#define NVJPEG_LIBRARY_FIND(name) CUEW_IMPL_LIBRARY_FIND(nvjpeg_lib, name)
static DynamicLibrary nvjpeg_lib;

static void cuewExitNVJPEG(void) {
  if (nvjpeg_lib != NULL) {
    /* ignore errors */
    dynamic_library_close(nvjpeg_lib);
    nvjpeg_lib = NULL;
  }
}

tnvjpegGetProperty *nvjpegGetProperty;
tnvjpegGetCudartProperty *nvjpegGetCudartProperty;
tnvjpegCreate *nvjpegCreate;
tnvjpegCreateSimple *nvjpegCreateSimple;
tnvjpegCreateEx *nvjpegCreateEx;
tnvjpegDestroy *nvjpegDestroy;
tnvjpegSetDeviceMemoryPadding *nvjpegSetDeviceMemoryPadding;
tnvjpegGetDeviceMemoryPadding *nvjpegGetDeviceMemoryPadding;
tnvjpegSetPinnedMemoryPadding *nvjpegSetPinnedMemoryPadding;
tnvjpegGetPinnedMemoryPadding *nvjpegGetPinnedMemoryPadding;
tnvjpegJpegStateCreate *nvjpegJpegStateCreate;
tnvjpegJpegStateDestroy *nvjpegJpegStateDestroy;
tnvjpegGetImageInfo *nvjpegGetImageInfo;
tnvjpegDecode *nvjpegDecode;
tnvjpegDecodeBatchedInitialize *nvjpegDecodeBatchedInitialize;
tnvjpegDecodeBatched *nvjpegDecodeBatched;
tnvjpegDecodeBatchedPreAllocate *nvjpegDecodeBatchedPreAllocate;
tnvjpegEncoderStateCreate *nvjpegEncoderStateCreate;
tnvjpegEncoderStateDestroy *nvjpegEncoderStateDestroy;
tnvjpegEncoderParamsCreate *nvjpegEncoderParamsCreate;
tnvjpegEncoderParamsDestroy *nvjpegEncoderParamsDestroy;
tnvjpegEncoderParamsSetQuality *nvjpegEncoderParamsSetQuality;
tnvjpegEncoderParamsSetEncoding *nvjpegEncoderParamsSetEncoding;
tnvjpegEncoderParamsSetOptimizedHuffman *nvjpegEncoderParamsSetOptimizedHuffman;
tnvjpegEncoderParamsSetSamplingFactors *nvjpegEncoderParamsSetSamplingFactors;
tnvjpegEncodeGetBufferSize *nvjpegEncodeGetBufferSize;
tnvjpegEncodeYUV *nvjpegEncodeYUV;
tnvjpegEncodeImage *nvjpegEncodeImage;
tnvjpegEncodeRetrieveBitstreamDevice *nvjpegEncodeRetrieveBitstreamDevice;
tnvjpegEncodeRetrieveBitstream *nvjpegEncodeRetrieveBitstream;
tnvjpegBufferPinnedCreate *nvjpegBufferPinnedCreate;
tnvjpegBufferPinnedDestroy *nvjpegBufferPinnedDestroy;
tnvjpegBufferDeviceCreate *nvjpegBufferDeviceCreate;
tnvjpegBufferDeviceDestroy *nvjpegBufferDeviceDestroy;
tnvjpegBufferPinnedRetrieve *nvjpegBufferPinnedRetrieve;
tnvjpegBufferDeviceRetrieve *nvjpegBufferDeviceRetrieve;
tnvjpegStateAttachPinnedBuffer *nvjpegStateAttachPinnedBuffer;
tnvjpegStateAttachDeviceBuffer *nvjpegStateAttachDeviceBuffer;
tnvjpegJpegStreamCreate *nvjpegJpegStreamCreate;
tnvjpegJpegStreamDestroy *nvjpegJpegStreamDestroy;
tnvjpegJpegStreamParse *nvjpegJpegStreamParse;
tnvjpegJpegStreamParseHeader *nvjpegJpegStreamParseHeader;
tnvjpegJpegStreamGetJpegEncoding *nvjpegJpegStreamGetJpegEncoding;
tnvjpegJpegStreamGetFrameDimensions *nvjpegJpegStreamGetFrameDimensions;
tnvjpegJpegStreamGetComponentsNum *nvjpegJpegStreamGetComponentsNum;
tnvjpegJpegStreamGetComponentDimensions *nvjpegJpegStreamGetComponentDimensions;
tnvjpegJpegStreamGetChromaSubsampling *nvjpegJpegStreamGetChromaSubsampling;
tnvjpegDecodeParamsCreate *nvjpegDecodeParamsCreate;
tnvjpegDecodeParamsDestroy *nvjpegDecodeParamsDestroy;
tnvjpegDecodeParamsSetOutputFormat *nvjpegDecodeParamsSetOutputFormat;
tnvjpegDecodeParamsSetROI *nvjpegDecodeParamsSetROI;
tnvjpegDecodeParamsSetAllowCMYK *nvjpegDecodeParamsSetAllowCMYK;
tnvjpegDecodeParamsSetScaleFactor *nvjpegDecodeParamsSetScaleFactor;
tnvjpegDecoderCreate *nvjpegDecoderCreate;
tnvjpegDecoderDestroy *nvjpegDecoderDestroy;
tnvjpegDecoderJpegSupported *nvjpegDecoderJpegSupported;
tnvjpegDecodeBatchedSupported *nvjpegDecodeBatchedSupported;
tnvjpegDecodeBatchedSupportedEx *nvjpegDecodeBatchedSupportedEx;
tnvjpegDecoderStateCreate *nvjpegDecoderStateCreate;
tnvjpegDecodeJpeg *nvjpegDecodeJpeg;
tnvjpegDecodeJpegHost *nvjpegDecodeJpegHost;
tnvjpegDecodeJpegTransferToDevice *nvjpegDecodeJpegTransferToDevice;
tnvjpegDecodeJpegDevice *nvjpegDecodeJpegDevice;
tnvjpegDecodeBatchedEx *nvjpegDecodeBatchedEx;
tnvjpegEncoderParamsCopyMetadata *nvjpegEncoderParamsCopyMetadata;
tnvjpegEncoderParamsCopyQuantizationTables *nvjpegEncoderParamsCopyQuantizationTables;
tnvjpegEncoderParamsCopyHuffmanTables *nvjpegEncoderParamsCopyHuffmanTables;

int cuewInitNVJPEG() {

#ifdef _WIN32
  const char *paths[] = {   "nvjpeg.dll",
NULL};
#else /* linux */
  const char *paths[] = {   "libnvjpeg.so",
   "/usr/local/cuda/lib64/libnvjpeg.so",
NULL};
#endif


  static int initialized = 0;
  static int result = 0;
  int error;

  if (initialized) {
    return result;
  }

  initialized = 1;
  error = atexit(cuewExitNVJPEG);

  if (error) {
    result = -2;
    return result;
  }
  nvjpeg_lib = dynamic_library_open_find(paths);
  if (nvjpeg_lib == NULL) { result = -1; return result; }

  NVJPEG_LIBRARY_FIND(nvjpegGetProperty)
  NVJPEG_LIBRARY_FIND(nvjpegGetCudartProperty)
  NVJPEG_LIBRARY_FIND(nvjpegCreate)
  NVJPEG_LIBRARY_FIND(nvjpegCreateSimple)
  NVJPEG_LIBRARY_FIND(nvjpegCreateEx)
  NVJPEG_LIBRARY_FIND(nvjpegDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegSetDeviceMemoryPadding)
  NVJPEG_LIBRARY_FIND(nvjpegGetDeviceMemoryPadding)
  NVJPEG_LIBRARY_FIND(nvjpegSetPinnedMemoryPadding)
  NVJPEG_LIBRARY_FIND(nvjpegGetPinnedMemoryPadding)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStateCreate)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStateDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegGetImageInfo)
  NVJPEG_LIBRARY_FIND(nvjpegDecode)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeBatchedInitialize)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeBatched)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeBatchedPreAllocate)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderStateCreate)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderStateDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsCreate)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsSetQuality)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsSetEncoding)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsSetOptimizedHuffman)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsSetSamplingFactors)
  NVJPEG_LIBRARY_FIND(nvjpegEncodeGetBufferSize)
  NVJPEG_LIBRARY_FIND(nvjpegEncodeYUV)
  NVJPEG_LIBRARY_FIND(nvjpegEncodeImage)
  NVJPEG_LIBRARY_FIND(nvjpegEncodeRetrieveBitstreamDevice)
  NVJPEG_LIBRARY_FIND(nvjpegEncodeRetrieveBitstream)
  NVJPEG_LIBRARY_FIND(nvjpegBufferPinnedCreate)
  NVJPEG_LIBRARY_FIND(nvjpegBufferPinnedDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegBufferDeviceCreate)
  NVJPEG_LIBRARY_FIND(nvjpegBufferDeviceDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegBufferPinnedRetrieve)
  NVJPEG_LIBRARY_FIND(nvjpegBufferDeviceRetrieve)
  NVJPEG_LIBRARY_FIND(nvjpegStateAttachPinnedBuffer)
  NVJPEG_LIBRARY_FIND(nvjpegStateAttachDeviceBuffer)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamCreate)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamParse)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamParseHeader)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamGetJpegEncoding)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamGetFrameDimensions)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamGetComponentsNum)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamGetComponentDimensions)
  NVJPEG_LIBRARY_FIND(nvjpegJpegStreamGetChromaSubsampling)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeParamsCreate)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeParamsDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeParamsSetOutputFormat)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeParamsSetROI)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeParamsSetAllowCMYK)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeParamsSetScaleFactor)
  NVJPEG_LIBRARY_FIND(nvjpegDecoderCreate)
  NVJPEG_LIBRARY_FIND(nvjpegDecoderDestroy)
  NVJPEG_LIBRARY_FIND(nvjpegDecoderJpegSupported)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeBatchedSupported)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeBatchedSupportedEx)
  NVJPEG_LIBRARY_FIND(nvjpegDecoderStateCreate)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeJpeg)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeJpegHost)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeJpegTransferToDevice)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeJpegDevice)
  NVJPEG_LIBRARY_FIND(nvjpegDecodeBatchedEx)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsCopyMetadata)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsCopyQuantizationTables)
  NVJPEG_LIBRARY_FIND(nvjpegEncoderParamsCopyHuffmanTables)
  result = 0; // success
  return result;
}
