/* This file is automatically generated. */
/* The content of this header file is a derived work of corresponding CUDA header files, */
/* and will need to comply NVIDIA CUDA SDK EULA. */


/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#ifndef CUEW_NVJPEG_H_
#define CUEW_NVJPEG_H_

#include <stdint.h>


#include <stddef.h>


#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


#ifndef CUDAAPI
#ifdef _WIN32
#  define CUDAAPI __stdcall
#  define CUDA_CB __stdcall
#else
#  define CUDAAPI
#  define CUDA_CB
#endif
#endif

/*struct cudaChannelFormatDesc; */

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wpadded"
#if __has_warning("-Wdocumentation-deprecated-sync")
  #pragma clang diagnostic ignored "-Wdocumentation-deprecated-sync"
#endif
#endif
#include "cuew_cudart.h"
#include "library_types.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif
// from cuda11-5 SDK

// Maximum number of channels nvjpeg decoder supports
#define NVJPEG_MAX_COMPONENT 4

// nvjpeg version information
#define NVJPEG_VER_MAJOR 11
#define NVJPEG_VER_MINOR 5
#define NVJPEG_VER_PATCH 4
#define NVJPEG_VER_BUILD 107


#define NVJPEG_FLAGS_DEFAULT 0
#define NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE 1
#define NVJPEG_FLAGS_ENABLE_MEMORY_POOLS   1<<1
#define NVJPEG_FLAGS_BITSTREAM_STRICT  1<<2

// Output descriptor.
// Data that is written to planes depends on output format
typedef struct
{
    unsigned char * channel[NVJPEG_MAX_COMPONENT];
    size_t    pitch[NVJPEG_MAX_COMPONENT];
} nvjpegImage_t;

// Prototype for device memory allocation, modelled after cudaMalloc()
typedef int (*tDevMalloc)(void**, size_t);
// Prototype for device memory release
typedef int (*tDevFree)(void*);

// Prototype for pinned memory allocation, modelled after cudaHostAlloc()
typedef int (*tPinnedMalloc)(void**, size_t, unsigned int flags);
// Prototype for device memory release
typedef int (*tPinnedFree)(void*);

// Memory allocator using mentioned prototypes, provided to nvjpegCreateEx
// This allocator will be used for all device memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct
{
    tDevMalloc dev_malloc;
    tDevFree dev_free;
} nvjpegDevAllocator_t;


// Pinned memory allocator using mentioned prototypes, provided to nvjpegCreate
// This allocator will be used for all pinned host memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct
{
    tPinnedMalloc pinned_malloc;
    tPinnedFree pinned_free;
} nvjpegPinnedAllocator_t;
typedef enum 
{
  NVJPEG_STATUS_SUCCESS = 0,
  NVJPEG_STATUS_NOT_INITIALIZED = 1,
  NVJPEG_STATUS_INVALID_PARAMETER = 2,
  NVJPEG_STATUS_BAD_JPEG = 3,
  NVJPEG_STATUS_JPEG_NOT_SUPPORTED = 4,
  NVJPEG_STATUS_ALLOCATOR_FAILURE = 5,
  NVJPEG_STATUS_EXECUTION_FAILED = 6,
  NVJPEG_STATUS_ARCH_MISMATCH = 7,
  NVJPEG_STATUS_INTERNAL_ERROR = 8,
  NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9
} nvjpegStatus_t; // id 0xe78fe0 

typedef enum 
{
  NVJPEG_CSS_444 = 0,
  NVJPEG_CSS_422 = 1,
  NVJPEG_CSS_420 = 2,
  NVJPEG_CSS_440 = 3,
  NVJPEG_CSS_411 = 4,
  NVJPEG_CSS_410 = 5,
  NVJPEG_CSS_GRAY = 6,
  NVJPEG_CSS_410V = 7,
  NVJPEG_CSS_UNKNOWN = -1
} nvjpegChromaSubsampling_t; // id 0xe796b8 

typedef enum 
{
  NVJPEG_OUTPUT_UNCHANGED = 0,
  NVJPEG_OUTPUT_YUV = 1,
  NVJPEG_OUTPUT_Y = 2,
  NVJPEG_OUTPUT_RGB = 3,
  NVJPEG_OUTPUT_BGR = 4,
  NVJPEG_OUTPUT_RGBI = 5,
  NVJPEG_OUTPUT_BGRI = 6,
  NVJPEG_OUTPUT_FORMAT_MAX = 6
} nvjpegOutputFormat_t; // id 0xe79d28 

typedef enum 
{
  NVJPEG_INPUT_RGB = 3,
  NVJPEG_INPUT_BGR = 4,
  NVJPEG_INPUT_RGBI = 5,
  NVJPEG_INPUT_BGRI = 6
} nvjpegInputFormat_t; // id 0xe7a328 

typedef enum 
{
  NVJPEG_BACKEND_DEFAULT = 0,
  NVJPEG_BACKEND_HYBRID = 1,
  NVJPEG_BACKEND_GPU_HYBRID = 2,
  NVJPEG_BACKEND_HARDWARE = 3
} nvjpegBackend_t; // id 0xe7a6d8 

typedef enum 
{
  NVJPEG_ENCODING_UNKNOWN = 0,
  NVJPEG_ENCODING_BASELINE_DCT = 192,
  NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN = 193,
  NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN = 194
} nvjpegJpegEncoding_t; // id 0xe7aa88 

typedef enum 
{
  NVJPEG_SCALE_NONE = 0,
  NVJPEG_SCALE_1_BY_2 = 1,
  NVJPEG_SCALE_1_BY_4 = 2,
  NVJPEG_SCALE_1_BY_8 = 3
} nvjpegScaleFactor_t; // id 0xe7ae38 

struct nvjpegHandle;
typedef struct nvjpegHandle * nvjpegHandle_t; // id 0xe7d500 

struct nvjpegJpegState;
typedef struct nvjpegJpegState * nvjpegJpegState_t; // id 0xe7d6b0 

typedef nvjpegStatus_t  CUDAAPI tnvjpegGetProperty(libraryPropertyType, int *);
extern tnvjpegGetProperty *nvjpegGetProperty;
typedef nvjpegStatus_t  CUDAAPI tnvjpegGetCudartProperty(libraryPropertyType, int *);
extern tnvjpegGetCudartProperty *nvjpegGetCudartProperty;
typedef nvjpegStatus_t  CUDAAPI tnvjpegCreate(nvjpegBackend_t, nvjpegDevAllocator_t *, nvjpegHandle_t *);
extern tnvjpegCreate *nvjpegCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegCreateSimple(nvjpegHandle_t *);
extern tnvjpegCreateSimple *nvjpegCreateSimple;
typedef nvjpegStatus_t  CUDAAPI tnvjpegCreateEx(nvjpegBackend_t, nvjpegDevAllocator_t *, nvjpegPinnedAllocator_t *, unsigned int, nvjpegHandle_t *);
extern tnvjpegCreateEx *nvjpegCreateEx;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDestroy(nvjpegHandle_t);
extern tnvjpegDestroy *nvjpegDestroy;
typedef nvjpegStatus_t  CUDAAPI tnvjpegSetDeviceMemoryPadding(size_t, nvjpegHandle_t);
extern tnvjpegSetDeviceMemoryPadding *nvjpegSetDeviceMemoryPadding;
typedef nvjpegStatus_t  CUDAAPI tnvjpegGetDeviceMemoryPadding(size_t *, nvjpegHandle_t);
extern tnvjpegGetDeviceMemoryPadding *nvjpegGetDeviceMemoryPadding;
typedef nvjpegStatus_t  CUDAAPI tnvjpegSetPinnedMemoryPadding(size_t, nvjpegHandle_t);
extern tnvjpegSetPinnedMemoryPadding *nvjpegSetPinnedMemoryPadding;
typedef nvjpegStatus_t  CUDAAPI tnvjpegGetPinnedMemoryPadding(size_t *, nvjpegHandle_t);
extern tnvjpegGetPinnedMemoryPadding *nvjpegGetPinnedMemoryPadding;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStateCreate(nvjpegHandle_t, nvjpegJpegState_t *);
extern tnvjpegJpegStateCreate *nvjpegJpegStateCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStateDestroy(nvjpegJpegState_t);
extern tnvjpegJpegStateDestroy *nvjpegJpegStateDestroy;
typedef nvjpegStatus_t  CUDAAPI tnvjpegGetImageInfo(nvjpegHandle_t, const unsigned char *, size_t, int *, nvjpegChromaSubsampling_t *, int *, int *);
extern tnvjpegGetImageInfo *nvjpegGetImageInfo;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecode(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *, size_t, nvjpegOutputFormat_t, nvjpegImage_t *, cudaStream_t);
extern tnvjpegDecode *nvjpegDecode;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeBatchedInitialize(nvjpegHandle_t, nvjpegJpegState_t, int, int, nvjpegOutputFormat_t);
extern tnvjpegDecodeBatchedInitialize *nvjpegDecodeBatchedInitialize;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeBatched(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *const *, const size_t *, nvjpegImage_t *, cudaStream_t);
extern tnvjpegDecodeBatched *nvjpegDecodeBatched;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeBatchedPreAllocate(nvjpegHandle_t, nvjpegJpegState_t, int, int, int, nvjpegChromaSubsampling_t, nvjpegOutputFormat_t);
extern tnvjpegDecodeBatchedPreAllocate *nvjpegDecodeBatchedPreAllocate;
struct nvjpegEncoderState;
typedef struct nvjpegEncoderState * nvjpegEncoderState_t; // id 0xe820c0 

typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderStateCreate(nvjpegHandle_t, nvjpegEncoderState_t *, cudaStream_t);
extern tnvjpegEncoderStateCreate *nvjpegEncoderStateCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderStateDestroy(nvjpegEncoderState_t);
extern tnvjpegEncoderStateDestroy *nvjpegEncoderStateDestroy;
struct nvjpegEncoderParams;
typedef struct nvjpegEncoderParams * nvjpegEncoderParams_t; // id 0xe82770 

typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsCreate(nvjpegHandle_t, nvjpegEncoderParams_t *, cudaStream_t);
extern tnvjpegEncoderParamsCreate *nvjpegEncoderParamsCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsDestroy(nvjpegEncoderParams_t);
extern tnvjpegEncoderParamsDestroy *nvjpegEncoderParamsDestroy;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsSetQuality(nvjpegEncoderParams_t, const int, cudaStream_t);
extern tnvjpegEncoderParamsSetQuality *nvjpegEncoderParamsSetQuality;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsSetEncoding(nvjpegEncoderParams_t, nvjpegJpegEncoding_t, cudaStream_t);
extern tnvjpegEncoderParamsSetEncoding *nvjpegEncoderParamsSetEncoding;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsSetOptimizedHuffman(nvjpegEncoderParams_t, const int, cudaStream_t);
extern tnvjpegEncoderParamsSetOptimizedHuffman *nvjpegEncoderParamsSetOptimizedHuffman;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsSetSamplingFactors(nvjpegEncoderParams_t, const nvjpegChromaSubsampling_t, cudaStream_t);
extern tnvjpegEncoderParamsSetSamplingFactors *nvjpegEncoderParamsSetSamplingFactors;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncodeGetBufferSize(nvjpegHandle_t, const nvjpegEncoderParams_t, int, int, size_t *);
extern tnvjpegEncodeGetBufferSize *nvjpegEncodeGetBufferSize;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncodeYUV(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t, const nvjpegImage_t *, nvjpegChromaSubsampling_t, int, int, cudaStream_t);
extern tnvjpegEncodeYUV *nvjpegEncodeYUV;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncodeImage(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t, const nvjpegImage_t *, nvjpegInputFormat_t, int, int, cudaStream_t);
extern tnvjpegEncodeImage *nvjpegEncodeImage;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncodeRetrieveBitstreamDevice(nvjpegHandle_t, nvjpegEncoderState_t, unsigned char *, size_t *, cudaStream_t);
extern tnvjpegEncodeRetrieveBitstreamDevice *nvjpegEncodeRetrieveBitstreamDevice;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncodeRetrieveBitstream(nvjpegHandle_t, nvjpegEncoderState_t, unsigned char *, size_t *, cudaStream_t);
extern tnvjpegEncodeRetrieveBitstream *nvjpegEncodeRetrieveBitstream;
struct nvjpegBufferPinned;
typedef struct nvjpegBufferPinned * nvjpegBufferPinned_t; // id 0xe86290 

typedef nvjpegStatus_t  CUDAAPI tnvjpegBufferPinnedCreate(nvjpegHandle_t, nvjpegPinnedAllocator_t *, nvjpegBufferPinned_t *);
extern tnvjpegBufferPinnedCreate *nvjpegBufferPinnedCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegBufferPinnedDestroy(nvjpegBufferPinned_t);
extern tnvjpegBufferPinnedDestroy *nvjpegBufferPinnedDestroy;
struct nvjpegBufferDevice;
typedef struct nvjpegBufferDevice * nvjpegBufferDevice_t; // id 0xe86940 

typedef nvjpegStatus_t  CUDAAPI tnvjpegBufferDeviceCreate(nvjpegHandle_t, nvjpegDevAllocator_t *, nvjpegBufferDevice_t *);
extern tnvjpegBufferDeviceCreate *nvjpegBufferDeviceCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegBufferDeviceDestroy(nvjpegBufferDevice_t);
extern tnvjpegBufferDeviceDestroy *nvjpegBufferDeviceDestroy;
typedef nvjpegStatus_t  CUDAAPI tnvjpegBufferPinnedRetrieve(nvjpegBufferPinned_t, size_t *, void **);
extern tnvjpegBufferPinnedRetrieve *nvjpegBufferPinnedRetrieve;
typedef nvjpegStatus_t  CUDAAPI tnvjpegBufferDeviceRetrieve(nvjpegBufferDevice_t, size_t *, void **);
extern tnvjpegBufferDeviceRetrieve *nvjpegBufferDeviceRetrieve;
typedef nvjpegStatus_t  CUDAAPI tnvjpegStateAttachPinnedBuffer(nvjpegJpegState_t, nvjpegBufferPinned_t);
extern tnvjpegStateAttachPinnedBuffer *nvjpegStateAttachPinnedBuffer;
typedef nvjpegStatus_t  CUDAAPI tnvjpegStateAttachDeviceBuffer(nvjpegJpegState_t, nvjpegBufferDevice_t);
extern tnvjpegStateAttachDeviceBuffer *nvjpegStateAttachDeviceBuffer;
struct nvjpegJpegStream;
typedef struct nvjpegJpegStream * nvjpegJpegStream_t; // id 0xe87ae0 

typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamCreate(nvjpegHandle_t, nvjpegJpegStream_t *);
extern tnvjpegJpegStreamCreate *nvjpegJpegStreamCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamDestroy(nvjpegJpegStream_t);
extern tnvjpegJpegStreamDestroy *nvjpegJpegStreamDestroy;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamParse(nvjpegHandle_t, const unsigned char *, size_t, int, int, nvjpegJpegStream_t);
extern tnvjpegJpegStreamParse *nvjpegJpegStreamParse;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamParseHeader(nvjpegHandle_t, const unsigned char *, size_t, nvjpegJpegStream_t);
extern tnvjpegJpegStreamParseHeader *nvjpegJpegStreamParseHeader;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamGetJpegEncoding(nvjpegJpegStream_t, nvjpegJpegEncoding_t *);
extern tnvjpegJpegStreamGetJpegEncoding *nvjpegJpegStreamGetJpegEncoding;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamGetFrameDimensions(nvjpegJpegStream_t, unsigned int *, unsigned int *);
extern tnvjpegJpegStreamGetFrameDimensions *nvjpegJpegStreamGetFrameDimensions;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamGetComponentsNum(nvjpegJpegStream_t, unsigned int *);
extern tnvjpegJpegStreamGetComponentsNum *nvjpegJpegStreamGetComponentsNum;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamGetComponentDimensions(nvjpegJpegStream_t, unsigned int, unsigned int *, unsigned int *);
extern tnvjpegJpegStreamGetComponentDimensions *nvjpegJpegStreamGetComponentDimensions;
typedef nvjpegStatus_t  CUDAAPI tnvjpegJpegStreamGetChromaSubsampling(nvjpegJpegStream_t, nvjpegChromaSubsampling_t *);
extern tnvjpegJpegStreamGetChromaSubsampling *nvjpegJpegStreamGetChromaSubsampling;
struct nvjpegDecodeParams;
typedef struct nvjpegDecodeParams * nvjpegDecodeParams_t; // id 0xe89780 

typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeParamsCreate(nvjpegHandle_t, nvjpegDecodeParams_t *);
extern tnvjpegDecodeParamsCreate *nvjpegDecodeParamsCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeParamsDestroy(nvjpegDecodeParams_t);
extern tnvjpegDecodeParamsDestroy *nvjpegDecodeParamsDestroy;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeParamsSetOutputFormat(nvjpegDecodeParams_t, nvjpegOutputFormat_t);
extern tnvjpegDecodeParamsSetOutputFormat *nvjpegDecodeParamsSetOutputFormat;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeParamsSetROI(nvjpegDecodeParams_t, int, int, int, int);
extern tnvjpegDecodeParamsSetROI *nvjpegDecodeParamsSetROI;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeParamsSetAllowCMYK(nvjpegDecodeParams_t, int);
extern tnvjpegDecodeParamsSetAllowCMYK *nvjpegDecodeParamsSetAllowCMYK;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeParamsSetScaleFactor(nvjpegDecodeParams_t, nvjpegScaleFactor_t);
extern tnvjpegDecodeParamsSetScaleFactor *nvjpegDecodeParamsSetScaleFactor;
struct nvjpegJpegDecoder;
typedef struct nvjpegJpegDecoder * nvjpegJpegDecoder_t; // id 0xe8a980 

typedef nvjpegStatus_t  CUDAAPI tnvjpegDecoderCreate(nvjpegHandle_t, nvjpegBackend_t, nvjpegJpegDecoder_t *);
extern tnvjpegDecoderCreate *nvjpegDecoderCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecoderDestroy(nvjpegJpegDecoder_t);
extern tnvjpegDecoderDestroy *nvjpegDecoderDestroy;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecoderJpegSupported(nvjpegJpegDecoder_t, nvjpegJpegStream_t, nvjpegDecodeParams_t, int *);
extern tnvjpegDecoderJpegSupported *nvjpegDecoderJpegSupported;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeBatchedSupported(nvjpegHandle_t, nvjpegJpegStream_t, int *);
extern tnvjpegDecodeBatchedSupported *nvjpegDecodeBatchedSupported;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeBatchedSupportedEx(nvjpegHandle_t, nvjpegJpegStream_t, nvjpegDecodeParams_t, int *);
extern tnvjpegDecodeBatchedSupportedEx *nvjpegDecodeBatchedSupportedEx;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecoderStateCreate(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t *);
extern tnvjpegDecoderStateCreate *nvjpegDecoderStateCreate;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeJpeg(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegJpegStream_t, nvjpegImage_t *, nvjpegDecodeParams_t, cudaStream_t);
extern tnvjpegDecodeJpeg *nvjpegDecodeJpeg;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeJpegHost(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegDecodeParams_t, nvjpegJpegStream_t);
extern tnvjpegDecodeJpegHost *nvjpegDecodeJpegHost;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeJpegTransferToDevice(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegJpegStream_t, cudaStream_t);
extern tnvjpegDecodeJpegTransferToDevice *nvjpegDecodeJpegTransferToDevice;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeJpegDevice(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegImage_t *, cudaStream_t);
extern tnvjpegDecodeJpegDevice *nvjpegDecodeJpegDevice;
typedef nvjpegStatus_t  CUDAAPI tnvjpegDecodeBatchedEx(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *const *, const size_t *, nvjpegImage_t *, nvjpegDecodeParams_t *, cudaStream_t);
extern tnvjpegDecodeBatchedEx *nvjpegDecodeBatchedEx;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsCopyMetadata(nvjpegEncoderState_t, nvjpegEncoderParams_t, nvjpegJpegStream_t, cudaStream_t);
extern tnvjpegEncoderParamsCopyMetadata *nvjpegEncoderParamsCopyMetadata;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsCopyQuantizationTables(nvjpegEncoderParams_t, nvjpegJpegStream_t, cudaStream_t);
extern tnvjpegEncoderParamsCopyQuantizationTables *nvjpegEncoderParamsCopyQuantizationTables;
typedef nvjpegStatus_t  CUDAAPI tnvjpegEncoderParamsCopyHuffmanTables(nvjpegEncoderState_t, nvjpegEncoderParams_t, nvjpegJpegStream_t, cudaStream_t);
extern tnvjpegEncoderParamsCopyHuffmanTables *nvjpegEncoderParamsCopyHuffmanTables;
extern int cuewInitNVJPEG(void);


#ifdef __cplusplus
}
#endif
#endif /* CUEW_NVJPEG_H_ */
