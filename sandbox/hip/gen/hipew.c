

#ifdef _MSC_VER
#  if _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#  define popen _popen
#  define pclose _pclose
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include "hipew.h"
#include "hipew_hip.h"
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

/*#define HIP_LIBRARY_FIND_CHECKED(name) HIPEW_IMPL_LIBRARY_FIND_CHECKED(hip_lib, name)*/
#define HIP_LIBRARY_FIND(name) HIPEW_IMPL_LIBRARY_FIND(hip_lib, name)
static DynamicLibrary hip_lib;

static void hipewExitHIP(void) {
  if (hip_lib != NULL) {
    /* ignore errors */
    dynamic_library_close(hip_lib);
    hip_lib = NULL;
  }
}

thipCreateChannelDesc *hipCreateChannelDesc;
thipInit *hipInit;
thipDriverGetVersion *hipDriverGetVersion;
thipRuntimeGetVersion *hipRuntimeGetVersion;
thipDeviceGet *hipDeviceGet;
thipDeviceComputeCapability *hipDeviceComputeCapability;
thipDeviceGetName *hipDeviceGetName;
thipDeviceGetUuid *hipDeviceGetUuid;
thipDeviceGetP2PAttribute *hipDeviceGetP2PAttribute;
thipDeviceGetPCIBusId *hipDeviceGetPCIBusId;
thipDeviceGetByPCIBusId *hipDeviceGetByPCIBusId;
thipDeviceTotalMem *hipDeviceTotalMem;
thipDeviceSynchronize *hipDeviceSynchronize;
thipDeviceReset *hipDeviceReset;
thipSetDevice *hipSetDevice;
thipGetDevice *hipGetDevice;
thipGetDeviceCount *hipGetDeviceCount;
thipDeviceGetAttribute *hipDeviceGetAttribute;
thipDeviceGetDefaultMemPool *hipDeviceGetDefaultMemPool;
thipDeviceSetMemPool *hipDeviceSetMemPool;
thipDeviceGetMemPool *hipDeviceGetMemPool;
thipGetDeviceProperties *hipGetDeviceProperties;
thipDeviceSetCacheConfig *hipDeviceSetCacheConfig;
thipDeviceGetCacheConfig *hipDeviceGetCacheConfig;
thipDeviceGetLimit *hipDeviceGetLimit;
thipDeviceSetLimit *hipDeviceSetLimit;
thipDeviceGetSharedMemConfig *hipDeviceGetSharedMemConfig;
thipGetDeviceFlags *hipGetDeviceFlags;
thipDeviceSetSharedMemConfig *hipDeviceSetSharedMemConfig;
thipSetDeviceFlags *hipSetDeviceFlags;
thipChooseDevice *hipChooseDevice;
thipExtGetLinkTypeAndHopCount *hipExtGetLinkTypeAndHopCount;
thipIpcGetMemHandle *hipIpcGetMemHandle;
thipIpcOpenMemHandle *hipIpcOpenMemHandle;
thipIpcCloseMemHandle *hipIpcCloseMemHandle;
thipIpcGetEventHandle *hipIpcGetEventHandle;
thipIpcOpenEventHandle *hipIpcOpenEventHandle;
thipFuncSetAttribute *hipFuncSetAttribute;
thipFuncSetCacheConfig *hipFuncSetCacheConfig;
thipFuncSetSharedMemConfig *hipFuncSetSharedMemConfig;
thipGetLastError *hipGetLastError;
thipPeekAtLastError *hipPeekAtLastError;
thipGetErrorName *hipGetErrorName;
thipGetErrorString *hipGetErrorString;
thipDrvGetErrorName *hipDrvGetErrorName;
thipDrvGetErrorString *hipDrvGetErrorString;
thipStreamCreate *hipStreamCreate;
thipStreamCreateWithFlags *hipStreamCreateWithFlags;
thipStreamCreateWithPriority *hipStreamCreateWithPriority;
thipDeviceGetStreamPriorityRange *hipDeviceGetStreamPriorityRange;
thipStreamDestroy *hipStreamDestroy;
thipStreamQuery *hipStreamQuery;
thipStreamSynchronize *hipStreamSynchronize;
thipStreamWaitEvent *hipStreamWaitEvent;
thipStreamGetFlags *hipStreamGetFlags;
thipStreamGetPriority *hipStreamGetPriority;
thipExtStreamCreateWithCUMask *hipExtStreamCreateWithCUMask;
thipExtStreamGetCUMask *hipExtStreamGetCUMask;
thipStreamAddCallback *hipStreamAddCallback;
thipStreamWaitValue32 *hipStreamWaitValue32;
thipStreamWaitValue64 *hipStreamWaitValue64;
thipStreamWriteValue32 *hipStreamWriteValue32;
thipStreamWriteValue64 *hipStreamWriteValue64;
thipEventCreateWithFlags *hipEventCreateWithFlags;
thipEventCreate *hipEventCreate;
thipEventRecord *hipEventRecord;
thipEventDestroy *hipEventDestroy;
thipEventSynchronize *hipEventSynchronize;
thipEventElapsedTime *hipEventElapsedTime;
thipEventQuery *hipEventQuery;
thipPointerSetAttribute *hipPointerSetAttribute;
thipPointerGetAttributes *hipPointerGetAttributes;
thipPointerGetAttribute *hipPointerGetAttribute;
thipDrvPointerGetAttributes *hipDrvPointerGetAttributes;
thipImportExternalSemaphore *hipImportExternalSemaphore;
thipSignalExternalSemaphoresAsync *hipSignalExternalSemaphoresAsync;
thipWaitExternalSemaphoresAsync *hipWaitExternalSemaphoresAsync;
thipDestroyExternalSemaphore *hipDestroyExternalSemaphore;
thipImportExternalMemory *hipImportExternalMemory;
thipExternalMemoryGetMappedBuffer *hipExternalMemoryGetMappedBuffer;
thipDestroyExternalMemory *hipDestroyExternalMemory;
thipMalloc *hipMalloc;
thipExtMallocWithFlags *hipExtMallocWithFlags;
thipMallocHost *hipMallocHost;
thipMemAllocHost *hipMemAllocHost;
thipHostMalloc *hipHostMalloc;
thipMallocManaged *hipMallocManaged;
thipMemPrefetchAsync *hipMemPrefetchAsync;
thipMemAdvise *hipMemAdvise;
thipMemRangeGetAttribute *hipMemRangeGetAttribute;
thipMemRangeGetAttributes *hipMemRangeGetAttributes;
thipStreamAttachMemAsync *hipStreamAttachMemAsync;
thipMallocAsync *hipMallocAsync;
thipFreeAsync *hipFreeAsync;
thipMemPoolTrimTo *hipMemPoolTrimTo;
thipMemPoolSetAttribute *hipMemPoolSetAttribute;
thipMemPoolGetAttribute *hipMemPoolGetAttribute;
thipMemPoolSetAccess *hipMemPoolSetAccess;
thipMemPoolGetAccess *hipMemPoolGetAccess;
thipMemPoolCreate *hipMemPoolCreate;
thipMemPoolDestroy *hipMemPoolDestroy;
thipMallocFromPoolAsync *hipMallocFromPoolAsync;
thipMemPoolExportToShareableHandle *hipMemPoolExportToShareableHandle;
thipMemPoolImportFromShareableHandle *hipMemPoolImportFromShareableHandle;
thipMemPoolExportPointer *hipMemPoolExportPointer;
thipMemPoolImportPointer *hipMemPoolImportPointer;
thipHostAlloc *hipHostAlloc;
thipHostGetDevicePointer *hipHostGetDevicePointer;
thipHostGetFlags *hipHostGetFlags;
thipHostRegister *hipHostRegister;
thipHostUnregister *hipHostUnregister;
thipMallocPitch *hipMallocPitch;
thipMemAllocPitch *hipMemAllocPitch;
thipFree *hipFree;
thipFreeHost *hipFreeHost;
thipHostFree *hipHostFree;
thipMemcpy *hipMemcpy;
thipMemcpyWithStream *hipMemcpyWithStream;
thipMemcpyHtoD *hipMemcpyHtoD;
thipMemcpyDtoH *hipMemcpyDtoH;
thipMemcpyDtoD *hipMemcpyDtoD;
thipMemcpyHtoDAsync *hipMemcpyHtoDAsync;
thipMemcpyDtoHAsync *hipMemcpyDtoHAsync;
thipMemcpyDtoDAsync *hipMemcpyDtoDAsync;
thipModuleGetGlobal *hipModuleGetGlobal;
thipGetSymbolAddress *hipGetSymbolAddress;
thipGetSymbolSize *hipGetSymbolSize;
thipMemcpyToSymbol *hipMemcpyToSymbol;
thipMemcpyToSymbolAsync *hipMemcpyToSymbolAsync;
thipMemcpyFromSymbol *hipMemcpyFromSymbol;
thipMemcpyFromSymbolAsync *hipMemcpyFromSymbolAsync;
thipMemcpyAsync *hipMemcpyAsync;
thipMemset *hipMemset;
thipMemsetD8 *hipMemsetD8;
thipMemsetD8Async *hipMemsetD8Async;
thipMemsetD16 *hipMemsetD16;
thipMemsetD16Async *hipMemsetD16Async;
thipMemsetD32 *hipMemsetD32;
thipMemsetAsync *hipMemsetAsync;
thipMemsetD32Async *hipMemsetD32Async;
thipMemset2D *hipMemset2D;
thipMemset2DAsync *hipMemset2DAsync;
thipMemset3D *hipMemset3D;
thipMemset3DAsync *hipMemset3DAsync;
thipMemGetInfo *hipMemGetInfo;
thipMemPtrGetInfo *hipMemPtrGetInfo;
thipMallocArray *hipMallocArray;
thipArrayCreate *hipArrayCreate;
thipArrayDestroy *hipArrayDestroy;
thipArray3DCreate *hipArray3DCreate;
thipMalloc3D *hipMalloc3D;
thipFreeArray *hipFreeArray;
thipFreeMipmappedArray *hipFreeMipmappedArray;
thipMalloc3DArray *hipMalloc3DArray;
thipMallocMipmappedArray *hipMallocMipmappedArray;
thipGetMipmappedArrayLevel *hipGetMipmappedArrayLevel;
thipMemcpy2D *hipMemcpy2D;
thipMemcpyParam2D *hipMemcpyParam2D;
thipMemcpyParam2DAsync *hipMemcpyParam2DAsync;
thipMemcpy2DAsync *hipMemcpy2DAsync;
thipMemcpy2DToArray *hipMemcpy2DToArray;
thipMemcpy2DToArrayAsync *hipMemcpy2DToArrayAsync;
thipMemcpyToArray *hipMemcpyToArray;
thipMemcpyFromArray *hipMemcpyFromArray;
thipMemcpy2DFromArray *hipMemcpy2DFromArray;
thipMemcpy2DFromArrayAsync *hipMemcpy2DFromArrayAsync;
thipMemcpyAtoH *hipMemcpyAtoH;
thipMemcpyHtoA *hipMemcpyHtoA;
thipMemcpy3D *hipMemcpy3D;
thipMemcpy3DAsync *hipMemcpy3DAsync;
thipDrvMemcpy3D *hipDrvMemcpy3D;
thipDrvMemcpy3DAsync *hipDrvMemcpy3DAsync;
thipDeviceCanAccessPeer *hipDeviceCanAccessPeer;
thipDeviceEnablePeerAccess *hipDeviceEnablePeerAccess;
thipDeviceDisablePeerAccess *hipDeviceDisablePeerAccess;
thipMemGetAddressRange *hipMemGetAddressRange;
thipMemcpyPeer *hipMemcpyPeer;
thipMemcpyPeerAsync *hipMemcpyPeerAsync;
thipCtxCreate *hipCtxCreate;
thipCtxDestroy *hipCtxDestroy;
thipCtxPopCurrent *hipCtxPopCurrent;
thipCtxPushCurrent *hipCtxPushCurrent;
thipCtxSetCurrent *hipCtxSetCurrent;
thipCtxGetCurrent *hipCtxGetCurrent;
thipCtxGetDevice *hipCtxGetDevice;
thipCtxGetApiVersion *hipCtxGetApiVersion;
thipCtxGetCacheConfig *hipCtxGetCacheConfig;
thipCtxSetCacheConfig *hipCtxSetCacheConfig;
thipCtxSetSharedMemConfig *hipCtxSetSharedMemConfig;
thipCtxGetSharedMemConfig *hipCtxGetSharedMemConfig;
thipCtxSynchronize *hipCtxSynchronize;
thipCtxGetFlags *hipCtxGetFlags;
thipCtxEnablePeerAccess *hipCtxEnablePeerAccess;
thipCtxDisablePeerAccess *hipCtxDisablePeerAccess;
thipDevicePrimaryCtxGetState *hipDevicePrimaryCtxGetState;
thipDevicePrimaryCtxRelease *hipDevicePrimaryCtxRelease;
thipDevicePrimaryCtxRetain *hipDevicePrimaryCtxRetain;
thipDevicePrimaryCtxReset *hipDevicePrimaryCtxReset;
thipDevicePrimaryCtxSetFlags *hipDevicePrimaryCtxSetFlags;
thipModuleLoad *hipModuleLoad;
thipModuleUnload *hipModuleUnload;
thipModuleGetFunction *hipModuleGetFunction;
thipFuncGetAttributes *hipFuncGetAttributes;
thipFuncGetAttribute *hipFuncGetAttribute;
thipModuleGetTexRef *hipModuleGetTexRef;
thipModuleLoadData *hipModuleLoadData;
thipModuleLoadDataEx *hipModuleLoadDataEx;
thipModuleLaunchKernel *hipModuleLaunchKernel;
thipModuleLaunchCooperativeKernel *hipModuleLaunchCooperativeKernel;
thipModuleLaunchCooperativeKernelMultiDevice *hipModuleLaunchCooperativeKernelMultiDevice;
thipLaunchCooperativeKernel *hipLaunchCooperativeKernel;
thipLaunchCooperativeKernelMultiDevice *hipLaunchCooperativeKernelMultiDevice;
thipExtLaunchMultiKernelMultiDevice *hipExtLaunchMultiKernelMultiDevice;
thipModuleOccupancyMaxPotentialBlockSize *hipModuleOccupancyMaxPotentialBlockSize;
thipModuleOccupancyMaxPotentialBlockSizeWithFlags *hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
thipModuleOccupancyMaxActiveBlocksPerMultiprocessor *hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
thipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags *hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
thipOccupancyMaxActiveBlocksPerMultiprocessor *hipOccupancyMaxActiveBlocksPerMultiprocessor;
thipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags *hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
thipOccupancyMaxPotentialBlockSize *hipOccupancyMaxPotentialBlockSize;
thipProfilerStart *hipProfilerStart;
thipProfilerStop *hipProfilerStop;
thipConfigureCall *hipConfigureCall;
thipSetupArgument *hipSetupArgument;
thipLaunchByPtr *hipLaunchByPtr;
thipLaunchKernel *hipLaunchKernel;
thipLaunchHostFunc *hipLaunchHostFunc;
thipDrvMemcpy2DUnaligned *hipDrvMemcpy2DUnaligned;
thipExtLaunchKernel *hipExtLaunchKernel;
thipBindTextureToMipmappedArray *hipBindTextureToMipmappedArray;
thipCreateTextureObject *hipCreateTextureObject;
thipDestroyTextureObject *hipDestroyTextureObject;
thipGetChannelDesc *hipGetChannelDesc;
thipGetTextureObjectResourceDesc *hipGetTextureObjectResourceDesc;
thipGetTextureObjectResourceViewDesc *hipGetTextureObjectResourceViewDesc;
thipGetTextureObjectTextureDesc *hipGetTextureObjectTextureDesc;
thipTexObjectCreate *hipTexObjectCreate;
thipTexObjectDestroy *hipTexObjectDestroy;
thipTexObjectGetResourceDesc *hipTexObjectGetResourceDesc;
thipTexObjectGetResourceViewDesc *hipTexObjectGetResourceViewDesc;
thipTexObjectGetTextureDesc *hipTexObjectGetTextureDesc;
thipGetTextureReference *hipGetTextureReference;
thipTexRefSetAddressMode *hipTexRefSetAddressMode;
thipTexRefSetArray *hipTexRefSetArray;
thipTexRefSetFilterMode *hipTexRefSetFilterMode;
thipTexRefSetFlags *hipTexRefSetFlags;
thipTexRefSetFormat *hipTexRefSetFormat;
thipBindTexture *hipBindTexture;
thipBindTexture2D *hipBindTexture2D;
thipBindTextureToArray *hipBindTextureToArray;
thipGetTextureAlignmentOffset *hipGetTextureAlignmentOffset;
thipUnbindTexture *hipUnbindTexture;
thipTexRefGetAddress *hipTexRefGetAddress;
thipTexRefGetAddressMode *hipTexRefGetAddressMode;
thipTexRefGetFilterMode *hipTexRefGetFilterMode;
thipTexRefGetFlags *hipTexRefGetFlags;
thipTexRefGetFormat *hipTexRefGetFormat;
thipTexRefGetMaxAnisotropy *hipTexRefGetMaxAnisotropy;
thipTexRefGetMipmapFilterMode *hipTexRefGetMipmapFilterMode;
thipTexRefGetMipmapLevelBias *hipTexRefGetMipmapLevelBias;
thipTexRefGetMipmapLevelClamp *hipTexRefGetMipmapLevelClamp;
thipTexRefGetMipMappedArray *hipTexRefGetMipMappedArray;
thipTexRefSetAddress *hipTexRefSetAddress;
thipTexRefSetAddress2D *hipTexRefSetAddress2D;
thipTexRefSetMaxAnisotropy *hipTexRefSetMaxAnisotropy;
thipTexRefSetBorderColor *hipTexRefSetBorderColor;
thipTexRefSetMipmapFilterMode *hipTexRefSetMipmapFilterMode;
thipTexRefSetMipmapLevelBias *hipTexRefSetMipmapLevelBias;
thipTexRefSetMipmapLevelClamp *hipTexRefSetMipmapLevelClamp;
thipTexRefSetMipmappedArray *hipTexRefSetMipmappedArray;
thipMipmappedArrayCreate *hipMipmappedArrayCreate;
thipMipmappedArrayDestroy *hipMipmappedArrayDestroy;
thipMipmappedArrayGetLevel *hipMipmappedArrayGetLevel;
thipApiName *hipApiName;
thipKernelNameRef *hipKernelNameRef;
thipKernelNameRefByPtr *hipKernelNameRefByPtr;
thipGetStreamDeviceId *hipGetStreamDeviceId;
thipStreamBeginCapture *hipStreamBeginCapture;
thipStreamEndCapture *hipStreamEndCapture;
thipStreamGetCaptureInfo *hipStreamGetCaptureInfo;
thipStreamGetCaptureInfo_v2 *hipStreamGetCaptureInfo_v2;
thipStreamIsCapturing *hipStreamIsCapturing;
thipStreamUpdateCaptureDependencies *hipStreamUpdateCaptureDependencies;
thipThreadExchangeStreamCaptureMode *hipThreadExchangeStreamCaptureMode;
thipGraphCreate *hipGraphCreate;
thipGraphDestroy *hipGraphDestroy;
thipGraphAddDependencies *hipGraphAddDependencies;
thipGraphRemoveDependencies *hipGraphRemoveDependencies;
thipGraphGetEdges *hipGraphGetEdges;
thipGraphGetNodes *hipGraphGetNodes;
thipGraphGetRootNodes *hipGraphGetRootNodes;
thipGraphNodeGetDependencies *hipGraphNodeGetDependencies;
thipGraphNodeGetDependentNodes *hipGraphNodeGetDependentNodes;
thipGraphNodeGetType *hipGraphNodeGetType;
thipGraphDestroyNode *hipGraphDestroyNode;
thipGraphClone *hipGraphClone;
thipGraphNodeFindInClone *hipGraphNodeFindInClone;
thipGraphInstantiate *hipGraphInstantiate;
thipGraphInstantiateWithFlags *hipGraphInstantiateWithFlags;
thipGraphLaunch *hipGraphLaunch;
thipGraphUpload *hipGraphUpload;
thipGraphExecDestroy *hipGraphExecDestroy;
thipGraphExecUpdate *hipGraphExecUpdate;
thipGraphAddKernelNode *hipGraphAddKernelNode;
thipGraphKernelNodeGetParams *hipGraphKernelNodeGetParams;
thipGraphKernelNodeSetParams *hipGraphKernelNodeSetParams;
thipGraphExecKernelNodeSetParams *hipGraphExecKernelNodeSetParams;
thipGraphAddMemcpyNode *hipGraphAddMemcpyNode;
thipGraphMemcpyNodeGetParams *hipGraphMemcpyNodeGetParams;
thipGraphMemcpyNodeSetParams *hipGraphMemcpyNodeSetParams;
thipGraphKernelNodeSetAttribute *hipGraphKernelNodeSetAttribute;
thipGraphKernelNodeGetAttribute *hipGraphKernelNodeGetAttribute;
thipGraphExecMemcpyNodeSetParams *hipGraphExecMemcpyNodeSetParams;
thipGraphAddMemcpyNode1D *hipGraphAddMemcpyNode1D;
thipGraphMemcpyNodeSetParams1D *hipGraphMemcpyNodeSetParams1D;
thipGraphExecMemcpyNodeSetParams1D *hipGraphExecMemcpyNodeSetParams1D;
thipGraphAddMemcpyNodeFromSymbol *hipGraphAddMemcpyNodeFromSymbol;
thipGraphMemcpyNodeSetParamsFromSymbol *hipGraphMemcpyNodeSetParamsFromSymbol;
thipGraphExecMemcpyNodeSetParamsFromSymbol *hipGraphExecMemcpyNodeSetParamsFromSymbol;
thipGraphAddMemcpyNodeToSymbol *hipGraphAddMemcpyNodeToSymbol;
thipGraphMemcpyNodeSetParamsToSymbol *hipGraphMemcpyNodeSetParamsToSymbol;
thipGraphExecMemcpyNodeSetParamsToSymbol *hipGraphExecMemcpyNodeSetParamsToSymbol;
thipGraphAddMemsetNode *hipGraphAddMemsetNode;
thipGraphMemsetNodeGetParams *hipGraphMemsetNodeGetParams;
thipGraphMemsetNodeSetParams *hipGraphMemsetNodeSetParams;
thipGraphExecMemsetNodeSetParams *hipGraphExecMemsetNodeSetParams;
thipGraphAddHostNode *hipGraphAddHostNode;
thipGraphHostNodeGetParams *hipGraphHostNodeGetParams;
thipGraphHostNodeSetParams *hipGraphHostNodeSetParams;
thipGraphExecHostNodeSetParams *hipGraphExecHostNodeSetParams;
thipGraphAddChildGraphNode *hipGraphAddChildGraphNode;
thipGraphChildGraphNodeGetGraph *hipGraphChildGraphNodeGetGraph;
thipGraphExecChildGraphNodeSetParams *hipGraphExecChildGraphNodeSetParams;
thipGraphAddEmptyNode *hipGraphAddEmptyNode;
thipGraphAddEventRecordNode *hipGraphAddEventRecordNode;
thipGraphEventRecordNodeGetEvent *hipGraphEventRecordNodeGetEvent;
thipGraphEventRecordNodeSetEvent *hipGraphEventRecordNodeSetEvent;
thipGraphExecEventRecordNodeSetEvent *hipGraphExecEventRecordNodeSetEvent;
thipGraphAddEventWaitNode *hipGraphAddEventWaitNode;
thipGraphEventWaitNodeGetEvent *hipGraphEventWaitNodeGetEvent;
thipGraphEventWaitNodeSetEvent *hipGraphEventWaitNodeSetEvent;
thipGraphExecEventWaitNodeSetEvent *hipGraphExecEventWaitNodeSetEvent;
thipGraphAddMemAllocNode *hipGraphAddMemAllocNode;
thipGraphMemAllocNodeGetParams *hipGraphMemAllocNodeGetParams;
thipGraphAddMemFreeNode *hipGraphAddMemFreeNode;
thipGraphMemFreeNodeGetParams *hipGraphMemFreeNodeGetParams;
thipDeviceGetGraphMemAttribute *hipDeviceGetGraphMemAttribute;
thipDeviceSetGraphMemAttribute *hipDeviceSetGraphMemAttribute;
thipDeviceGraphMemTrim *hipDeviceGraphMemTrim;
thipUserObjectCreate *hipUserObjectCreate;
thipUserObjectRelease *hipUserObjectRelease;
thipUserObjectRetain *hipUserObjectRetain;
thipGraphRetainUserObject *hipGraphRetainUserObject;
thipGraphReleaseUserObject *hipGraphReleaseUserObject;
thipGraphDebugDotPrint *hipGraphDebugDotPrint;
thipGraphKernelNodeCopyAttributes *hipGraphKernelNodeCopyAttributes;
thipGraphNodeSetEnabled *hipGraphNodeSetEnabled;
thipGraphNodeGetEnabled *hipGraphNodeGetEnabled;
thipMemAddressFree *hipMemAddressFree;
thipMemAddressReserve *hipMemAddressReserve;
thipMemCreate *hipMemCreate;
thipMemExportToShareableHandle *hipMemExportToShareableHandle;
thipMemGetAccess *hipMemGetAccess;
thipMemGetAllocationGranularity *hipMemGetAllocationGranularity;
thipMemGetAllocationPropertiesFromHandle *hipMemGetAllocationPropertiesFromHandle;
thipMemImportFromShareableHandle *hipMemImportFromShareableHandle;
thipMemMap *hipMemMap;
thipMemMapArrayAsync *hipMemMapArrayAsync;
thipMemRelease *hipMemRelease;
thipMemRetainAllocationHandle *hipMemRetainAllocationHandle;
thipMemSetAccess *hipMemSetAccess;
thipMemUnmap *hipMemUnmap;
thipGLGetDevices *hipGLGetDevices;
//thipGraphicsGLRegisterBuffer *hipGraphicsGLRegisterBuffer;
//thipGraphicsGLRegisterImage *hipGraphicsGLRegisterImage;
//thipGraphicsMapResources *hipGraphicsMapResources;
//thipGraphicsSubResourceGetMappedArray *hipGraphicsSubResourceGetMappedArray;
//thipGraphicsResourceGetMappedPointer *hipGraphicsResourceGetMappedPointer;
//thipGraphicsUnmapResources *hipGraphicsUnmapResources;
//thipGraphicsUnregisterResource *hipGraphicsUnregisterResource;
thipMemcpy_spt *hipMemcpy_spt;
thipMemcpyToSymbol_spt *hipMemcpyToSymbol_spt;
thipMemcpyFromSymbol_spt *hipMemcpyFromSymbol_spt;
thipMemcpy2D_spt *hipMemcpy2D_spt;
thipMemcpy2DFromArray_spt *hipMemcpy2DFromArray_spt;
thipMemcpy3D_spt *hipMemcpy3D_spt;
thipMemset_spt *hipMemset_spt;
thipMemsetAsync_spt *hipMemsetAsync_spt;
thipMemset2D_spt *hipMemset2D_spt;
thipMemset2DAsync_spt *hipMemset2DAsync_spt;
thipMemset3DAsync_spt *hipMemset3DAsync_spt;
thipMemset3D_spt *hipMemset3D_spt;
thipMemcpyAsync_spt *hipMemcpyAsync_spt;
thipMemcpy3DAsync_spt *hipMemcpy3DAsync_spt;
thipMemcpy2DAsync_spt *hipMemcpy2DAsync_spt;
thipMemcpyFromSymbolAsync_spt *hipMemcpyFromSymbolAsync_spt;
thipMemcpyToSymbolAsync_spt *hipMemcpyToSymbolAsync_spt;
thipMemcpyFromArray_spt *hipMemcpyFromArray_spt;
thipMemcpy2DToArray_spt *hipMemcpy2DToArray_spt;
thipMemcpy2DFromArrayAsync_spt *hipMemcpy2DFromArrayAsync_spt;
thipMemcpy2DToArrayAsync_spt *hipMemcpy2DToArrayAsync_spt;
thipStreamQuery_spt *hipStreamQuery_spt;
thipStreamSynchronize_spt *hipStreamSynchronize_spt;
thipStreamGetPriority_spt *hipStreamGetPriority_spt;
thipStreamWaitEvent_spt *hipStreamWaitEvent_spt;
thipStreamGetFlags_spt *hipStreamGetFlags_spt;
thipStreamAddCallback_spt *hipStreamAddCallback_spt;
thipEventRecord_spt *hipEventRecord_spt;
thipLaunchCooperativeKernel_spt *hipLaunchCooperativeKernel_spt;
thipLaunchKernel_spt *hipLaunchKernel_spt;
thipGraphLaunch_spt *hipGraphLaunch_spt;
thipStreamBeginCapture_spt *hipStreamBeginCapture_spt;
thipStreamEndCapture_spt *hipStreamEndCapture_spt;
thipStreamIsCapturing_spt *hipStreamIsCapturing_spt;
thipStreamGetCaptureInfo_spt *hipStreamGetCaptureInfo_spt;
thipStreamGetCaptureInfo_v2_spt *hipStreamGetCaptureInfo_v2_spt;
thipLaunchHostFunc_spt *hipLaunchHostFunc_spt;

int hipewInitHIP(const char **extra_dll_search_paths) {

#ifdef _WIN32
  const char *paths[] = {   "hip_runtime.dll",
NULL};
#else /* linux */
  const char *paths[] = {   "libamdhip64.so",
   "/opt/rocm/lib/libamdhip64.so",
NULL};
#endif


  static int initialized = 0;
  static int result = 0;
  int error;

  if (initialized) {
    return result;
  }

  initialized = 1;
  error = atexit(hipewExitHIP);

  if (error) {
    result = -2;
    return result;
  }
  hip_lib = dynamic_library_open_find(paths);
  if (hip_lib == NULL) { 
    if (extra_dll_search_paths) { 
      hip_lib = dynamic_library_open_find(extra_dll_search_paths);
    }
  }
  if (hip_lib == NULL) { result = -1; return result; }

  HIP_LIBRARY_FIND(hipCreateChannelDesc)
  HIP_LIBRARY_FIND(hipInit)
  HIP_LIBRARY_FIND(hipDriverGetVersion)
  HIP_LIBRARY_FIND(hipRuntimeGetVersion)
  HIP_LIBRARY_FIND(hipDeviceGet)
  HIP_LIBRARY_FIND(hipDeviceComputeCapability)
  HIP_LIBRARY_FIND(hipDeviceGetName)
  HIP_LIBRARY_FIND(hipDeviceGetUuid)
  HIP_LIBRARY_FIND(hipDeviceGetP2PAttribute)
  HIP_LIBRARY_FIND(hipDeviceGetPCIBusId)
  HIP_LIBRARY_FIND(hipDeviceGetByPCIBusId)
  HIP_LIBRARY_FIND(hipDeviceTotalMem)
  HIP_LIBRARY_FIND(hipDeviceSynchronize)
  HIP_LIBRARY_FIND(hipDeviceReset)
  HIP_LIBRARY_FIND(hipSetDevice)
  HIP_LIBRARY_FIND(hipGetDevice)
  HIP_LIBRARY_FIND(hipGetDeviceCount)
  HIP_LIBRARY_FIND(hipDeviceGetAttribute)
  HIP_LIBRARY_FIND(hipDeviceGetDefaultMemPool)
  HIP_LIBRARY_FIND(hipDeviceSetMemPool)
  HIP_LIBRARY_FIND(hipDeviceGetMemPool)
  HIP_LIBRARY_FIND(hipGetDeviceProperties)
  HIP_LIBRARY_FIND(hipDeviceSetCacheConfig)
  HIP_LIBRARY_FIND(hipDeviceGetCacheConfig)
  HIP_LIBRARY_FIND(hipDeviceGetLimit)
  HIP_LIBRARY_FIND(hipDeviceSetLimit)
  HIP_LIBRARY_FIND(hipDeviceGetSharedMemConfig)
  HIP_LIBRARY_FIND(hipGetDeviceFlags)
  HIP_LIBRARY_FIND(hipDeviceSetSharedMemConfig)
  HIP_LIBRARY_FIND(hipSetDeviceFlags)
  HIP_LIBRARY_FIND(hipChooseDevice)
  HIP_LIBRARY_FIND(hipExtGetLinkTypeAndHopCount)
  HIP_LIBRARY_FIND(hipIpcGetMemHandle)
  HIP_LIBRARY_FIND(hipIpcOpenMemHandle)
  HIP_LIBRARY_FIND(hipIpcCloseMemHandle)
  HIP_LIBRARY_FIND(hipIpcGetEventHandle)
  HIP_LIBRARY_FIND(hipIpcOpenEventHandle)
  HIP_LIBRARY_FIND(hipFuncSetAttribute)
  HIP_LIBRARY_FIND(hipFuncSetCacheConfig)
  HIP_LIBRARY_FIND(hipFuncSetSharedMemConfig)
  HIP_LIBRARY_FIND(hipGetLastError)
  HIP_LIBRARY_FIND(hipPeekAtLastError)
  HIP_LIBRARY_FIND(hipGetErrorName)
  HIP_LIBRARY_FIND(hipGetErrorString)
  HIP_LIBRARY_FIND(hipDrvGetErrorName)
  HIP_LIBRARY_FIND(hipDrvGetErrorString)
  HIP_LIBRARY_FIND(hipStreamCreate)
  HIP_LIBRARY_FIND(hipStreamCreateWithFlags)
  HIP_LIBRARY_FIND(hipStreamCreateWithPriority)
  HIP_LIBRARY_FIND(hipDeviceGetStreamPriorityRange)
  HIP_LIBRARY_FIND(hipStreamDestroy)
  HIP_LIBRARY_FIND(hipStreamQuery)
  HIP_LIBRARY_FIND(hipStreamSynchronize)
  HIP_LIBRARY_FIND(hipStreamWaitEvent)
  HIP_LIBRARY_FIND(hipStreamGetFlags)
  HIP_LIBRARY_FIND(hipStreamGetPriority)
  HIP_LIBRARY_FIND(hipExtStreamCreateWithCUMask)
  HIP_LIBRARY_FIND(hipExtStreamGetCUMask)
  HIP_LIBRARY_FIND(hipStreamAddCallback)
  HIP_LIBRARY_FIND(hipStreamWaitValue32)
  HIP_LIBRARY_FIND(hipStreamWaitValue64)
  HIP_LIBRARY_FIND(hipStreamWriteValue32)
  HIP_LIBRARY_FIND(hipStreamWriteValue64)
  HIP_LIBRARY_FIND(hipEventCreateWithFlags)
  HIP_LIBRARY_FIND(hipEventCreate)
  HIP_LIBRARY_FIND(hipEventRecord)
  HIP_LIBRARY_FIND(hipEventDestroy)
  HIP_LIBRARY_FIND(hipEventSynchronize)
  HIP_LIBRARY_FIND(hipEventElapsedTime)
  HIP_LIBRARY_FIND(hipEventQuery)
  HIP_LIBRARY_FIND(hipPointerSetAttribute)
  HIP_LIBRARY_FIND(hipPointerGetAttributes)
  HIP_LIBRARY_FIND(hipPointerGetAttribute)
  HIP_LIBRARY_FIND(hipDrvPointerGetAttributes)
  HIP_LIBRARY_FIND(hipImportExternalSemaphore)
  HIP_LIBRARY_FIND(hipSignalExternalSemaphoresAsync)
  HIP_LIBRARY_FIND(hipWaitExternalSemaphoresAsync)
  HIP_LIBRARY_FIND(hipDestroyExternalSemaphore)
  HIP_LIBRARY_FIND(hipImportExternalMemory)
  HIP_LIBRARY_FIND(hipExternalMemoryGetMappedBuffer)
  HIP_LIBRARY_FIND(hipDestroyExternalMemory)
  HIP_LIBRARY_FIND(hipMalloc)
  HIP_LIBRARY_FIND(hipExtMallocWithFlags)
  HIP_LIBRARY_FIND(hipMallocHost)
  HIP_LIBRARY_FIND(hipMemAllocHost)
  HIP_LIBRARY_FIND(hipHostMalloc)
  HIP_LIBRARY_FIND(hipMallocManaged)
  HIP_LIBRARY_FIND(hipMemPrefetchAsync)
  HIP_LIBRARY_FIND(hipMemAdvise)
  HIP_LIBRARY_FIND(hipMemRangeGetAttribute)
  HIP_LIBRARY_FIND(hipMemRangeGetAttributes)
  HIP_LIBRARY_FIND(hipStreamAttachMemAsync)
  HIP_LIBRARY_FIND(hipMallocAsync)
  HIP_LIBRARY_FIND(hipFreeAsync)
  HIP_LIBRARY_FIND(hipMemPoolTrimTo)
  HIP_LIBRARY_FIND(hipMemPoolSetAttribute)
  HIP_LIBRARY_FIND(hipMemPoolGetAttribute)
  HIP_LIBRARY_FIND(hipMemPoolSetAccess)
  HIP_LIBRARY_FIND(hipMemPoolGetAccess)
  HIP_LIBRARY_FIND(hipMemPoolCreate)
  HIP_LIBRARY_FIND(hipMemPoolDestroy)
  HIP_LIBRARY_FIND(hipMallocFromPoolAsync)
  HIP_LIBRARY_FIND(hipMemPoolExportToShareableHandle)
  HIP_LIBRARY_FIND(hipMemPoolImportFromShareableHandle)
  HIP_LIBRARY_FIND(hipMemPoolExportPointer)
  HIP_LIBRARY_FIND(hipMemPoolImportPointer)
  HIP_LIBRARY_FIND(hipHostAlloc)
  HIP_LIBRARY_FIND(hipHostGetDevicePointer)
  HIP_LIBRARY_FIND(hipHostGetFlags)
  HIP_LIBRARY_FIND(hipHostRegister)
  HIP_LIBRARY_FIND(hipHostUnregister)
  HIP_LIBRARY_FIND(hipMallocPitch)
  HIP_LIBRARY_FIND(hipMemAllocPitch)
  HIP_LIBRARY_FIND(hipFree)
  HIP_LIBRARY_FIND(hipFreeHost)
  HIP_LIBRARY_FIND(hipHostFree)
  HIP_LIBRARY_FIND(hipMemcpy)
  HIP_LIBRARY_FIND(hipMemcpyWithStream)
  HIP_LIBRARY_FIND(hipMemcpyHtoD)
  HIP_LIBRARY_FIND(hipMemcpyDtoH)
  HIP_LIBRARY_FIND(hipMemcpyDtoD)
  HIP_LIBRARY_FIND(hipMemcpyHtoDAsync)
  HIP_LIBRARY_FIND(hipMemcpyDtoHAsync)
  HIP_LIBRARY_FIND(hipMemcpyDtoDAsync)
  HIP_LIBRARY_FIND(hipModuleGetGlobal)
  HIP_LIBRARY_FIND(hipGetSymbolAddress)
  HIP_LIBRARY_FIND(hipGetSymbolSize)
  HIP_LIBRARY_FIND(hipMemcpyToSymbol)
  HIP_LIBRARY_FIND(hipMemcpyToSymbolAsync)
  HIP_LIBRARY_FIND(hipMemcpyFromSymbol)
  HIP_LIBRARY_FIND(hipMemcpyFromSymbolAsync)
  HIP_LIBRARY_FIND(hipMemcpyAsync)
  HIP_LIBRARY_FIND(hipMemset)
  HIP_LIBRARY_FIND(hipMemsetD8)
  HIP_LIBRARY_FIND(hipMemsetD8Async)
  HIP_LIBRARY_FIND(hipMemsetD16)
  HIP_LIBRARY_FIND(hipMemsetD16Async)
  HIP_LIBRARY_FIND(hipMemsetD32)
  HIP_LIBRARY_FIND(hipMemsetAsync)
  HIP_LIBRARY_FIND(hipMemsetD32Async)
  HIP_LIBRARY_FIND(hipMemset2D)
  HIP_LIBRARY_FIND(hipMemset2DAsync)
  HIP_LIBRARY_FIND(hipMemset3D)
  HIP_LIBRARY_FIND(hipMemset3DAsync)
  HIP_LIBRARY_FIND(hipMemGetInfo)
  HIP_LIBRARY_FIND(hipMemPtrGetInfo)
  HIP_LIBRARY_FIND(hipMallocArray)
  HIP_LIBRARY_FIND(hipArrayCreate)
  HIP_LIBRARY_FIND(hipArrayDestroy)
  HIP_LIBRARY_FIND(hipArray3DCreate)
  HIP_LIBRARY_FIND(hipMalloc3D)
  HIP_LIBRARY_FIND(hipFreeArray)
  HIP_LIBRARY_FIND(hipFreeMipmappedArray)
  HIP_LIBRARY_FIND(hipMalloc3DArray)
  HIP_LIBRARY_FIND(hipMallocMipmappedArray)
  HIP_LIBRARY_FIND(hipGetMipmappedArrayLevel)
  HIP_LIBRARY_FIND(hipMemcpy2D)
  HIP_LIBRARY_FIND(hipMemcpyParam2D)
  HIP_LIBRARY_FIND(hipMemcpyParam2DAsync)
  HIP_LIBRARY_FIND(hipMemcpy2DAsync)
  HIP_LIBRARY_FIND(hipMemcpy2DToArray)
  HIP_LIBRARY_FIND(hipMemcpy2DToArrayAsync)
  HIP_LIBRARY_FIND(hipMemcpyToArray)
  HIP_LIBRARY_FIND(hipMemcpyFromArray)
  HIP_LIBRARY_FIND(hipMemcpy2DFromArray)
  HIP_LIBRARY_FIND(hipMemcpy2DFromArrayAsync)
  HIP_LIBRARY_FIND(hipMemcpyAtoH)
  HIP_LIBRARY_FIND(hipMemcpyHtoA)
  HIP_LIBRARY_FIND(hipMemcpy3D)
  HIP_LIBRARY_FIND(hipMemcpy3DAsync)
  HIP_LIBRARY_FIND(hipDrvMemcpy3D)
  HIP_LIBRARY_FIND(hipDrvMemcpy3DAsync)
  HIP_LIBRARY_FIND(hipDeviceCanAccessPeer)
  HIP_LIBRARY_FIND(hipDeviceEnablePeerAccess)
  HIP_LIBRARY_FIND(hipDeviceDisablePeerAccess)
  HIP_LIBRARY_FIND(hipMemGetAddressRange)
  HIP_LIBRARY_FIND(hipMemcpyPeer)
  HIP_LIBRARY_FIND(hipMemcpyPeerAsync)
  HIP_LIBRARY_FIND(hipCtxCreate)
  HIP_LIBRARY_FIND(hipCtxDestroy)
  HIP_LIBRARY_FIND(hipCtxPopCurrent)
  HIP_LIBRARY_FIND(hipCtxPushCurrent)
  HIP_LIBRARY_FIND(hipCtxSetCurrent)
  HIP_LIBRARY_FIND(hipCtxGetCurrent)
  HIP_LIBRARY_FIND(hipCtxGetDevice)
  HIP_LIBRARY_FIND(hipCtxGetApiVersion)
  HIP_LIBRARY_FIND(hipCtxGetCacheConfig)
  HIP_LIBRARY_FIND(hipCtxSetCacheConfig)
  HIP_LIBRARY_FIND(hipCtxSetSharedMemConfig)
  HIP_LIBRARY_FIND(hipCtxGetSharedMemConfig)
  HIP_LIBRARY_FIND(hipCtxSynchronize)
  HIP_LIBRARY_FIND(hipCtxGetFlags)
  HIP_LIBRARY_FIND(hipCtxEnablePeerAccess)
  HIP_LIBRARY_FIND(hipCtxDisablePeerAccess)
  HIP_LIBRARY_FIND(hipDevicePrimaryCtxGetState)
  HIP_LIBRARY_FIND(hipDevicePrimaryCtxRelease)
  HIP_LIBRARY_FIND(hipDevicePrimaryCtxRetain)
  HIP_LIBRARY_FIND(hipDevicePrimaryCtxReset)
  HIP_LIBRARY_FIND(hipDevicePrimaryCtxSetFlags)
  HIP_LIBRARY_FIND(hipModuleLoad)
  HIP_LIBRARY_FIND(hipModuleUnload)
  HIP_LIBRARY_FIND(hipModuleGetFunction)
  HIP_LIBRARY_FIND(hipFuncGetAttributes)
  HIP_LIBRARY_FIND(hipFuncGetAttribute)
  HIP_LIBRARY_FIND(hipModuleGetTexRef)
  HIP_LIBRARY_FIND(hipModuleLoadData)
  HIP_LIBRARY_FIND(hipModuleLoadDataEx)
  HIP_LIBRARY_FIND(hipModuleLaunchKernel)
  HIP_LIBRARY_FIND(hipModuleLaunchCooperativeKernel)
  HIP_LIBRARY_FIND(hipModuleLaunchCooperativeKernelMultiDevice)
  HIP_LIBRARY_FIND(hipLaunchCooperativeKernel)
  HIP_LIBRARY_FIND(hipLaunchCooperativeKernelMultiDevice)
  HIP_LIBRARY_FIND(hipExtLaunchMultiKernelMultiDevice)
  HIP_LIBRARY_FIND(hipModuleOccupancyMaxPotentialBlockSize)
  HIP_LIBRARY_FIND(hipModuleOccupancyMaxPotentialBlockSizeWithFlags)
  HIP_LIBRARY_FIND(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)
  HIP_LIBRARY_FIND(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
  HIP_LIBRARY_FIND(hipOccupancyMaxActiveBlocksPerMultiprocessor)
  HIP_LIBRARY_FIND(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
  HIP_LIBRARY_FIND(hipOccupancyMaxPotentialBlockSize)
  HIP_LIBRARY_FIND(hipProfilerStart)
  HIP_LIBRARY_FIND(hipProfilerStop)
  HIP_LIBRARY_FIND(hipConfigureCall)
  HIP_LIBRARY_FIND(hipSetupArgument)
  HIP_LIBRARY_FIND(hipLaunchByPtr)
  HIP_LIBRARY_FIND(hipLaunchKernel)
  HIP_LIBRARY_FIND(hipLaunchHostFunc)
  HIP_LIBRARY_FIND(hipDrvMemcpy2DUnaligned)
  HIP_LIBRARY_FIND(hipExtLaunchKernel)
  HIP_LIBRARY_FIND(hipBindTextureToMipmappedArray)
  HIP_LIBRARY_FIND(hipCreateTextureObject)
  HIP_LIBRARY_FIND(hipDestroyTextureObject)
  HIP_LIBRARY_FIND(hipGetChannelDesc)
  HIP_LIBRARY_FIND(hipGetTextureObjectResourceDesc)
  HIP_LIBRARY_FIND(hipGetTextureObjectResourceViewDesc)
  HIP_LIBRARY_FIND(hipGetTextureObjectTextureDesc)
  HIP_LIBRARY_FIND(hipTexObjectCreate)
  HIP_LIBRARY_FIND(hipTexObjectDestroy)
  HIP_LIBRARY_FIND(hipTexObjectGetResourceDesc)
  HIP_LIBRARY_FIND(hipTexObjectGetResourceViewDesc)
  HIP_LIBRARY_FIND(hipTexObjectGetTextureDesc)
  HIP_LIBRARY_FIND(hipGetTextureReference)
  HIP_LIBRARY_FIND(hipTexRefSetAddressMode)
  HIP_LIBRARY_FIND(hipTexRefSetArray)
  HIP_LIBRARY_FIND(hipTexRefSetFilterMode)
  HIP_LIBRARY_FIND(hipTexRefSetFlags)
  HIP_LIBRARY_FIND(hipTexRefSetFormat)
  HIP_LIBRARY_FIND(hipBindTexture)
  HIP_LIBRARY_FIND(hipBindTexture2D)
  HIP_LIBRARY_FIND(hipBindTextureToArray)
  HIP_LIBRARY_FIND(hipGetTextureAlignmentOffset)
  HIP_LIBRARY_FIND(hipUnbindTexture)
  HIP_LIBRARY_FIND(hipTexRefGetAddress)
  HIP_LIBRARY_FIND(hipTexRefGetAddressMode)
  HIP_LIBRARY_FIND(hipTexRefGetFilterMode)
  HIP_LIBRARY_FIND(hipTexRefGetFlags)
  HIP_LIBRARY_FIND(hipTexRefGetFormat)
  HIP_LIBRARY_FIND(hipTexRefGetMaxAnisotropy)
  HIP_LIBRARY_FIND(hipTexRefGetMipmapFilterMode)
  HIP_LIBRARY_FIND(hipTexRefGetMipmapLevelBias)
  HIP_LIBRARY_FIND(hipTexRefGetMipmapLevelClamp)
  HIP_LIBRARY_FIND(hipTexRefGetMipMappedArray)
  HIP_LIBRARY_FIND(hipTexRefSetAddress)
  HIP_LIBRARY_FIND(hipTexRefSetAddress2D)
  HIP_LIBRARY_FIND(hipTexRefSetMaxAnisotropy)
  HIP_LIBRARY_FIND(hipTexRefSetBorderColor)
  HIP_LIBRARY_FIND(hipTexRefSetMipmapFilterMode)
  HIP_LIBRARY_FIND(hipTexRefSetMipmapLevelBias)
  HIP_LIBRARY_FIND(hipTexRefSetMipmapLevelClamp)
  HIP_LIBRARY_FIND(hipTexRefSetMipmappedArray)
  HIP_LIBRARY_FIND(hipMipmappedArrayCreate)
  HIP_LIBRARY_FIND(hipMipmappedArrayDestroy)
  HIP_LIBRARY_FIND(hipMipmappedArrayGetLevel)
  HIP_LIBRARY_FIND(hipApiName)
  HIP_LIBRARY_FIND(hipKernelNameRef)
  HIP_LIBRARY_FIND(hipKernelNameRefByPtr)
  HIP_LIBRARY_FIND(hipGetStreamDeviceId)
  HIP_LIBRARY_FIND(hipStreamBeginCapture)
  HIP_LIBRARY_FIND(hipStreamEndCapture)
  HIP_LIBRARY_FIND(hipStreamGetCaptureInfo)
  HIP_LIBRARY_FIND(hipStreamGetCaptureInfo_v2)
  HIP_LIBRARY_FIND(hipStreamIsCapturing)
  HIP_LIBRARY_FIND(hipStreamUpdateCaptureDependencies)
  HIP_LIBRARY_FIND(hipThreadExchangeStreamCaptureMode)
  HIP_LIBRARY_FIND(hipGraphCreate)
  HIP_LIBRARY_FIND(hipGraphDestroy)
  HIP_LIBRARY_FIND(hipGraphAddDependencies)
  HIP_LIBRARY_FIND(hipGraphRemoveDependencies)
  HIP_LIBRARY_FIND(hipGraphGetEdges)
  HIP_LIBRARY_FIND(hipGraphGetNodes)
  HIP_LIBRARY_FIND(hipGraphGetRootNodes)
  HIP_LIBRARY_FIND(hipGraphNodeGetDependencies)
  HIP_LIBRARY_FIND(hipGraphNodeGetDependentNodes)
  HIP_LIBRARY_FIND(hipGraphNodeGetType)
  HIP_LIBRARY_FIND(hipGraphDestroyNode)
  HIP_LIBRARY_FIND(hipGraphClone)
  HIP_LIBRARY_FIND(hipGraphNodeFindInClone)
  HIP_LIBRARY_FIND(hipGraphInstantiate)
  HIP_LIBRARY_FIND(hipGraphInstantiateWithFlags)
  HIP_LIBRARY_FIND(hipGraphLaunch)
  HIP_LIBRARY_FIND(hipGraphUpload)
  HIP_LIBRARY_FIND(hipGraphExecDestroy)
  HIP_LIBRARY_FIND(hipGraphExecUpdate)
  HIP_LIBRARY_FIND(hipGraphAddKernelNode)
  HIP_LIBRARY_FIND(hipGraphKernelNodeGetParams)
  HIP_LIBRARY_FIND(hipGraphKernelNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphExecKernelNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphAddMemcpyNode)
  HIP_LIBRARY_FIND(hipGraphMemcpyNodeGetParams)
  HIP_LIBRARY_FIND(hipGraphMemcpyNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphKernelNodeSetAttribute)
  HIP_LIBRARY_FIND(hipGraphKernelNodeGetAttribute)
  HIP_LIBRARY_FIND(hipGraphExecMemcpyNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphAddMemcpyNode1D)
  HIP_LIBRARY_FIND(hipGraphMemcpyNodeSetParams1D)
  HIP_LIBRARY_FIND(hipGraphExecMemcpyNodeSetParams1D)
  HIP_LIBRARY_FIND(hipGraphAddMemcpyNodeFromSymbol)
  HIP_LIBRARY_FIND(hipGraphMemcpyNodeSetParamsFromSymbol)
  HIP_LIBRARY_FIND(hipGraphExecMemcpyNodeSetParamsFromSymbol)
  HIP_LIBRARY_FIND(hipGraphAddMemcpyNodeToSymbol)
  HIP_LIBRARY_FIND(hipGraphMemcpyNodeSetParamsToSymbol)
  HIP_LIBRARY_FIND(hipGraphExecMemcpyNodeSetParamsToSymbol)
  HIP_LIBRARY_FIND(hipGraphAddMemsetNode)
  HIP_LIBRARY_FIND(hipGraphMemsetNodeGetParams)
  HIP_LIBRARY_FIND(hipGraphMemsetNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphExecMemsetNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphAddHostNode)
  HIP_LIBRARY_FIND(hipGraphHostNodeGetParams)
  HIP_LIBRARY_FIND(hipGraphHostNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphExecHostNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphAddChildGraphNode)
  HIP_LIBRARY_FIND(hipGraphChildGraphNodeGetGraph)
  HIP_LIBRARY_FIND(hipGraphExecChildGraphNodeSetParams)
  HIP_LIBRARY_FIND(hipGraphAddEmptyNode)
  HIP_LIBRARY_FIND(hipGraphAddEventRecordNode)
  HIP_LIBRARY_FIND(hipGraphEventRecordNodeGetEvent)
  HIP_LIBRARY_FIND(hipGraphEventRecordNodeSetEvent)
  HIP_LIBRARY_FIND(hipGraphExecEventRecordNodeSetEvent)
  HIP_LIBRARY_FIND(hipGraphAddEventWaitNode)
  HIP_LIBRARY_FIND(hipGraphEventWaitNodeGetEvent)
  HIP_LIBRARY_FIND(hipGraphEventWaitNodeSetEvent)
  HIP_LIBRARY_FIND(hipGraphExecEventWaitNodeSetEvent)
  HIP_LIBRARY_FIND(hipGraphAddMemAllocNode)
  HIP_LIBRARY_FIND(hipGraphMemAllocNodeGetParams)
  HIP_LIBRARY_FIND(hipGraphAddMemFreeNode)
  HIP_LIBRARY_FIND(hipGraphMemFreeNodeGetParams)
  HIP_LIBRARY_FIND(hipDeviceGetGraphMemAttribute)
  HIP_LIBRARY_FIND(hipDeviceSetGraphMemAttribute)
  HIP_LIBRARY_FIND(hipDeviceGraphMemTrim)
  HIP_LIBRARY_FIND(hipUserObjectCreate)
  HIP_LIBRARY_FIND(hipUserObjectRelease)
  HIP_LIBRARY_FIND(hipUserObjectRetain)
  HIP_LIBRARY_FIND(hipGraphRetainUserObject)
  HIP_LIBRARY_FIND(hipGraphReleaseUserObject)
  HIP_LIBRARY_FIND(hipGraphDebugDotPrint)
  HIP_LIBRARY_FIND(hipGraphKernelNodeCopyAttributes)
  HIP_LIBRARY_FIND(hipGraphNodeSetEnabled)
  HIP_LIBRARY_FIND(hipGraphNodeGetEnabled)
  HIP_LIBRARY_FIND(hipMemAddressFree)
  HIP_LIBRARY_FIND(hipMemAddressReserve)
  HIP_LIBRARY_FIND(hipMemCreate)
  HIP_LIBRARY_FIND(hipMemExportToShareableHandle)
  HIP_LIBRARY_FIND(hipMemGetAccess)
  HIP_LIBRARY_FIND(hipMemGetAllocationGranularity)
  HIP_LIBRARY_FIND(hipMemGetAllocationPropertiesFromHandle)
  HIP_LIBRARY_FIND(hipMemImportFromShareableHandle)
  HIP_LIBRARY_FIND(hipMemMap)
  HIP_LIBRARY_FIND(hipMemMapArrayAsync)
  HIP_LIBRARY_FIND(hipMemRelease)
  HIP_LIBRARY_FIND(hipMemRetainAllocationHandle)
  HIP_LIBRARY_FIND(hipMemSetAccess)
  HIP_LIBRARY_FIND(hipMemUnmap)
  HIP_LIBRARY_FIND(hipGLGetDevices)
  //HIP_LIBRARY_FIND(hipGraphicsGLRegisterBuffer)
  //HIP_LIBRARY_FIND(hipGraphicsGLRegisterImage)
  //HIP_LIBRARY_FIND(hipGraphicsMapResources)
  //HIP_LIBRARY_FIND(hipGraphicsSubResourceGetMappedArray)
  //HIP_LIBRARY_FIND(hipGraphicsResourceGetMappedPointer)
  //HIP_LIBRARY_FIND(hipGraphicsUnmapResources)
  //HIP_LIBRARY_FIND(hipGraphicsUnregisterResource)
  HIP_LIBRARY_FIND(hipMemcpy_spt)
  HIP_LIBRARY_FIND(hipMemcpyToSymbol_spt)
  HIP_LIBRARY_FIND(hipMemcpyFromSymbol_spt)
  HIP_LIBRARY_FIND(hipMemcpy2D_spt)
  HIP_LIBRARY_FIND(hipMemcpy2DFromArray_spt)
  HIP_LIBRARY_FIND(hipMemcpy3D_spt)
  HIP_LIBRARY_FIND(hipMemset_spt)
  HIP_LIBRARY_FIND(hipMemsetAsync_spt)
  HIP_LIBRARY_FIND(hipMemset2D_spt)
  HIP_LIBRARY_FIND(hipMemset2DAsync_spt)
  HIP_LIBRARY_FIND(hipMemset3DAsync_spt)
  HIP_LIBRARY_FIND(hipMemset3D_spt)
  HIP_LIBRARY_FIND(hipMemcpyAsync_spt)
  HIP_LIBRARY_FIND(hipMemcpy3DAsync_spt)
  HIP_LIBRARY_FIND(hipMemcpy2DAsync_spt)
  HIP_LIBRARY_FIND(hipMemcpyFromSymbolAsync_spt)
  HIP_LIBRARY_FIND(hipMemcpyToSymbolAsync_spt)
  HIP_LIBRARY_FIND(hipMemcpyFromArray_spt)
  HIP_LIBRARY_FIND(hipMemcpy2DToArray_spt)
  HIP_LIBRARY_FIND(hipMemcpy2DFromArrayAsync_spt)
  HIP_LIBRARY_FIND(hipMemcpy2DToArrayAsync_spt)
  HIP_LIBRARY_FIND(hipStreamQuery_spt)
  HIP_LIBRARY_FIND(hipStreamSynchronize_spt)
  HIP_LIBRARY_FIND(hipStreamGetPriority_spt)
  HIP_LIBRARY_FIND(hipStreamWaitEvent_spt)
  HIP_LIBRARY_FIND(hipStreamGetFlags_spt)
  HIP_LIBRARY_FIND(hipStreamAddCallback_spt)
  HIP_LIBRARY_FIND(hipEventRecord_spt)
  HIP_LIBRARY_FIND(hipLaunchCooperativeKernel_spt)
  HIP_LIBRARY_FIND(hipLaunchKernel_spt)
  HIP_LIBRARY_FIND(hipGraphLaunch_spt)
  HIP_LIBRARY_FIND(hipStreamBeginCapture_spt)
  HIP_LIBRARY_FIND(hipStreamEndCapture_spt)
  HIP_LIBRARY_FIND(hipStreamIsCapturing_spt)
  HIP_LIBRARY_FIND(hipStreamGetCaptureInfo_spt)
  HIP_LIBRARY_FIND(hipStreamGetCaptureInfo_v2_spt)
  HIP_LIBRARY_FIND(hipLaunchHostFunc_spt)
  result = 0; // success
  return result;
}
