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
