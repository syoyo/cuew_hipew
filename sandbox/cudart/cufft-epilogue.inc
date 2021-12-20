#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#endif

//
// Define is not handled gen.py
//
#define MAX_CUFFT_ERROR 0x11

// CUFFT transform directions
#define CUFFT_FORWARD -1 // Forward FFT
#define CUFFT_INVERSE  1 // Inverse FFT


#include "library_types.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif
