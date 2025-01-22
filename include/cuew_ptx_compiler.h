// SPDX-License-Identifier: MIT
// Based on https://docs.nvidia.com/cuda/ptx-compiler-api/index.html
//
// PTX compiler API are not part of cuew(assume nvptxcompiler_static.a is statically linked to your app)

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nvPTXCompiler *nvPTXCompilerHandle;
typedef enum _nvPTXCompileResult {
    NVPTXCOMPILE_SUCCESS = 0,
    NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE,
    NVPTXCOMPILE_ERROR_INVALID_INPUT,
    NVPTXCOMPILE_ERROR_COMPILATION_FAILURE,
    NVPTXCOMPILE_ERROR_INTERNAL,
    NVPTXCOMPILE_ERROR_OUT_OF_MEMORY,
    NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE,
    NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION,
    NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC
} nvPTXCompileResult;


nvPTXCompileResult nvPTXCompilerGetVersion(unsigned int *major, unsigned int *minor);
nvPTXCompileResult nvPTXCompilerCompile(nvPTXCompilerHandle compiler, int numCompileOptions, const char *const *compileOptions);
nvPTXCompileResult nvPTXCompilerCreate(nvPTXCompilerHandle *compiler, size_t ptxCodeLen, const char *ptxCode);
nvPTXCompileResult nvPTXCompilerDestroy(nvPTXCompilerHandle *compiler);
nvPTXCompileResult nvPTXCompilerGetCompiledProgram(nvPTXCompilerHandle compiler, void *binaryImage);
nvPTXCompileResult nvPTXCompilerGetCompiledProgramSize(nvPTXCompilerHandle compiler, size_t *binaryImageSize);
nvPTXCompileResult nvPTXCompilerGetErrorLog(nvPTXCompilerHandle compiler, char *errorLog);
nvPTXCompileResult nvPTXCompilerGetErrorLogSize(nvPTXCompilerHandle compiler, size_t *errorLogSize);
nvPTXCompileResult nvPTXCompilerGetInfoLog(nvPTXCompilerHandle compiler, char *infoLog);
nvPTXCompileResult nvPTXCompilerGetInfoLogSize(nvPTXCompilerHandle compiler, size_t *infoLogSize);


#ifdef __cplusplus
}
#endif

