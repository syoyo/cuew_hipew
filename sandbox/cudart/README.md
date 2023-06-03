Dump CUDA C/C++ header AST to JSON using clang.
Parse JSON and emit stub loader file in python.

## Requirements

* Python 3.8+

## CUDA version

- 12.1

## Dump ast of CUDA header file.

Run `generate_json.sh`.

## Generate stub header and loader

Edit `generate_stub.sh`, then

Run `generate_stub.sh`

## Config JSON

Config JSON is provided manually. The content will be changed over CUDA versions.

Some of important variables.

* "api\_prefix": API symbol prefix(e.g. `cufft` for cuFFT)
* "lib\_prefix": Specify the library prefix(e.g. `cufft` for cuFFT)
* "dllpaths": List of dll(`.so`) search path. Key "win32" and "linux" are required.
* "allowedSymbols" : This list describes allowed symbol names(struct, union, etc) which does not starts with "cuda" prefix(e.g. textureReference).

## TODO

* [x] cudart CUDA runtime API
* [ ] cuBlas API
  * [x] cuBLAS
  * [ ] cuBLASXt
  * [ ] cuBLASLt
  * [ ] NVBLAS
* [ ] cuSOLVER API
  * [x] cuSOLVER
  * [ ] cuSolverMg
* [ ] cuSPARSE API
  * [ ] cuSPARSE(W.I.P.)
  * [ ] cuSPARSELt
* [x] cuRAND API
* [ ] cuFFT API
  * [x] cuFFT
  * [ ] cuFFT Xt
  * [ ] cuFFTW
* [ ] nvJPEG API(W.I.P)
* [ ] NPP API
* [ ] nvenc API


EoL.
