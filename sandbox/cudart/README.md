Dump ast to JSON using clang.
Parse JSON and emit stub file in python.

## Requirements

* Python 3.9+

## Dump ast

Run `generate_json.sh`.

## Generate stub header and .c

Edit `Makefile`, then:

```
$ make
```


## allowlist

allowlist JSON is provided manually.
This list describes allowed symbol names(struct, union, etc) which does not starts with "cuda" prefix(e.g. textureReference).

## TODO

* [x] enum
* [x] CUDA runtime API
* [ ] cuBlas API
  * [ ] NVBLAS API?
* [ ] cuSOLVER API
* [ ] cuSPARSE API
* [ ] cuRAND API
* [ ] cuFFT API
  * [ ] cuFFT
  * [ ] cuFFT Xt
  * [ ] cuFFTW
* [ ] nvJPEG API(depends on CUDA runtime API)
* [ ] NPP API(depends on CUDA runtime API)



EoL.
