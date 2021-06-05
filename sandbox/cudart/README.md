Dump ast to JSON using clang.
Parse JSON and emit stub file in python.

## Requirements

* [x] Linux
* [ ] Windows

## Dump ast

Run `generate_json.sh`.

## Generate stub header and .c

Edit `Makefile`, then:

```
$ make
```


## TODO

* [x] enum
* [ ] CUDA runtime API
* [ ] cuBlas API
  * [ ] NVBLAS API?
* [ ] cuSOLVER API
* [ ] cuSPARSE API
* [ ] cuRAND API
* [ ] cuFFT API
* [ ] nvJPEG API(depends on CUDA runtime API)
* [ ] NPP API(depends on CUDA runtime API)



EoL.
