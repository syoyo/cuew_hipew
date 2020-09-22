syoyo's modification to CUEW.

Currently tested on Ubuntu 18.04 x86-64 and Windows 10 64bit.

## Changes compared to original version

* Support CUDA 11.0
* Support CUDNN 8.0.3
* Disable cudaGL

## Supported API

* [x] cuda.h
* [x] cudnn.h
* [x] nvrtc.h
* [ ] cudaGL.h

## Generate cuew(for developer)

You need python3.
Install pycparser.

```
$ python -m pip install pycparser
```

```
$ cd auto
# Edit header path in cuew_ge.py if required
$ ./cuew_gen.sh
```

## Buld tests

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Known issues

* Combining with Address Sanitizer(`-fsanitizer=address`) won't work
  * calling CUDA API results in undefined behavior or seg faults
  * https://github.com/google/sanitizers/issues/629
* CUEW does not report warning when using deprecated CUDA API

## TODO

* [ ] Test on MSVC.
  * [x] `clang-cl` wors
* [ ] Support cudaGL.h
* [ ] Test cuDNN API call.
* [ ] Find a way to co-exist with Address Sanitizer

=================

The CUDA Extension Wrangler Library (CUEW) is a cross-platform open-source
C/C++ extension loading library. CUEW provides efficient run-time mechanisms
for determining which CUDA functions and extensions extensions are supported
on the target platform.

CUDA core and extension functionality is exposed in a single header file.
CUEW has been tested on a variety of operating systems, including Windows,
Linux, Mac OS X.

LICENSE

CUEW library is released under the Apache 2.0 license.

