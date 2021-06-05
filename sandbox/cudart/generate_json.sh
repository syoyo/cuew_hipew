clang -I/usr/local/cuda/include -Xclang -ast-dump=json -fsyntax-only cudart.c > cudart.json
clang -I/usr/local/cuda/include -Xclang -ast-dump=json -fsyntax-only cublas.c > cublas.json
