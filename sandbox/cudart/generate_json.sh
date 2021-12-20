#clang -I/usr/local/cuda/include -Xclang -ast-dump=json -fsyntax-only cudart_astgen.c > cudart.json
#clang -I/usr/local/cuda/include -Xclang -ast-dump=json -fsyntax-only cublas_astgen.c > cublas.json
#clang -I/usr/local/cuda/include -Xclang -ast-dump=json -fsyntax-only cufft_astgen.c > cufft.json
clang -I/usr/local/cuda/include -Xclang -ast-dump=json -fsyntax-only curand_astgen.c > curand.json
