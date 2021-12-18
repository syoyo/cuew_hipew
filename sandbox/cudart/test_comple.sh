#clang -Weverything -Wall test.c cudart_lib.c -ldl
#gcc test.c cudart_lib.c -ldl

clang -Weverything -Wall test.c cufft_lib.c -ldl
