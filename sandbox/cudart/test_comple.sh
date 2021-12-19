#clang -Weverything -Wall test.c cudart_lib.c -ldl
#gcc test.c cudart_lib.c -ldl

clang -Weverything -Wall -Icuda_local test_cuffw.c cuew_cufft.c -ldl
