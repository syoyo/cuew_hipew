#clang -Weverything -Wall -Icuda_local -I../../include test.c cuew_cudart.c -ldl
#gcc test.c cudart_lib.c -ldl

clang -Weverything -Wall -Icuda_local -I../../include test_cuffw.c cuew_cufft.c -ldl
