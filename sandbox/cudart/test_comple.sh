clang -Weverything -Wall -o test_cudart -Icuda_local -I../../include test_cudart.c cuew_cudart.c -ldl
#gcc test.c cudart_lib.c -ldl

clang -o test_cufft -Weverything -Wall -Icuda_local -I../../include test_cufft.c cuew_cufft.c cuew_cudart.c -ldl
