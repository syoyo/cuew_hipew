#clang -Weverything -Wall -std=c11 -o test_cudart -Icuda_local -I../../include test_cudart.c cuew_cudart.c -ldl

#clang -o test_cufft -std=c11 -Weverything -Wall -Icuda_local -I../../include test_cufft.c cuew_cufft.c cuew_cudart.c -ldl
clang -o test_curand -std=c11 -Weverything -Wall -Icuda_local -I../../include test_curand.c cuew_curand.c cuew_cudart.c -ldl
