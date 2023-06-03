clang -Weverything -Wall -std=c11 -o test_cudart -I../../include/cuda_local -I../../include test_cudart.c cuew_cudart.c -ldl
clang -Weverything -Wall -std=c11 -o test_cublas -I../../include/cuda_local -I../../include test_cublas.c cuew_cublas.c cuew_cudart.c -ldl

clang -o test_cufft -std=c11 -Weverything -Wall -I../../include/cuda_local -I../../include test_cufft.c cuew_cufft.c cuew_cudart.c -ldl
clang -o test_curand -std=c11 -Weverything -Wall -I../../include/cuda_local -I../../include test_curand.c cuew_curand.c cuew_cudart.c -ldl
#clang -o test_nvjpeg -std=c11 -Weverything -Wall -I../../include/cuda_local -I../../include test_nvjpeg.c cuew_nvjpeg.c cuew_cudart.c -ldl
clang -o test_cusparse -std=c11 -Weverything -Wall -I../../include/cuda_local -I../../include test_cusparse.c cuew_cusparse.c cuew_cudart.c -ldl

clang -o test_cusolver -std=c11 -Weverything -Wall -I../../include/cuda_local -I../../include test_cusolver.c cuew_cusolver.c cuew_cudart.c -ldl
