clang -I/opt/rocm/include -Xclang -ast-dump=json -fsyntax-only hip_astgen.c > hip.json
