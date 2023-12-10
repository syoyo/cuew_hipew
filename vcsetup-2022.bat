rmdir /q /s build
mkdir build

cmake.exe -G "Visual Studio 17 2022" -Ax64 -Bbuild -H.
