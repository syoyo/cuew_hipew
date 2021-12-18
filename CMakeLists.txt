project(cuew)

cmake_minimum_required (VERSION 2.6)
if (NOT CMAKE_VERSION VERSION_LESS 2.8.4)
    cmake_policy (SET CMP0017 NEW)
endif()

include(CMakeParseArguments)

find_package(Threads REQUIRED)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
message(STATUS "Project source dir = ${PROJECT_SOURCE_DIR}")
message(STATUS "Project build dir = ${CMAKE_BINARY_DIR}")

if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

include_directories(include)

add_library(cuew src/cuew.c include/cuew.h)

add_executable(testcuew cuewTest/cuewTest.c include/cuew.h)
target_link_libraries(testcuew cuew ${CMAKE_DL_LIBS})

# [VisualStudio]
if(WIN32)
  # Set testcuew as a startup project for VS IDE
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT testcuew)

  # For easier debugging in VS IDE(cmake 3.8.0 or later required) Set working
  # directory to cuew git repo root.
  if(CMAKE_VERSION VERSION_GREATER 3.8.0)
    set_target_properties(testcuew
                          PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
                                     "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endif()


# ptxcompiler
#include_directories(/usr/local/cuda-11.1/include)
#add_executable(testptxcompiler cuewTest/ptxCompilerTest.c include/cuew.h)
#target_link_libraries(testptxcompiler cuew ${CMAKE_DL_LIBS})
#target_link_libraries(testptxcompiler cuew /usr/local/cuda-11.1/targets/x86_64-linux/lib/libnvptxcompiler_static.a Threads::Threads m)
