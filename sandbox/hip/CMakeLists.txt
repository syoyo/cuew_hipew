cmake_minimum_required (VERSION 3.1)

project(hipew CXX C)

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

add_library(hipew hipew.c)

add_executable(testhipew test_hipew.cc)
target_link_libraries(testhipew hipew ${CMAKE_DL_LIBS})

# [VisualStudio]
if(WIN32)
  # Set testhipew as a startup project for VS IDE
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT testhipew)

  # For easier debugging in VS IDE(cmake 3.8.0 or later required) Set working
  # directory to hipew git repo root.
  if(CMAKE_VERSION VERSION_GREATER 3.8.0)
    set_target_properties(testhipew
                          PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
                                     "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endif()
