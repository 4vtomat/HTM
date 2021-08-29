cmake_minimum_required(VERSION 3.14)

project(iir_blur)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE OFF CACHE BOOL "OFF" FORCE)

include(../../dependencies.cmake)
link_directories(../../build)

add_executable(runme
  main.cpp
  ${DEP_FILES}
)

target_link_libraries(runme ${DEP_LIBS})
