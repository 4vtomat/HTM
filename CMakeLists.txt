cmake_minimum_required(VERSION 3.14)

project(halide_mlir)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE OFF CACHE BOOL "OFF" FORCE)

include(dependencies.cmake)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)
add_library(Utils
  SHARED
  utils/MainFuncGenerator.cpp
  utils/ConvertFromLLVMIR.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.posix_allocator_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.posix_error_handler_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.posix_print_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.linux_clock_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.posix_io_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.linux_host_cpu_count_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.linux_yield_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.posix_threads_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.posix_get_symbol_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.halide_buffer_t_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.destructors_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.pseudostack_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.posix_math_ll.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.to_string_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.x86_ll.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.can_use_target_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.x86_cpu_features_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.module_aot_ref_count_64.cpp
  ${HALIDE_BUILD_DIR}/bin/build/initmod.runtime_api_64.cpp
  )
add_library(CPURunnerUtils
  SHARED
  utils/CPURunnerUtils.cpp
  )
add_library(MainLib
  SHARED
  AffineHelper.cpp
  Simplifier.cpp
)
