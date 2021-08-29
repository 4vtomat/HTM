if(NOT DEFINED ENV{LLVM_ROOT})
  message(FATAL_ERROR "please set LLVM_ROOT to the root of llvm build folder (e.g. export LLVM_ROOT=<path-to-project>/_deps/mlirx/build")
endif()
if(NOT DEFINED ENV{HALIDE_SRC_DIR})
  message(FATAL_ERROR "please set HALIDE_SRC_DIR to the root of HalideX source folder (e.g. export HALIDE_SRC_DIR=<path-to-project>/_deps/HalideX")
endif()
if(NOT DEFINED ENV{HALIDE_BUILD_DIR})
  message(FATAL_ERROR "please set HALIDE_SRC_DIR to the root of HalideX build folder (e.g. export HALIDE_BUILD_DIR=<path-to-project>/_deps/HalideX/build")
endif()

set(LLVM_DIR $ENV{LLVM_ROOT}/lib/cmake/llvm)
set(MLIR_DIR $ENV{LLVM_ROOT}/lib/cmake/mlir)
find_package(MLIR REQUIRED CONFIG)

set(HALIDE_SRC_DIR $ENV{HALIDE_SRC_DIR})
set(HALIDE_BUILD_DIR $ENV{HALIDE_BUILD_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${HALIDE_SRC_DIR})
include_directories(${HALIDE_SRC_DIR}/include/)
include_directories(${HALIDE_BUILD_DIR}/include/)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(${HALIDE_BUILD_DIR}/bin)

add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_MODULE_PATH
  ${LLVM_CMAKE_DIR}
  ${MLIR_CMAKE_DIR}
)

include(AddLLVM)
include(TableGen)
include(AddMLIR)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(DEP_FILES ${CMAKE_CURRENT_LIST_DIR}/CodeGen_MLIR.cpp)
set(DEP_LIBS
  MainLib
  Utils
  ${dialect_libs}
  ${conversion_libs}
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRTransforms
  MLIRTransformUtils
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRTargetLLVMIRImport
  MLIRSupport
  MLIRIR
  MLIROptLib
  LLVMSupport
  LLVMCore
  LLVMAsmParser
  Halide
  jpeg
  png
  pthread
  dl
  curses
)
