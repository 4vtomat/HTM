#!/bin/bash
if [ -z "${PROJECT_ROOT}" ]; then
  echo "Please run the command \". script/setup.sh\" first to setup the environment variables."
  exit
fi

if [ ! -d "_deps" ]; then
  mkdir _deps
fi
cd _deps

if [ ! -z "${CMAKE_ROOT}" ]; then
  CMAKE=${CMAKE_ROOT}/bin/cmake
else
  CMAKE=cmake
fi

if [ ! -d "mlirx" ]; then
  git clone https://github.com/polymage-labs/mlirx.git
else
  echo "MLIRX found, skipping..."
fi

if [ ! -d "mlirx/build" ]; then
  cd mlirx && git checkout beb7ccf && mkdir build && cd build
  $CMAKE -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra;mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Mips;Hexagon;WebAssembly;AMDGPU" \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_BUILD_32_BITS=OFF
  echo "Start building llvm & mlir projects, it might take about half an hour..."
  $CMAKE --build .
  if [ ! $? -eq 0 ]; then
    exit
  fi
  cd ../../
fi

if [ ! -d "HalideX" ]; then
  git clone https://github.com/songwu0813/HalideX.git
else
  echo "HalideX found, skipping..."
fi

if [ ! -d "HalideX/build" ]; then
  cd HalideX && mkdir build && cd build
  make -f ../Makefile USE_MLIRX=1 MULTIDIM_BUFFER=1 -j8
  if [ ! $? -eq 0 ]; then
    exit
  fi
  cd ../../
fi

cd ..

rm -rf build
mkdir build && cd build
$CMAKE -DMLIR_DIR=$LLVM_ROOT/lib/cmake/mlir -DLLVM_DIR=$LLVM_ROOT/lib/cmake/llvm ..
$CMAKE --build .
