#!/bin/bash

OPT=$LLVM_ROOT/bin/mlir-opt
CPU_RUNNER=$LLVM_ROOT/bin/mlir-cpu-runner
OPT_OPTIONS="-convert-linalg-to-loops -lower-affine -convert-vector-to-llvm -convert-scf-to-std -std-expand -convert-std-to-llvm"
CPU_RUNNER_OPTIONS="-O3 -e main -reps=1 -entry-point-result=void -shared-libs=$LLVM_ROOT/lib/libmlir_runner_utils.so,$LLVM_ROOT/lib/libmlir_c_runner_utils.so,$PROJECT_ROOT/build/libCPURunnerUtils.so"
$OPT $OPT_OPTIONS $1 | $CPU_RUNNER $CPU_RUNNER_OPTIONS
