# HTM

HTM is a Halide language To MLIR compiler which converts the applications written in [Halide](https://github.com/halide/Halide)
to [MLIR](https://mlir.llvm.org).  
This project targets acquiring performance boost on Halide through the conversion to MLIR. With this conversion, we can benefit from MLIR with its main
property, namely "Dialect", which is able to performing abstraction on operations(instructions) at different level of granularities, thus the
data is able to be well-structurized on every stages during operation(instruction) lowering as well as transformation passes to obtain more opportunities on
potential optimizations and conversions on the application.

# How to run HTM?

To build HTM, [MLIRX](https://github.com/polymage-labs/mlirx) and [HalideX](https://github.com/songwu0813/HalideX) are needed. We've provided a script to
automatically build the entire project and all of dependencies that needed.

Prerequisite:
- Ubuntu 16.04 or Ubuntu 18.04
- [CMake](https://cmake.org) version 3.14 or later
- [libpng](http://www.libpng.org/pub/png/libpng.html)
- [libjpeg](https://www.libjpeg-turbo.org)


Please simply clone this project and run the following commands step by step:  
(Note that it may takes an hour or half to build, please be patient.)
```
$ git clone https://github.com/songwu0813/HTM.git
$ cd HTM
$ . scripts/setup.sh
$ ./scripts/build_project.sh
```

After building processes finished, you can run the apps in [Halide](https://github.com/halide/Halide), this project already provided some samples,
to build the app, please run the following commands:
```
$ cd apps/<app name>
$ mkdir build && cd build
$ cmake ..
$ cp ../*.png .
$ ./runme
```

The corresponding MLIR file(runme.mlir or something) then will be generated, developer can either take this file for further optimization using MLIR or
execute directly through the interpreter provided by [LLVM](https://github.com/llvm/llvm-project), we also provide the script for doing this.
Run the following command under same build directory:
```
$ ../../../scripts/run_mlir.sh <file name>.mlir
```
