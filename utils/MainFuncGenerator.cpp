#include "MainFuncGenerator.h"

namespace Utils {
MainFuncGenerator::MainFuncGenerator(Halide::Internal::CodeGen_MLIR& cg)
    : cg_mlir_(cg), context_(cg_mlir_.context_), builder_(cg_mlir_.builder_), module_(cg_mlir_.module_) {
}

void MainFuncGenerator::ProcessInputBuffers() noexcept {
  for (auto& arg : arguments_) {
    if (arg.is_input) {
      if (arg.base_val->is(ArgType::kBuffer)) {
        auto buffer = ((BufferArg*)arg.base_val)->value;
        auto& shape = ((BufferArg*)arg.base_val)->shape;
        mlir::Type buffer_elem_type = cg_mlir_.ConvertHalideTypeTToMlirType(buffer->type);
        if (!shape.size()) {
          for (int i = buffer->dimensions - 1; i >= 0; --i) {
            shape.push_back(buffer->dim[i].extent);
          }
        }
        size_t total_size = 1;
        for (auto s : shape) {
          total_size *= s;
        }
        mlir::RankedTensorType buf_type = mlir::RankedTensorType::get(shape, buffer_elem_type);
        std::vector<char> buf_data((char*)buffer->host, (char*)buffer->host + total_size * (buffer->type.bits / 8));
        mlir::Attribute buf_attr = mlir::DenseElementsAttr::getFromRawBuffer(buf_type, llvm::ArrayRef<char>(buf_data), false);

        ((BufferArg*)arg.base_val)->global_memref = CreateGlobalBuffer(buf_type, std::move(buf_attr));
      }
    }
  }
}


void MainFuncGenerator::DeclareUtilities() noexcept {
  mlir::FuncOp func;
  mlir::FunctionType func_type;
  mlir::StringAttr visability = builder_.getStringAttr("private");

  func_type = builder_.getFunctionType({builder_.getF64Type()}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "print_flops", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({builder_.getF64Type()}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "print_time_elapsed", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({builder_.getIndexType()}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printIndex", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({builder_.getI64Type()}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printI64", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({builder_.getI32Type()}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printI32", func_type, visability);
  module_.push_back(func);
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printIU32", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({builder_.getIntegerType(16)}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printI16", func_type, visability);
  module_.push_back(func);
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printIU16", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({builder_.getIntegerType(8)}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printI8", func_type, visability);
  module_.push_back(func);
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printIU8", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({builder_.getF64Type()}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printF64", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({builder_.getF32Type()}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printF32", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({}, {builder_.getF64Type()});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "rtclock", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printOpen", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printClose", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printSpace", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printComma", func_type, visability);
  module_.push_back(func);

  func_type = builder_.getFunctionType({}, {});
  func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "printNewline", func_type, visability);
  module_.push_back(func);
}

std::string MainFuncGenerator::GenerateRandomName() noexcept {
  static size_t hash_value = 0x1234;
  static std::string name = "name_";
  srand(time(NULL));
  hash_value = (hash_value + std::hash<int>()(rand())) % (unsigned long)1e9;
  return name + std::to_string(hash_value);
}

mlir::memref::GlobalOp MainFuncGenerator::CreateGlobalBuffer(mlir::Type buf_type, mlir::Attribute&& buf_attr) noexcept {
  auto gen = this->GenerateRandomName();
  auto global_memref = builder_.create<mlir::memref::GlobalOp>(
                       builder_.getUnknownLoc(),
                       gen,
                       builder_.getStringAttr("private"),
                       mlir::BufferizeTypeConverter{}.convertType(buf_type),
                       std::move(buf_attr.cast<mlir::ElementsAttr>()),
                       true);
  module_.push_back(global_memref);
  return global_memref;
}

mlir::Type MainFuncGenerator::ConvertToMlirType(ArgumentBase& arg) noexcept {
  auto& type = arg.arg_type;
  switch (type) {
    case ArgType::kInt:
    {
      auto* base = (IntArg*)arg.base_val;
      return builder_.getIntegerType(base->arg_bits);
    }
    case ArgType::kUInt:
    {
      auto* base = (UIntArg*)arg.base_val;
      return builder_.getIntegerType(base->arg_bits);
    }
    case ArgType::kFloat:
    {
      auto* base = (FloatArg*)arg.base_val;
      if (base->arg_bits == 32) {
        return builder_.getF32Type();
      } else {
        return builder_.getF64Type();
      }
    }
    case ArgType::kBuffer:
    {
      auto* base = (BufferArg*)arg.base_val;
      auto elem_type = base->elem_type;
      if (elem_type == ArgType::kInt || elem_type == ArgType::kUInt) {
        return builder_.getIntegerType(base->elem_bits);
      } else if (elem_type == ArgType::kFloat) {
        if (base->elem_bits == 32) {
          return builder_.getF32Type();
        } else {
          return builder_.getF64Type();
        }
      } else {
        assert(0);
      }
    }
  }
}

void MainFuncGenerator::GenerateBody() noexcept {
  mlir::OpBuilder::InsertionGuard guard(builder_);

  auto func_type = builder_.getFunctionType({}, {});

  auto mlir_func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), "main", func_type);

  auto entry = mlir_func.addEntryBlock();
  builder_.setInsertionPointToStart(entry);

  llvm::SmallVector<mlir::Value, 1> args, args1;
  llvm::SmallVector<mlir::Type, 1> ret_type;
  mlir::Value start, end, elapsed;
  std::vector<mlir::Value> outputs;

  ret_type = {};
  args = {};
  mlir::Attribute attr;
  for (int i = 0; i < input_count; ++i) {
    mlir::Value input;
    auto arg = arguments_[i];
    auto arg_type = arg.arg_type;
    switch (arg_type) {
      case ArgType::kInt:
        attr = builder_.getIntegerAttr(builder_.getIntegerType(((IntArg*)arg.base_val)->arg_bits), ((IntArg*)arg.base_val)->value);
        input = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), attr);
        break;
      case ArgType::kUInt:
        attr = builder_.getIntegerAttr(builder_.getIntegerType(((UIntArg*)arg.base_val)->arg_bits), ((UIntArg*)arg.base_val)->value);
        input = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), attr);
        break;
      case ArgType::kFloat:
        if (((FloatArg*)arg.base_val)->arg_bits == 32) {
          attr = builder_.getFloatAttr(builder_.getF32Type(), ((FloatArg*)arg.base_val)->value);
        } else if (((FloatArg*)arg.base_val)->arg_bits == 64) {
          attr = builder_.getFloatAttr(builder_.getF64Type(), ((FloatArg*)arg.base_val)->value);
        } else {
          assert(false);
        }
        input = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), attr);
        break;
      case ArgType::kBuffer:
        auto global_memref = ((BufferArg*)arg.base_val)->global_memref;
        input = builder_.create<mlir::memref::GetGlobalOp>(builder_.getUnknownLoc(),
                                                           global_memref.type(),
                                                           global_memref.getName());
        break;
    }
    args.push_back(input);
  }


  for (int i = input_count; i < arguments_.size(); ++i) {
    mlir::Value output;
    auto arg = arguments_[i];
    auto elem_type = ConvertToMlirType(arg);
    std::vector<int64_t> shape;
    if (arg.arg_type == ArgType::kBuffer) {
      shape = ((BufferArg*)arg.base_val)->shape;
    } else {
      shape.push_back(1);
    }

    mlir::MemRefType output_type = mlir::MemRefType::get(shape, elem_type);
    mlir::IntegerAttr alignment = builder_.getIntegerAttr(builder_.getI64Type(), 8);
    output = builder_.create<mlir::memref::AllocOp>(builder_.getUnknownLoc(), output_type, alignment);
    mlir::Value value_to_fill;
    if (auto ty = output_type.getElementType().dyn_cast_or_null<mlir::FloatType>()) {
      value_to_fill = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getFloatAttr(ty, 0));
    } else if (auto ty = output_type.getElementType().dyn_cast_or_null<mlir::IntegerType>()) {
      value_to_fill = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getIntegerAttr(ty, 0));
    }
    builder_.create<mlir::linalg::FillOp>(builder_.getUnknownLoc(), output, value_to_fill);

    args.push_back(output);
    outputs.push_back(output);
  }


  ret_type = {builder_.getF64Type()};
  args1 = {};
  start = builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("rtclock"), ret_type, args1).getResult(0);

  ret_type = {};
  builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), cg_mlir_.func_names_[0], ret_type, args);

  ret_type = {builder_.getF64Type()};
  args1 = {};
  end = builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("rtclock"), ret_type, args1).getResult(0);

  elapsed = builder_.create<mlir::SubFOp>(builder_.getUnknownLoc(), builder_.getF64Type(), end, start);

  ret_type = {};
  args1 = {elapsed};
  builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("print_time_elapsed"), ret_type, args1);

  if (print_result) {
    GeneratePrintMemref(outputs);
  }

  builder_.create<mlir::ReturnOp>(builder_.getUnknownLoc());

  module_.push_back(mlir_func);
}

void MainFuncGenerator::GeneratePrintMemref(std::vector<mlir::Value>& memrefs) noexcept {
  for (int i = 0; i < memrefs.size(); ++i) {
    auto arg = arguments_[input_count + i];
    if (arg.base_val->is(ArgType::kBuffer)) {
      GeneratePrintMemref(memrefs[i], ((BufferArg*)arg.base_val)->elem_type);
    } else {
      GeneratePrintMemref(memrefs[i], arg.arg_type);
    }
  }
}

void MainFuncGenerator::GeneratePrintMemref(mlir::Value memref, ArgType type) noexcept {
  std::vector<mlir::Value> iters;
  std::vector<mlir::scf::ForOp> loops;
  mlir::MemRefType memref_type = memref.getType().cast<mlir::MemRefType>();
  auto shape = memref_type.getShape();
  int index = 0;
  llvm::SmallVector<mlir::Value, 1> args;
  llvm::SmallVector<mlir::Type, 1> ret_type;
  std::function<void ()> gen_loop_nest = [&]() {
    if (index >= shape.size()) {
      mlir::Value val = builder_.create<mlir::memref::LoadOp>(builder_.getUnknownLoc(), memref, iters);
      mlir::Type elem_type = memref_type.getElementType();
      ret_type = {};
      args = {val};
      if (elem_type.isF32()) {
        builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printF32"), ret_type, args);
      } else if (elem_type.isF64()) {
        builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printF64"), ret_type, args);
      } else if (elem_type.isInteger(32)) {
        if (type == ArgType::kUInt) {
          builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printIU32"), ret_type, args);
        } else {
          builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printI32"), ret_type, args);
        }
      } else if (elem_type.isInteger(64)) {
        if (type == ArgType::kUInt) {
          builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printIU64"), ret_type, args);
        } else {
          builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printI64"), ret_type, args);
        }
      } else if (elem_type.isInteger(16)) {
        if (type == ArgType::kUInt) {
          builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printIU16"), ret_type, args);
        } else {
          builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printI16"), ret_type, args);
        }
      } else if (elem_type.isInteger(8)) {
        if (type == ArgType::kUInt) {
          builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printIU8"), ret_type, args);
        } else {
          builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printI8"), ret_type, args);
        }
      }
      ret_type = {};
      args = {};
      builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printComma"), ret_type, args);
  
      return;
    }

    mlir::Value lower, upper, step;
    if (shape[index] != -1) {
      upper = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getIndexType(), builder_.getIndexAttr(shape[index]));
    } else {
      upper = builder_.create<mlir::memref::DimOp>(builder_.getUnknownLoc(), memref, index);
    }
    lower = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getIndexType(), builder_.getIndexAttr(0));
    step = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getIndexType(), builder_.getIndexAttr(1));
    mlir::scf::ForOp loop = builder_.create<mlir::scf::ForOp>(builder_.getUnknownLoc(), lower, upper, step);
    iters.push_back(loop.getInductionVar());
    loops.push_back(loop);

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(loop.getBody());
      index++;

      gen_loop_nest();
      mlir::scf::ForOp::ensureTerminator(loop.region(), builder_, builder_.getUnknownLoc());
    }

    ret_type = {};
    args = {};
    builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printNewline"), ret_type, args);
  };
  gen_loop_nest();
  builder_.create<mlir::CallOp>(builder_.getUnknownLoc(), llvm::StringRef("printNewline"), ret_type, args);
}

void MainFuncGenerator::GenerateMainFunc() noexcept { 
  ProcessInputBuffers();
  DeclareUtilities();
  GenerateBody();
}

std::vector<ArgumentBase> MainFuncGenerator::arguments_ = {};
bool MainFuncGenerator::print_result = true;
int MainFuncGenerator::input_count = 0;
}
