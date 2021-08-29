#include <iostream>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <functional>

#include "CodeGen_MLIR.h"
#include "MainFuncGenerator.h"
#include "AffineHelper.h"
#include "Debug.h"

#ifndef N
#define N 2600
#endif

namespace Halide {
namespace Internal {


CodeGen_MLIR::CodeGen_MLIR(const Target& t)
    : IRVisitor(),
      target_(t),
      context_(),
      builder_(mlir::OpBuilder(&context_)) {
  module_ = builder_.create<mlir::ModuleOp>(builder_.getUnknownLoc());
  Init();
  DEBUG() << "init done" << std::endl;
}

template <>
void CodeGen_MLIR::TablePush<mlir::Type>(const std::string& name, mlir::Type type) noexcept {
  DEBUG() << "push type name: " << name << std::endl;
  type_table_.push(name, type);
}

template <>
void CodeGen_MLIR::TablePop<mlir::Type>(const std::string& name) noexcept {
  type_table_.pop(name);
}

template <>
mlir::Type CodeGen_MLIR::TableGet<mlir::Type>(const std::string& name) noexcept {
  if (type_table_.count(name)) {
    return type_table_.get(name);
  }

  DEBUG() << "type not found : " << name << std::endl;
  assert(false && "type not found");
}

template <>
bool CodeGen_MLIR::TableContains<mlir::Type>(const std::string& name) noexcept {
  return type_table_.contains(name);
}

template <>
void CodeGen_MLIR::TablePush<mlir::LLVM::LLVMFuncOp&&>(const std::string& name, mlir::LLVM::LLVMFuncOp&& func) noexcept {
  DEBUG() << "push func name: " << name << std::endl;
  func_table_.push(name, std::forward<mlir::LLVM::LLVMFuncOp>(func));
}

template <>
void CodeGen_MLIR::TablePop<mlir::LLVM::LLVMFuncOp>(const std::string& name) noexcept {
  func_table_.pop(name);
}

template <>
mlir::LLVM::LLVMFuncOp CodeGen_MLIR::TableGet<mlir::LLVM::LLVMFuncOp>(const std::string& name) noexcept {
  if (func_table_.count(name)) {
    return func_table_.get(name);
  }

  DEBUG() << "function not found : " << name << std::endl;
  assert(false && "function not found");
}

template <>
bool CodeGen_MLIR::TableContains<mlir::LLVM::LLVMFuncOp>(const std::string& name) noexcept {
  return func_table_.contains(name);
}

mlir::LLVM::LLVMStructType CodeGen_MLIR::CreateLlvmStructType(std::string name) noexcept {
  mlir::LLVM::LLVMStructType T = mlir::LLVM::LLVMStructType::getIdentified(&context_,
                                                                           llvm::StringRef(name));
  TablePush<mlir::Type>(name, T);

  return T;
}

void CodeGen_MLIR::DeclareHalideFunction(const std::string& name, mlir::Type ret_type, std::vector<mlir::Type> args_type) noexcept {
  auto func_type = mlir::LLVM::LLVMFunctionType::get(ret_type, args_type, false);
  auto halide_func = builder_.create<mlir::LLVM::LLVMFuncOp>(builder_.getUnknownLoc(),
                                                             name,
                                                             func_type);

  TablePush<mlir::LLVM::LLVMFuncOp&&>(name, std::move(halide_func));
}

void CodeGen_MLIR::InitHalideStructs() noexcept {
  halide_buffer_t_type = CreateLlvmStructType("struct.halide_buffer_t");
  halide_type_t_type = CreateLlvmStructType("struct.halide_type_t");
  halide_dimension_t_type = CreateLlvmStructType("struct.halide_dimension_t");
  halide_metadata_t_type = CreateLlvmStructType("struct.halide_metadata_t");
  halide_argument_t_type = CreateLlvmStructType("struct.halide_argument_t");
  halide_scalar_value_t_type = CreateLlvmStructType("struct.halide_scalar_value_t");
  halide_device_interface_t_type = CreateLlvmStructType("struct.halide_device_interface_t");
  halide_pseudostack_slot_t_type = CreateLlvmStructType("struct.halide_pseudostack_slot_t");
  halide_semaphore_t_type = CreateLlvmStructType("struct.halide_semaphore_t");
  halide_semaphore_acquire_t_type = CreateLlvmStructType("struct.halide_semaphore_acquire_t");
  halide_parallel_task_t_type = CreateLlvmStructType("struct.halide_parallel_task_t");
}

void CodeGen_MLIR::InitHalideFunctions() noexcept {
  DeclareHalideFunction("_halide_buffer_get_dimensions",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_get_host",
                        mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8)),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_get_device",
                        builder_.getI64Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});
  
  DeclareHalideFunction("_halide_buffer_get_device_interface",
                        mlir::LLVM::LLVMPointerType::get(halide_device_interface_t_type),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_get_min",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        builder_.getI32Type()});

  DeclareHalideFunction("_halide_buffer_get_max",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        builder_.getI32Type()});
  
  DeclareHalideFunction("_halide_buffer_get_extent",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        builder_.getI32Type()});

  DeclareHalideFunction("_halide_buffer_get_stride",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        builder_.getI32Type()});

  DeclareHalideFunction("_halide_buffer_set_host_dirty",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        builder_.getI1Type()});
  
  DeclareHalideFunction("_halide_buffer_set_device_dirty",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        builder_.getI1Type()});

  DeclareHalideFunction("_halide_buffer_get_host_dirty",
                        builder_.getI1Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_get_device_dirty",
                        builder_.getI1Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_get_shape",
                        mlir::LLVM::LLVMPointerType::get(halide_dimension_t_type),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_is_bounds_query",
                        builder_.getI1Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_get_type",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_init",
                        mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        mlir::LLVM::LLVMPointerType::get(halide_dimension_t_type),
                        mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8)),
                        builder_.getI64Type(),
                        mlir::LLVM::LLVMPointerType::get(halide_device_interface_t_type),
                        builder_.getI32Type(),
                        builder_.getI32Type(),
                        builder_.getI32Type(),
                        mlir::LLVM::LLVMPointerType::get(halide_dimension_t_type),
                        builder_.getI64Type()});

  DeclareHalideFunction("_halide_buffer_init_from_buffer",
                        mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        mlir::LLVM::LLVMPointerType::get(halide_dimension_t_type),
                        mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type)});

  DeclareHalideFunction("_halide_buffer_crop",
                        mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        {mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8)),
                        mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        mlir::LLVM::LLVMPointerType::get(halide_dimension_t_type),
                        mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8)),
                        mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8))});

  DeclareHalideFunction("_halide_buffer_retire_crop_after_extern_stage",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8)),
                        mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8))});

  DeclareHalideFunction("_halide_buffer_retire_crops_after_extern_stage",
                        builder_.getI32Type(),
                        {mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8)),
                        mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8))});

  DeclareHalideFunction("_halide_buffer_set_bounds",
                        mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        {mlir::LLVM::LLVMPointerType::get(halide_buffer_t_type),
                        builder_.getI32Type(),
                        builder_.getI32Type(),
                        builder_.getI32Type()});
}

void CodeGen_MLIR::InitDialect() noexcept {
  context_.loadDialect<mlir::StandardOpsDialect>();
  context_.loadDialect<mlir::AffineDialect>();
  context_.loadDialect<mlir::LLVM::LLVMDialect>();
  context_.loadDialect<mlir::scf::SCFDialect>();
  context_.loadDialect<mlir::vector::VectorDialect>();
  context_.loadDialect<mlir::shape::ShapeDialect>();
  context_.loadDialect<mlir::linalg::LinalgDialect>();
  context_.loadDialect<mlir::memref::MemRefDialect>();
  context_.loadDialect<mlir::math::MathDialect>();
}

#if defined(IMPORT_LLVM_MODULES)
namespace {

std::vector<std::string> registered_modules;

}

#define DECLARE_GET_MODULE_(name, suffix) \
  extern "C" unsigned char halide_internal_initmod_##name##_##suffix[]; \
  extern "C" int halide_internal_initmod_##name##_##suffix##_length; \
  inline std::unique_ptr<llvm::Module> get_module_##name(llvm::LLVMContext& context) { \
    registered_modules.push_back(std::string(#name) + "_" + std::string(#suffix)); \
    llvm::StringRef data((const char*)halide_internal_initmod_##name##_##suffix, halide_internal_initmod_##name##_##suffix##_length); \
    llvm::MemoryBufferRef bufRef(data, #name); \
    auto ret = llvm::expectedToErrorOr(llvm::parseBitcodeFile(bufRef, context)); \
    assert(ret); \
    return std::move(*ret); \
  }

#ifndef BITS
#define BITS 64
#endif

#if BITS == 64
#define DECLARE_GET_MODULE(name) \
  DECLARE_GET_MODULE_(name, 64)
#elif BITS == 32
#define DECLARE_GET_MODULE(name) \
  DECLARE_GET_MODULE_(name, 32)
#else
#error should be 32 or 64 bits
#endif

#define DECLARE_GET_MODULE_LL(name) \
  DECLARE_GET_MODULE_(name, ll)


DECLARE_GET_MODULE(posix_allocator)
DECLARE_GET_MODULE(posix_error_handler)
DECLARE_GET_MODULE(posix_print)
DECLARE_GET_MODULE(linux_clock)
DECLARE_GET_MODULE(posix_io)
DECLARE_GET_MODULE(linux_host_cpu_count)
DECLARE_GET_MODULE(linux_yield)
DECLARE_GET_MODULE(posix_threads)
DECLARE_GET_MODULE(posix_get_symbol)
DECLARE_GET_MODULE(halide_buffer_t)
DECLARE_GET_MODULE(destructors)
DECLARE_GET_MODULE(pseudostack)
DECLARE_GET_MODULE_LL(posix_math)
DECLARE_GET_MODULE(to_string)
DECLARE_GET_MODULE_LL(x86)
DECLARE_GET_MODULE(can_use_target)
DECLARE_GET_MODULE(x86_cpu_features)
DECLARE_GET_MODULE(module_aot_ref_count)
DECLARE_GET_MODULE(runtime_api)

void CodeGen_MLIR::InitLlvmModules() noexcept {
  std::vector<std::unique_ptr<llvm::Module>> llvm_modules;
  llvm::LLVMContext context;
  Halide::Target target = get_target_from_environment();

  llvm_modules.push_back(get_module_posix_allocator(context));
  llvm_modules.push_back(get_module_posix_print(context));
  llvm_modules.push_back(get_module_posix_io(context));
  llvm_modules.push_back(get_module_linux_host_cpu_count(context));
  llvm_modules.push_back(get_module_linux_yield(context));
  llvm_modules.push_back(get_module_posix_get_symbol(context));
  llvm_modules.push_back(get_module_halide_buffer_t(context));
  llvm_modules.push_back(get_module_posix_math(context));
  llvm_modules.push_back(get_module_posix_error_handler(context));
  llvm_modules.push_back(get_module_linux_clock(context));
  // llvm_modules.push_back(get_module_posix_threads(context));
  llvm_modules.push_back(get_module_destructors(context));
  // llvm_modules.push_back(get_module_pseudostack(context));
  llvm_modules.push_back(get_module_to_string(context));
  llvm_modules.push_back(get_module_x86(context));


  DEBUG() << "registeded modules:" << std::endl;
  for (auto& m : registered_modules) {
    DEBUG() << m << std::endl;
  }

  const llvm::DataLayout& layout = llvm_modules[0]->getDataLayout();
  const std::string& triple = llvm_modules[0]->getTargetTriple();

  for (size_t i = 1; i < llvm_modules.size(); ++i) {
    llvm_modules[i]->setDataLayout(layout);
    llvm_modules[i]->setTargetTriple(triple);

    bool failed = llvm::Linker::linkModules(*llvm_modules[0],
                                            std::move(llvm_modules[i]));
    assert(!failed);
  }

  mlir::ModuleOp mod = mlir::translateLLVMIRToModule(std::move(llvm_modules[0]),
                                                     &this->context_).release();
  for (auto& bc : mod.body().getBlocks()) {
    for (auto& op : bc.without_terminator()) {
      module_.push_back(op.clone());
    }
  }

  for (auto& m : llvm_modules) {
    m.release();
  }
}

#endif

void CodeGen_MLIR::Init() noexcept {
#if defined(IMPORT_LLVM_MODULES)
  InitLlvmModules();
#endif
  InitDialect();
  InitHalideStructs();
  InitHalideFunctions();
}

mlir::Type CodeGen_MLIR::ConvertToMlirType(const Type& type) noexcept {
  int bits = type.bits();

  auto validate = [&]() {
    if (type.is_bool()) {
    } else if (type.is_int_or_uint()) {
      if (bits != 1 && bits!= 8 && bits != 16 && bits != 32 && bits != 64) {
        assert(false && "invalid bit width");
      }
    } else if (type.is_bfloat()) {
      if (bits != 16) {
        assert(false && "invalid bit width");
      }
    } else if (type.is_float()) {
      if (bits != 16 && bits != 32 && bits != 64) {
        assert(false && "invalid bit width");
      }
    } else if (type.is_handle()) {
      if (type.handle_type) {
        DEBUG() << "handle type: " << type.handle_type->inner_name.name << std::endl;
      }
    } else {
      assert(false && "unknown type");
    }
    return;
  };

  validate();

  if (type.is_handle()) {
    if (type.handle_type) {
      std::string handle_type_name = "struct." + type.handle_type->inner_name.name;
      if (TableContains<mlir::Type>(handle_type_name)) {
        return TableGet<mlir::Type>(handle_type_name);
      }
    }
    return mlir::LLVM::LLVMPointerType::get(builder_.getIntegerType(8));
  } else {
    return ConvertHalideTypeTToMlirType(type);
  }
}

mlir::Type CodeGen_MLIR::ConvertHalideTypeTToMlirType(const halide_type_t& h_type) noexcept {
  mlir::Type type;
  switch (h_type.code) {
    case halide_type_code_t::halide_type_int:
    case halide_type_code_t::halide_type_uint:
      type = builder_.getIntegerType(h_type.bits);
      break;
    case halide_type_code_t::halide_type_float:
      if (h_type.bits == 16) {
        type = builder_.getF16Type();
      } else if (h_type.bits == 32) {
        type = builder_.getF32Type();
      } else if (h_type.bits == 64) {
        type = builder_.getF64Type();
      } else {
        assert(false && "unknown float type");
      }
      break;
    case halide_type_code_t::halide_type_bfloat:
      type = builder_.getBF16Type();
      break;
    default:
      assert(false && "unknown type");
  }

  if (h_type.lanes != 1) {
    std::vector<int64_t> lanes = {h_type.lanes};
    type = mlir::VectorType::get(lanes, type);
  }
  return type;
}

bool CodeGen_MLIR::MaybeRegenerateToIndexOrInteger(mlir::Value& v, bool to_integer, mlir::Type target_type) noexcept {
  auto v_op = v.getDefiningOp();
  // index cast can't be used as affine symbol, so we create a new constant
  if (auto temp = mlir::dyn_cast_or_null<mlir::ConstantOp>(v_op)) {
    auto int_attr = temp.getValue().dyn_cast_or_null<mlir::IntegerAttr>();
    if (int_attr) {
      if (to_integer) {
          v = builder_.create<mlir::ConstantOp>(v_op->getLoc(),
                                                builder_.getIntegerAttr(target_type,
                                                                        int_attr.getInt()));
      } else {
          v = builder_.create<mlir::ConstantOp>(v_op->getLoc(),
                                                builder_.getIndexAttr(int_attr.getInt()));
      }
      return true;
    }
  }
  return false;
}

bool CodeGen_MLIR::MaybeRegenerateToIndexOrInteger(mlir::Value& v1, mlir::Value& v2, bool to_integer) noexcept {
  if (v1.getType() == v2.getType()) {
    return true;
  }

  if (v1.getType().isIndex() && v2.getType().isSignlessInteger()) {
    if (to_integer && MaybeRegenerateToIndexOrInteger(v1, true, v2.getType())) {
      return true;
    } else {
      return MaybeRegenerateToIndexOrInteger(v2);
    }
  } else if (v2.getType().isIndex() && v1.getType().isSignlessInteger()) {
    if (to_integer && MaybeRegenerateToIndexOrInteger(v2, true, v1.getType())) {
      return true;
    } else {
      return MaybeRegenerateToIndexOrInteger(v1);
    }
  }

  return false;
}

template<typename T, typename U1, typename U2>
void CodeGen_MLIR::GenerateArithOp(const T* op) noexcept {
  assert(op->type.code() == op->a.type().code() && op->type.code() == op->b.type().code());

  mlir::Value v1 = Codegen(op->a);
  mlir::Value v2 = Codegen(op->b);

  if ((IsValidAffineExpr(op->a) && IsValidAffineExpr(op->b))) {
    mlir::AffineMap affine_map;
    llvm::SmallVector<mlir::Value, 4> operands;
    AffineHelper H(*this);
    affine_map = H.GetAffineMap(T::make(op->a, op->b));
    operands = H.GetAffineOperands();

    value_ = builder_.create<mlir::AffineApplyOp>(builder_.getUnknownLoc(), affine_map, operands);
  } else {
    if (op->type.lanes() == 1) {
      if (!v1.getType().isIndex() && !v2.getType().isIndex()) {
        auto v1_bits = v1.getType().getIntOrFloatBitWidth();
        auto v2_bits = v2.getType().getIntOrFloatBitWidth();
        if (v1_bits > v2_bits) {
          v2 = builder_.create<mlir::SignExtendIOp>(builder_.getUnknownLoc(),
                                                    v2,
                                                    v1.getType());
        } else if (v1_bits < v2_bits) {
          v1 = builder_.create<mlir::SignExtendIOp>(builder_.getUnknownLoc(),
                                                    v1,
                                                    v2.getType());
        }
      }
    }

    if (op->a.type().is_int_or_uint()) {
      if (op->type.lanes() == 1) {
        if (!MaybeRegenerateToIndexOrInteger(v1, v2, true)) {
          if (v1.getType().isIndex()) {
            v2 = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), v2);
          } else if (v2.getType().isIndex()) {
            v1 = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), v1);
          } else {
            // assert(false && "unimplemented type comparison(currently only int and index)");
          }
        }
      }

      if (op->_node_type == IRNodeType::Div) {
        value_ = Codegen(lower_int_uint_div(op->a, op->b));
      } else if (op->_node_type == IRNodeType::Mod) {
        value_ = Codegen(lower_int_uint_mod(op->a, op->b));
      } else {
        value_ = builder_.create<U2>(builder_.getUnknownLoc(), v1.getType(), v1, v2);
      }
    } else {
      value_ = builder_.create<U1>(builder_.getUnknownLoc(), v1.getType(), v1, v2);
    }
  }
}

template<typename T>
void CodeGen_MLIR::GenerateCmpOp(const T* op, mlir::CmpFPredicate type_f, mlir::CmpIPredicate type_i) noexcept {
  Type type;

  if (op->a.type().is_float()) {
    type = Float(std::max(op->a.type().bits(),
                          op->b.type().bits()),
                 op->a.type().lanes());
  } else {
    type = op->a.type().with_bits(std::max(op->a.type().bits(),
                                                 op->b.type().bits()));
  }

  mlir::Value v1 = Codegen(cast(type, op->a));
  assert(v1);

  mlir::Value v2 = Codegen(cast(type, op->b));
  assert(v2);

  if (type.is_int_or_uint()) {
    if (op->type.lanes() == 1) {
    if (!MaybeRegenerateToIndexOrInteger(v1, v2, true)) {
      if (v1.getType().isIndex()) {
        v2 = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), v2);
      } else if (v2.getType().isIndex()) {
        v1 = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), v1);
      } else {
        assert(false && "unimplemented type comparison(currently only int and index)");
      }
    }
    }

    value_ = builder_.create<mlir::CmpIOp>(builder_.getUnknownLoc(), type_i, v1, v2);
  } else {
    value_ = builder_.create<mlir::CmpFOp>(builder_.getUnknownLoc(), type_f, v1, v2);
  }
}

mlir::Value CodeGen_MLIR::Codegen(const Expr& e) noexcept {
  assert(e.defined());

  value_ = nullptr;

  e.accept(this);

  return value_;
}

void CodeGen_MLIR::Codegen(const Stmt& s) noexcept {
  s->accept(this);
}

template <>
void CodeGen_MLIR::TablePush<CodeGen_MLIR::VariableT>(const std::string& name, VariableT var) noexcept {
  symbol_table_.push(name, var);
}

template <>
void CodeGen_MLIR::TablePop<CodeGen_MLIR::VariableT>(const std::string& name) noexcept {
  symbol_table_.pop(name);
}

template <>
CodeGen_MLIR::VariableT CodeGen_MLIR::TableGet<CodeGen_MLIR::VariableT>(const std::string& name) noexcept {
  if (symbol_table_.count(name)) {
    return symbol_table_.get(name);
  }

  DEBUG() << "symbol not found : " << name << std::endl;
  return {};
}

template <>
bool CodeGen_MLIR::TableContains<CodeGen_MLIR::VariableT>(const std::string& name) noexcept {
  return symbol_table_.contains(name);
}

mlir::OpBuilder& CodeGen_MLIR::GetBuilder() noexcept {
  return builder_;
}

mlir::MLIRContext& CodeGen_MLIR::GetContext() noexcept {
  return context_;
}

void CodeGen_MLIR::compile(const Module& module) {
  for (const auto& buf : module.buffers()) {
    CompileBuffer(buf);
  }

  for (const auto& func : module.functions()) {
    CompileFunction(func);
  }

  mlir::PassManager pm(&context_);
  mlir::applyPassManagerCLOptions(pm);
  mlir::OpPassManager &func_pm = pm.nest<mlir::FuncOp>();

  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // pm.addPass(mlir::createCanonicalizerPass());
  func_pm.addPass(mlir::createSimplifyAffineStructuresPass());
  func_pm.addPass(mlir::createPromoteBuffersToStackPass(1024, 64, 3));
  func_pm.addPass(mlir::createBufferDeallocationPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  if (mlir::failed(pm.run(module_))) {
    std::cerr << "error happened" << std::endl;
    return;
  }

  if (mlir::failed(module_.verify())) {
    std::cerr << "fail to verify" << std::endl;
    return;
  }
  
  Utils::MainFuncGenerator g{*this};
  g.GenerateMainFunc();

  std::error_code ec;
  llvm::raw_fd_ostream os{"runme.mlir", ec};
  module_.print(os);

  return;
}


void CodeGen_MLIR::CompileFunction(const LoweredFunc& func) noexcept {
  DEBUG() << "func: " << func.name << std::endl;
  mlir::OpBuilder::InsertionGuard guard(builder_);
  func_names_.push_back(func.name);
  for (auto& func_arg : func.args) {
      
    if (func_arg.is_buffer()) {
      DEBUG() << func_arg.name << " is buffer, " << "dimensions: " << (int)func_arg.dimensions << "bits: " << func_arg.type.bits() << std::endl;;
      std::vector<int64_t> shape;
      if (func_arg.argument_estimates.buffer_estimates.size()) {
        auto buf_estimates = func_arg.argument_estimates.buffer_estimates;
        for (size_t i = 0; i < buf_estimates.size(); ++i) {
          if (!buf_estimates[i].min.defined()) {
            constants_int_[func_arg.name + ".min." + std::to_string(i)] = 0;
          } else {
            auto intimm = buf_estimates[i].min.as<IntImm>();
            if (intimm) {
              constants_int_[func_arg.name + ".min." + std::to_string(i)] = intimm->value;
            }
          }
          if (!buf_estimates[i].extent.defined()) {
            shape.push_back(-1);
          } else {
            auto intimm = buf_estimates[i].extent.as<IntImm>();
            if (intimm) {
              shape.push_back(intimm->value);
              constants_int_[func_arg.name + ".extent." + std::to_string(i)] = intimm->value;
            }
          }
        }
      }

      std::reverse(shape.begin(), shape.end());
      mlir::Type buffer_elem_type = ConvertHalideTypeTToMlirType(func_arg.type);
      mlir::MemRefType memref_type = mlir::MemRefType::get(shape, buffer_elem_type);

      func_args_[func.name].push_back(memref_type);
    } else {
      DEBUG() << "is not buffer: " << func_arg.name << std::endl;
      auto est = func_arg.argument_estimates.scalar_estimate;
      if (est.defined()) {
        if (auto i = est.as<IntImm>()) {
          DEBUG() << func_arg.name << ": int" << std::endl;
          constants_int_[func_arg.name] = i->value;
        } else if (auto f = est.as<FloatImm>()) {
          DEBUG() << func_arg.name << ": float" << std::endl;
          constants_float_[func_arg.name] = f->value;
        }
      }
      func_args_[func.name].push_back(ConvertToMlirType(func_arg.type).cast<mlir::Type>());
    }
  }

  auto func_type = builder_.getFunctionType(func_args_[func.name], {});
  auto mlir_func = builder_.create<mlir::FuncOp>(builder_.getUnknownLoc(), func.name, func_type);

  auto entry = mlir_func.addEntryBlock();
  builder_.setInsertionPointToStart(entry);

  for (size_t i = 0; i < func.args.size(); ++i) {
    //TODO: how to convert integral block argument to constant
    mlir::Value func_arg = mlir_func.getArgument(i);
    if (func.args[i].is_buffer()) {
      TablePush(func.args[i].name,
                VariableT({func_arg, VarType::kInputArg, func_arg.getType(), func.args[i].dimensions}));

      for (size_t j = 0; j < func.args[i].dimensions; ++j) {
        if (constants_int_.count(func.args[i].name + ".extent." + std::to_string(j))) {
          continue;
        }
        mlir::Value temp = builder_.create<mlir::memref::DimOp>(builder_.getUnknownLoc(),
                                                                func_arg,
                                                                func.args[i].dimensions - 1 - j);
        TablePush(func.args[i].name + ".extent." + std::to_string(j),
                  VariableT({temp, VarType::kInputArg}));
      }
    } else {
      // if (constants_int_.count(func.args[i].name)) {
      //   mlir::Value temp = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
      //                                                        builder_.getIndexType(),
      //                                                        builder_.getIndexAttr(constants_int_[func.args[i].name]));
      //   TablePush(func.args[i].name,
      //             VariableT({temp, VarType::kAffineSym}));
      // } else {
        TablePush(func.args[i].name,
                  VariableT({func_arg, VarType::kInputArg}));
      // }
    }
  }

  for (auto it = global_buffers_.cbegin(); it != global_buffers_.cend(); ++it) {
    mlir::memref::GlobalOp global_memref = it.stack().top();
    mlir::Value buffer = builder_.create<mlir::memref::GetGlobalOp>(builder_.getUnknownLoc(),
                                                                    global_memref.type(),
                                                                    global_memref.getName());
    TablePush<VariableT>(it.name(), {buffer, VarType::kInputArg});
  }

  std::cout << "simplifying..." << std::endl;
  Simplifier s(*this);
  // std::cout << func.body << std::endl;
  Stmt sim = simplify(s.mutate(func.body));
  // std::cout << sim << std::endl;
  // Codegen(sim);
  Codegen(func.body);

  builder_.create<mlir::ReturnOp>(builder_.getUnknownLoc());

  allocas_.clear();
  module_.push_back(mlir_func);
}


void CodeGen_MLIR::CompileBuffer(const Buffer<>& buffer) noexcept {
  DEBUG() << "buf name: " << buffer.name() << " dimension: " << buffer.dimensions() << std::endl;

  std::vector<int64_t> buffer_size;

  constants_int_[buffer.name() + ".dimensions"] = buffer.dimensions();
  for (int i = 0; i < buffer.dimensions(); ++i) {
    constants_int_[buffer.name() + ".min." + std::to_string(i)] = buffer.min(i);
    constants_int_[buffer.name() + ".extent." + std::to_string(i)] = buffer.extent(i);
    buffer_size.push_back(buffer.extent(i));
    constants_int_[buffer.name() + ".stride." + std::to_string(i)] = buffer.stride(i);
  }

  std::reverse(buffer_size.begin(), buffer_size.end());

  mlir::Type buffer_elem_type = ConvertHalideTypeTToMlirType(buffer.raw_buffer()->type);
  mlir::RankedTensorType buf_type = mlir::RankedTensorType::get(buffer_size, buffer_elem_type);
  std::vector<char> buf_data((char*)buffer.data(), (char*)buffer.data() + buffer.size_in_bytes());
  mlir::Attribute buf_attr = mlir::DenseElementsAttr::getFromRawBuffer(buf_type, llvm::ArrayRef<char>(buf_data), false);

  auto global_memref = builder_.create<mlir::memref::GlobalOp>(
                         builder_.getUnknownLoc(),
                         mlir::Twine("__buffer_" + buffer.name()).str(),
                         builder_.getStringAttr("private"),
                         mlir::BufferizeTypeConverter{}.convertType(buf_type),
                         buf_attr.cast<mlir::ElementsAttr>(),
                         true);
  global_buffers_.push(buffer.name(), global_memref);

  module_.push_back(global_memref);
}

void CodeGen_MLIR::visit(const IntImm* op) {
  DEBUG() << "IntImm: " << op->value << std::endl;

  mlir::Attribute attr = builder_.getIntegerAttr(ConvertToMlirType(op->type), op->value);

  value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), attr);
}

void CodeGen_MLIR::visit(const UIntImm* op) {
  DEBUG() << "UntImm: " << op->value << std::endl;

  mlir::Attribute attr = builder_.getIntegerAttr(ConvertToMlirType(op->type), op->value);

  value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), attr);
}

void CodeGen_MLIR::visit(const FloatImm* op) {
  DEBUG() << "FloatImm: " << op->value << std::endl;
  value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                             builder_.getFloatAttr(ConvertToMlirType(op->type),
                                                                   op->value));
}

void CodeGen_MLIR::visit(const StringImm* op) {
  DEBUG() << "string" << std::endl;
  value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                             builder_.getStringAttr(op->value));
}

int CodeGen_MLIR::IsValidAffineExpr(const Expr& index) {
  if (index.type().lanes() != 1) {
    return 0;
  }

  if (index.as<IntImm>() || index.as<UIntImm>()) {
    return 1;
  } else if (auto var = index.as<Variable>()) {
    if (TableContains<VariableT>(var->name)) {
      if (TableGet<VariableT>(var->name).var_type == VarType::kAffineDim) {
        return 2;
      } else if (TableGet<VariableT>(var->name).var_type == VarType::kInputArg) {
        return 0;
      } else {
        return 0;
      }
    } else if (constants_int_.count(var->name)) {
      return 1;
    } else if (lazy_evals_.contains(var->name)) {
      return IsValidAffineExpr(lazy_evals_.get(var->name));
    }

    DEBUG() << "Unknown variable: " << var->name << std::endl;
    assert(false);
  } else if (auto add = index.as<Add>()) {
    int valid_a = IsValidAffineExpr(add->a);
    int valid_b = IsValidAffineExpr(add->b);
    if (!(valid_a && valid_b)) {
      return 0;
    } else if (valid_a == 2 || valid_b == 2) {
      return 2;
    } else {
      return 1;
    }
  } else if (auto sub = index.as<Sub>()) {
    int valid_a = IsValidAffineExpr(sub->a);
    int valid_b = IsValidAffineExpr(sub->b);
    if (!(valid_a && valid_b)) {
      return 0;
    } else if (valid_a == 2 || valid_b == 2) {
      return 2;
    } else {
      return 1;
    }
  } else if (auto mul = index.as<Mul>()) {
    int valid_a = IsValidAffineExpr(mul->a);
    int valid_b = IsValidAffineExpr(mul->b);
    if (!(valid_a && valid_b)) {
      return 0;
    } else if (valid_a == 2 && valid_b == 2) {
      return 0;
    } else if (valid_a == 2 || valid_b == 2) {
      return 2;
    } else {
      return 1;
    }
  } else if (auto div = index.as<Div>()) {
    int valid_a = IsValidAffineExpr(div->a);
    int valid_b = IsValidAffineExpr(div->b);
    if (!(valid_a && valid_b)) {
      return 0;
    } else if (valid_b == 2) {
      return 0;
    } else if (valid_a == 2) {
      return 2;
    } else {
      return 1;
    }
  } else if (auto mod = index.as<Mod>()) {
    int valid_a = IsValidAffineExpr(mod->a);
    int valid_b = IsValidAffineExpr(mod->b);
    if (!(valid_a && valid_b)) {
      return 0;
    } else if (valid_b == 2) {
      return 0;
    } else if (valid_a == 2) {
      return 2;
    } else {
      return 1;
    }
  } else {
    return 0;
  }
}

bool CodeGen_MLIR::IsValidAffineVariable(mlir::Value v) {
  return isValidSymbol(v) || isValidDim(v);
}

void CodeGen_MLIR::visit(const For* op) {
  DEBUG() << "for name: " << op->name << std::endl;

  bool is_valid_affine_op = IsValidAffineExpr(op->min) &&
                            IsValidAffineExpr(Add::make(op->min, op->extent));

  if (is_valid_affine_op) {
    mlir::AffineMap lbMap, ubMap;
    llvm::SmallVector<mlir::Value, 4> lbOperands, ubOperands;
    // {
      AffineHelper H(*this);
      lbMap = H.GetAffineMap(op->min);
      lbOperands = H.GetAffineOperands();
    // }

    // {
      AffineHelper H1(*this);
      ubMap = H1.GetAffineMap(Add::make(op->min, op->extent));
      ubOperands = H1.GetAffineOperands();
    // }

      
    mlir::AffineForOp loop = builder_.create<mlir::AffineForOp>(builder_.getUnknownLoc(),
                                                                lbOperands,
                                                                lbMap,
                                                                ubOperands,
                                                                ubMap);

    if (TableContains<VariableT>(op->name)) {
      TablePop<VariableT>(op->name);
    }
    TablePush(op->name, VariableT({loop.getInductionVar(), VarType::kAffineDim}));

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);

      builder_.setInsertionPointToStart(loop.getBody());

      op->body.accept(this);

      mlir::AffineForOp::ensureTerminator(loop.region(), builder_, builder_.getUnknownLoc());
    }
  } else {
    mlir::Value lower = Codegen(op->min);
    if (!lower.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(lower)) {
        lower = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), lower);
      }
    }

    mlir::Value upper = Codegen(Add::make(op->min, op->extent));
    if (!upper.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(upper)) {
        upper = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), upper);
      }
    }

    mlir::Value step = builder_.create<mlir::ConstantIndexOp>(builder_.getUnknownLoc(), 1);
    if (!step.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(step)) {
        step = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), step);
      }
    }


    mlir::scf::ForOp loop = builder_.create<mlir::scf::ForOp>(builder_.getUnknownLoc(), lower, upper, step);

    if (TableContains<VariableT>(op->name)) {
      TablePop<VariableT>(op->name);
    }
    TablePush(op->name, VariableT({loop.getInductionVar(), VarType::kScfDim}));

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(loop.getBody());

      op->body.accept(this);

      mlir::scf::ForOp::ensureTerminator(loop.region(), builder_, builder_.getUnknownLoc());
    }
  }
}

void CodeGen_MLIR::visit(const Add* op) {
  assert(op->a.defined() && op->b.defined());
  DEBUG() << "add: " << op->a << " + " << op->b << std::endl;

  GenerateArithOp<Add, mlir::AddFOp, mlir::AddIOp>(op);

  DEBUG() << "add done" << std::endl;
}

void CodeGen_MLIR::visit(const Sub* op) {
  assert(op->a.defined() && op->b.defined());
  DEBUG() << "sub: " << op->a << " - " << op->b << std::endl;

  GenerateArithOp<Sub, mlir::SubFOp, mlir::SubIOp>(op);

  DEBUG() << "sub done" << std::endl;
}

void CodeGen_MLIR::visit(const Mul* op) {
  assert(op->a.defined() && op->b.defined());
  DEBUG() << "mul: " << op->a << " * " << op->b << std::endl;

  GenerateArithOp<Mul, mlir::MulFOp, mlir::MulIOp>(op);

  DEBUG() << "mul done" << std::endl;
}

void CodeGen_MLIR::visit(const Div* op) {
  assert(op->a.defined() && op->b.defined());
  DEBUG() << "div: " << op->a << " / " << op->b << std::endl;

  GenerateArithOp<Div, mlir::DivFOp, mlir::SignedDivIOp>(op);

  DEBUG() << "div done" << std::endl;
}

void CodeGen_MLIR::visit(const Mod* op) {
  assert(op->a.defined() && op->b.defined());
  DEBUG() << "mod: " << op->a << " % " << op->b << std::endl;

  GenerateArithOp<Mod, mlir::RemFOp, mlir::UnsignedRemIOp>(op);

  DEBUG() << "mod done" << std::endl;
}


void CodeGen_MLIR::visit(const Let* op) {
  DEBUG() << "let: " << op->name << std::endl;
  assert(op->value.defined() && op->body.defined());

  // The global buffers and the information of them might be in symbol table
  // already, if so, skip generating such variables
  if (!(TableContains<VariableT>(op->name) || global_buffers_.contains(op->name))) {
    DEBUG() << "not in tableeeeeeeeee: " << op->name << std::endl;
    if (constants_int_.count(op->name)) {
      value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                                 builder_.getIndexType(),
                                                 builder_.getIndexAttr(constants_int_[op->name]));
      TablePush(op->name, VariableT({value_, VarType::kAffineSym}));
    } else {
      lazy_evals_.push(op->name, op->value);
      value_ = Codegen(op->value);
      if (auto i = op->value.as<IntImm>()) {
        constants_int_[op->name] = i->value;
      } else if (auto f = op->value.as<FloatImm>()) {
        constants_float_[op->name] = f->value;
      }
      TablePush(op->name, VariableT({value_, VarType::kNormal}));
    }
  }

  value_ = Codegen(op->body);
  DEBUG() << "let done: " << op->name << std::endl;
  TablePop<VariableT>(op->name);
}

void CodeGen_MLIR::visit(const LetStmt* op) {
  DEBUG() << "let stmt: " << op->name << std::endl;
  assert(op->value.defined() && op->body.defined());

  // The global buffers and the information of them might be in symbol table
  // already, if so, skip generating such variables
  if (!(TableContains<VariableT>(op->name) || global_buffers_.contains(op->name))) {
    DEBUG() << "not in tableeeeeeeeee: " << op->name << std::endl;
    if (constants_int_.count(op->name)) {
      value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                                 builder_.getIndexType(),
                                                 builder_.getIndexAttr(constants_int_[op->name]));
      TablePush(op->name, VariableT({value_, VarType::kAffineSym}));
    } else {
      lazy_evals_.push(op->name, op->value);
      DEBUG() << "lazy evals push: " << op->name << std::endl;
      value_ = Codegen(op->value);
      if (auto i = op->value.as<IntImm>()) {
        constants_int_[op->name] = i->value;
      } else if (auto f = op->value.as<FloatImm>()) {
        constants_float_[op->name] = f->value;
      }
      TablePush(op->name, VariableT({value_, VarType::kNormal}));
    }
  }

  Codegen(op->body);
  DEBUG() << "let stmt done: " << op->name << std::endl;
  TablePop<VariableT>(op->name);
}

void CodeGen_MLIR::visit(const IfThenElse* op) {
  DEBUG() << "if then else" << std::endl;
  
  bool withElse = op->else_case.defined();

  mlir::scf::IfOp ifThenElse = builder_.create<mlir::scf::IfOp>(builder_.getUnknownLoc(), Codegen(op->condition), withElse);
  mlir::OpBuilder::InsertionGuard guard(builder_);
  builder_.setInsertionPointToStart(ifThenElse.getBody(0));
  op->then_case.accept(this);

  mlir::scf::IfOp::ensureTerminator(ifThenElse.thenRegion(), builder_, builder_.getUnknownLoc());

  if (withElse) {
    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPointToStart(ifThenElse.getBody(1));
    op->else_case.accept(this);

    mlir::scf::IfOp::ensureTerminator(ifThenElse.elseRegion(), builder_, builder_.getUnknownLoc());
  }
}

void CodeGen_MLIR::visit(const Block* op) {
  DEBUG() << "block" << std::endl;

  op->first.accept(this);
  op->rest.accept(this);
}


void CodeGen_MLIR::visit(const AssertStmt* op) {
  return;
}

void CodeGen_MLIR::visit(const Cast* op) {
  DEBUG() << "castttttttttt: " << op->value << std::endl;

  Type type_dst_halide = op->type;
  Type type_src_halide = op->value.type();
  DEBUG() << " bitssssssss: " << op->type.bits() << std::endl;
  mlir::Type type_dest_mlir = ConvertToMlirType(type_dst_halide);

  value_ = Codegen(op->value);

  if (type_src_halide.is_float()) {
    if (type_dst_halide.is_float()) {
      if (type_src_halide.bits() < type_dst_halide.bits()) {
        value_ = builder_.create<mlir::FPExtOp>(builder_.getUnknownLoc(),
                                                value_, ConvertToMlirType(type_dst_halide));
      } else {
        assert(false && "no fp truncation op available");
      }
    } else if (type_dst_halide.is_int_or_uint()) {
      if (type_dst_halide.is_uint()) {
        value_ = builder_.create<mlir::FPToUIOp>(builder_.getUnknownLoc(),
                                                 value_,
                                                 ConvertToMlirType(type_dst_halide));
      } else {
        value_ = builder_.create<mlir::FPToSIOp>(builder_.getUnknownLoc(),
                                                 value_,
                                                 ConvertToMlirType(type_dst_halide));
      }
    } else {
      assert(false && "unknown dst type");
    }
  } else if (type_src_halide.is_int_or_uint()) {
    if (type_dst_halide.is_float()) {
      if (value_.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(value_, true, ConvertToMlirType(type_src_halide))) {
          value_ = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                      ConvertToMlirType(type_src_halide), value_);
        }
      }
      if (type_src_halide.is_uint()) {
        value_ = builder_.create<mlir::UIToFPOp>(builder_.getUnknownLoc(),
                                                 value_,
                                                 ConvertToMlirType(type_dst_halide));
      } else {
        value_ = builder_.create<mlir::SIToFPOp>(builder_.getUnknownLoc(),
                                                 value_,
                                                 ConvertToMlirType(type_dst_halide));
      }
    } else if (type_dst_halide.is_int_or_uint()) {
      if (type_src_halide.bits() < type_dst_halide.bits()) {
        if (value_.getType().isIndex()) {
          if (!MaybeRegenerateToIndexOrInteger(value_, true, ConvertToMlirType(type_src_halide))) {
            value_ = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                        type_dest_mlir,
                                                        value_);
          }
        } else {
          if (type_src_halide.is_uint()) {
            value_ = builder_.create<mlir::ZeroExtendIOp>(builder_.getUnknownLoc(),
                                                          value_,
                                                          type_dest_mlir);
          } else {
            value_ = builder_.create<mlir::SignExtendIOp>(builder_.getUnknownLoc(),
                                                          value_,
                                                          type_dest_mlir);
          }
        }
      } else if (type_src_halide.bits() > type_dst_halide.bits()) {
        value_ = builder_.create<mlir::TruncateIOp>(builder_.getUnknownLoc(),
                                                    value_,
                                                    type_dest_mlir);
      }
    } else {
      assert(false && "unknown src type");
    }
  } else {
    assert(false && "unknown src type");
  }
  DEBUG() << "cast done" << std::endl;
  assert(value_ != nullptr);
}

void CodeGen_MLIR::visit(const Variable* op) {
  DEBUG() << "variable name: " << op->name << std::endl;
  if (TableContains<VariableT>(op->name)) {
    value_ = TableGet<VariableT>(op->name).value;
  } else if (global_buffers_.contains(op->name)) {
    mlir::memref::GlobalOp global_memref = global_buffers_.get(op->name);
    value_ = builder_.create<mlir::memref::GetGlobalOp>(builder_.getUnknownLoc(),
                                                        global_memref.type(),
                                                        global_memref.getName());
    TablePush<VariableT>(op->name, {value_, VarType::kNone});
  } else {
    value_ = Codegen(lazy_evals_.get(op->name));
    TablePush<VariableT>(op->name, {value_, VarType::kNone});
    // assert(false && "variable undefined");
  }
  DEBUG() << "variable done" << std::endl;
}

void CodeGen_MLIR::visit(const EQ* op) {
  DEBUG() << "eq" << std::endl;

  GenerateCmpOp<EQ>(op, mlir::CmpFPredicate::OEQ, mlir::CmpIPredicate::eq);
  DEBUG() << "eq done" << std::endl;
}

void CodeGen_MLIR::visit(const NE* op) {
  DEBUG() << "ne" << std::endl;

  GenerateCmpOp<NE>(op, mlir::CmpFPredicate::ONE, mlir::CmpIPredicate::ne);
  DEBUG() << "ne done" << std::endl;
}

void CodeGen_MLIR::visit(const LT* op) {
  DEBUG() << "lt" << std::endl;

  mlir::CmpIPredicate pred = op->a.type().is_int()? mlir::CmpIPredicate::slt : mlir::CmpIPredicate::ult;
  GenerateCmpOp<LT>(op, mlir::CmpFPredicate::OLT, pred);
  DEBUG() << "lt done" << std::endl;
}

void CodeGen_MLIR::visit(const LE* op) {
  DEBUG() << "le" << std::endl;

  mlir::CmpIPredicate pred = op->a.type().is_int()? mlir::CmpIPredicate::sle : mlir::CmpIPredicate::ule;
  GenerateCmpOp<LE>(op, mlir::CmpFPredicate::OLE, pred);
  DEBUG() << "le done" << std::endl;
}

void CodeGen_MLIR::visit(const GT* op) {
  DEBUG() << "gt" << std::endl;

  mlir::CmpIPredicate pred = op->a.type().is_int()? mlir::CmpIPredicate::sgt : mlir::CmpIPredicate::ugt;
  GenerateCmpOp<GT>(op, mlir::CmpFPredicate::OGT, pred);
  DEBUG() << "gt done" << std::endl;
}

void CodeGen_MLIR::visit(const GE* op) {
  DEBUG() << "ge" << std::endl;

  mlir::CmpIPredicate pred = op->a.type().is_int()? mlir::CmpIPredicate::sge : mlir::CmpIPredicate::uge;
  GenerateCmpOp<GE>(op, mlir::CmpFPredicate::OGE, pred);
  DEBUG() << "g3 done" << std::endl;
}

void CodeGen_MLIR::visit(const Not* op) {
  DEBUG() << "not" << std::endl;

  // TODO: figure out how to create negation integer op
  // value_ = builder_.create<mlir::NegFOp>(builder_.getUnknownLoc(), ConvertToMlirType(op->type), Codegen(op->a));
  value_ = Codegen(op->a);
}

void CodeGen_MLIR::visit(const Min* op) {
  DEBUG() << "min" << std::endl;
  // if (IsValidAffineExpr({op->a, op->b})) {
  if (IsValidAffineExpr(op->a) && IsValidAffineExpr(op->b)) {
    mlir::AffineMap affine_map;
    llvm::SmallVector<mlir::Value, 4> operands;
    AffineHelper H(*this);
    affine_map = H.GetAffineMap({op->a, op->b});
    operands = H.GetAffineOperands();

    value_ = builder_.create<mlir::AffineMinOp>(builder_.getUnknownLoc(), affine_map, operands);
  } else {
    std::string a_name = unique_name('a');
    std::string b_name = unique_name('b');
    Expr a = Variable::make(op->a.type(), a_name);
    Expr b = Variable::make(op->b.type(), b_name);

    value_ = Codegen(Let::make(a_name,
                               op->a,
                               Let::make(b_name, op->b, select(a < b, a, b))));
  }
}

void CodeGen_MLIR::visit(const Max* op) {
  DEBUG() << "max" << std::endl;
  if (IsValidAffineExpr(op->a) && IsValidAffineExpr(op->b)) {
    mlir::AffineMap affine_map;
    llvm::SmallVector<mlir::Value, 4> operands;
    AffineHelper H(*this);
    affine_map = H.GetAffineMap({op->a, op->b});
    operands = H.GetAffineOperands();

    value_ = builder_.create<mlir::AffineMaxOp>(builder_.getUnknownLoc(), affine_map, operands);
  } else {
    std::string a_name = unique_name('a');
    std::string b_name = unique_name('b');
    Expr a = Variable::make(op->a.type(), a_name);
    Expr b = Variable::make(op->b.type(), b_name);

    value_ = Codegen(Let::make(a_name,
                               op->a,
                               Let::make(b_name, op->b, select(a > b, a, b))));
  }
}

void CodeGen_MLIR::visit(const And* op) {
  DEBUG() << "and" << std::endl;

  assert(op->a.type() == op->b.type() && "value a and value b should have same type");
  mlir::Value a = Codegen(op->a);
  mlir::Value b = Codegen(op->b);

  value_ = builder_.create<mlir::AndOp>(builder_.getUnknownLoc(), a.getType(), a, b);
}

void CodeGen_MLIR::visit(const Or* op) {
  DEBUG() << "or" << std::endl;

  assert(op->a.type() == op->b.type() && "value a and value b should have same type");
  mlir::Value a = Codegen(op->a);
  mlir::Value b = Codegen(op->b);

  value_ = builder_.create<mlir::OrOp>(builder_.getUnknownLoc(), a.getType(), a, b);
}


void CodeGen_MLIR::visit(const Select* op) {
  DEBUG() << "select" << std::endl;

  mlir::Value cond = Codegen(op->condition);
  mlir::Value true_branch = Codegen(op->true_value);
  mlir::Value false_branch = Codegen(op->false_value);

  if (!MaybeRegenerateToIndexOrInteger(true_branch, false_branch, true)) {
    if (true_branch.getType().isIndex()) {
      false_branch = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), false_branch);
    } else if (false_branch.getType().isIndex()) {
      true_branch = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getIndexType(), true_branch);
    } else {
      // assert(false && "unimplemented type comparison(currently only int and index)");
    }
  }

  value_ = builder_.create<mlir::SelectOp>(builder_.getUnknownLoc(),
                                           cond,
                                           true_branch,
                                           false_branch);
  DEBUG() << "select done" << std::endl;
}


// FIXME: predicated load is not implemented yet
void CodeGen_MLIR::visit(const BufferLoad* op) {
  DEBUG() << "bufferload name: " << op->name << " dimension: " << op->index.size() << std::endl;
  assert(TableContains<VariableT>(op->name) || global_buffers_.contains(op->name));

  mlir::Value buffer; 
  std::vector<mlir::Value> indices;

  if (TableContains<VariableT>(op->name)) {
    VariableT var = TableGet<VariableT>(op->name);
    buffer = var.value;
  } else {
    mlir::memref::GlobalOp global_memref = global_buffers_.get(op->name);
    buffer = builder_.create<mlir::memref::GetGlobalOp>(builder_.getUnknownLoc(),
                                                        global_memref.type(),
                                                        global_memref.getName());
    TablePush<VariableT>(op->name, {buffer, VarType::kNone});
  }

  mlir::MemRefType buf_type = buffer.getType().cast<mlir::MemRefType>();
  if (buf_type.getShape().size() > op->index.size()) {
    std::vector<int64_t> shape_dim(op->index.size(), -1);
    mlir::MemRefType type = mlir::MemRefType::get(shape_dim, buf_type.getElementType());
    mlir::MemRefType shape_type = mlir::MemRefType::get({(int)op->index.size()}, builder_.getI32Type());
    mlir::IntegerAttr alignment = builder_.getI64IntegerAttr(8);
    mlir::Value new_shape = builder_.create<mlir::memref::AllocaOp>(builder_.getUnknownLoc(), shape_type, alignment);
    for (size_t i = 0; i < op->index.size(); ++i) {
      mlir::Value temp = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getI32Type(), builder_.getI32IntegerAttr(-1));
      mlir::Value index = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getIndexType(), builder_.getIndexAttr(i));
      builder_.create<mlir::memref::StoreOp>(builder_.getUnknownLoc(), temp, new_shape, mlir::ValueRange({index}));
    }
    buffer = builder_.create<mlir::memref::ReshapeOp>(builder_.getUnknownLoc(), type, buffer, new_shape);
  } else if (buf_type.getShape().size() < op->index.size()) {
    assert(false);
  }

  if (op->type.is_scalar()) {
    int valid_affine_index = 1;
    for (auto ind : op->index) {
      valid_affine_index = valid_affine_index && IsValidAffineExpr(ind);
    }
    if (valid_affine_index) {
      AffineHelper H(*this);
      mlir::AffineMap affine_map = H.GetAffineMap(op->index);
      llvm::SmallVector<mlir::Value, 4> operands = H.GetAffineOperands();

      value_ = builder_.create<mlir::AffineLoadOp>(builder_.getUnknownLoc(),
                                                   buffer,
                                                   affine_map,
                                                   operands);
    } else {
      for (auto& index : op->index) {
        indices.push_back(Codegen(index));
      }

      std::reverse(indices.begin(), indices.end());

      for (mlir::Value& index : indices) {
        if (!index.getType().isIndex()) {
          if (!MaybeRegenerateToIndexOrInteger(index)) {
            index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                       index,
                                                       builder_.getIndexType());
          }
        }
      }
      value_ = builder_.create<mlir::memref::LoadOp>(builder_.getUnknownLoc(),
                                                     buffer,
                                                     indices);
    }
  } else {
    auto ramp = op->index[0].as<Ramp>();
    std::vector<bool> is_vectorized_dimension(op->index.size(), false);
    int vectorized_dimensions_count = 0;
    for (size_t i = 0; i < op->index.size(); ++i) {
      if (op->index[i].type().lanes() > 1) {
        is_vectorized_dimension[i] = true;
        vectorized_dimensions_count++;
      }
    }

    std::vector<Expr> halide_index = op->index;

    if (ramp && is_const_one(ramp->stride) && vectorized_dimensions_count == 1) {
      halide_index[0] = ramp->base;
      int valid_affine_index = 1;
      for (auto ind : halide_index) {
        valid_affine_index = valid_affine_index && IsValidAffineExpr(ind);
      }

      assert(256 >= op->type.bits());

      mlir::Type type = ConvertToMlirType(op->type);

      if (valid_affine_index) {
        AffineHelper H(*this);
        mlir::AffineMap affine_map = H.GetAffineMap(halide_index);
        llvm::SmallVector<mlir::Value, 4> operands = H.GetAffineOperands();

        value_ = builder_.create<mlir::AffineVectorLoadOp>(builder_.getUnknownLoc(),
                                                           type.cast<mlir::VectorType>(),
                                                           buffer,
                                                           affine_map,
                                                           operands);
      } else {
        for (size_t i = 0; i < op->index.size(); ++i) {
          indices.push_back(Codegen(halide_index[i]));
        }
        std::reverse(indices.begin(), indices.end());

        for (mlir::Value& index : indices) {
          if (!index.getType().isIndex()) {
            if (!MaybeRegenerateToIndexOrInteger(index)) {
              index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                         index,
                                                         builder_.getIndexType());
            }
          }
        }
        value_ = builder_.create<mlir::vector::LoadOp>(builder_.getUnknownLoc(), type, buffer, indices);
      }
    } else {
      // "non dense(stride 1)" ramp load and vector of index load
      std::vector<mlir::Value> base_indices;
      int result_lanes = op->type.lanes();
      mlir::VectorType result_type = ConvertToMlirType(op->type).cast<mlir::VectorType>();
      mlir::Value zero_offset = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                                                  builder_.getIndexAttr(0));

      if (vectorized_dimensions_count == 1 && op->index[0].type().lanes() > 1) {
        mlir::Value offset_vec;

        if (auto ramp = op->index[0].as<Ramp>()) {
        // if (stride_val <= 4) {
        //     std::vector<int64_t> indices;
        //     for (int i = 0; i < stride_val * ramp->lanes; i += stride_val) {
        //       indices.push_back(i);
        //     }

        //     std::vector<mlir::Value> vectors1;
        //     std::vector<mlir::Value> vectors2;
        //     for (int i = 0; i < stride_val / 2; ++i) {
        //       vectors1.push_back(vectors[i]);
        //     }
        //     for (int i = stride_val / 2; i < stride_val; ++i) {
        //       vectors2.push_back(vectors[i]);
        //     }

        //     mlir::Value concat1 = ConcatVector(vectors1);
        //     mlir::Value concat2 = ConcatVector(vectors2);

        //     value_ = CreateShuffle(concat1, concat2, indices);
        // }
          // gather with ramp
          halide_index[0] = ramp->base;
          for (size_t i = 0; i < op->index.size(); ++i) {
            mlir::Value temp_index = Codegen(halide_index[i]);
            if (!temp_index.getType().isIndex()) {
              if (!MaybeRegenerateToIndexOrInteger(temp_index)) {
                temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                                temp_index,
                                                                builder_.getIndexType());
              }
            }
            base_indices.push_back(temp_index);
          }

          mlir::Type index_type = ConvertToMlirType(op->index[0].type());
          mlir::Attribute index_attr = builder_.getZeroAttr(index_type);
          offset_vec = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), index_type, index_attr);
          auto stride = ramp->stride;
          for (int i = 0; i < ramp->lanes; ++i) {
            mlir::Value temp_index = Codegen(simplify(i * stride));
            if (temp_index.getType().isIndex()) {
              temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getI32Type(), temp_index);
            }
            offset_vec = builder_.create<mlir::vector::InsertElementOp>(builder_.getUnknownLoc(), temp_index, offset_vec, i);
          }
        } else {
          // gather with vector of index
          base_indices.push_back(zero_offset);
          for (size_t i = 1; i < op->index.size(); ++i) {
            mlir::Value temp_index = Codegen(op->index[i]);
            if (!temp_index.getType().isIndex()) {
              if (!MaybeRegenerateToIndexOrInteger(temp_index)) {
                temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                                temp_index,
                                                                builder_.getIndexType());
              }
            }
            base_indices.push_back(temp_index);
          }
          offset_vec = Codegen(op->index[0]);
        }

        llvm::SmallVector<bool, 4> mask_(result_lanes, true);
        mlir::Attribute mask_attr = builder_.getBoolVectorAttr(mask_);
        mlir::Value mask = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), 
                                                             mlir::VectorType::get(result_lanes, builder_.getI1Type()),
                                                             mask_attr);

        mlir::Attribute zero_attr = builder_.getZeroAttr(result_type);
        mlir::Value pass_thru = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), result_type, zero_attr);

        std::reverse(base_indices.begin(), base_indices.end());
        value_ = builder_.create<mlir::vector::GatherOp>(builder_.getUnknownLoc(), result_type, buffer, base_indices, offset_vec, mask, pass_thru);
      } else {
        // the vectorized dimension is not the lowest one, or there are more than one vectorized dimensions, so we have to do the load separately
        mlir::VectorType result_type = ConvertToMlirType(op->type).cast<mlir::VectorType>();
        mlir::Attribute result_attr = builder_.getZeroAttr(result_type);
        mlir::Value result = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), result_type, result_attr);
        std::vector<mlir::Value> indices(halide_index.size());
        std::map<int, mlir::Value> index_vecs;

        std::reverse(halide_index.begin(), halide_index.end());
        for (size_t i = 0; i < halide_index.size(); ++i) {
          if (halide_index[i].type().lanes() == 1) {
            mlir::Value temp_index = Codegen(halide_index[i]);
            if (!temp_index.getType().isIndex()) {
              if (!MaybeRegenerateToIndexOrInteger(temp_index)) {
                temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                                temp_index,
                                                                builder_.getIndexType());
              }
            }
            indices[i] = temp_index;
          } else if (!halide_index[i].as<Ramp>()) {
            index_vecs[i] = Codegen(halide_index[i]);
          }
        }

        for (int i = 0; i < op->type.lanes(); ++i) {
          for (size_t j = 0; j < halide_index.size(); ++j) {
            if (halide_index[j].type().lanes() > 1) {
              mlir::Value temp_index;
              if (auto ramp = halide_index[j].as<Ramp>()) {
                temp_index = Codegen(simplify(ramp->base + i * ramp->stride));
              } else if (index_vecs.count(j)) {
                temp_index = builder_.create<mlir::vector::ExtractElementOp>(builder_.getUnknownLoc(), index_vecs[j], i);
              } else {
                assert(false && "unrecognized");
              }
              if (!temp_index.getType().isIndex()) {
                if (!MaybeRegenerateToIndexOrInteger(temp_index)) {
                  temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                                  temp_index,
                                                                  builder_.getIndexType());
                }
              }
              indices[j] = temp_index;
            }
          }
          mlir::Value value_to_insert = builder_.create<mlir::memref::LoadOp>(builder_.getUnknownLoc(), buffer, indices);
          result = builder_.create<mlir::vector::InsertElementOp>(builder_.getUnknownLoc(), value_to_insert, result, i);
        }
        value_ = result;
      }
    }
  }

  DEBUG() << "bufferload done: " << op->name << std::endl;
  return;
}

// FIXME
void CodeGen_MLIR::visit(const Load* op) {
  mlir::Value buffer; 

  if (TableContains<VariableT>(op->name)) {
    VariableT var = TableGet<VariableT>(op->name);
    buffer = var.value;
  } else {
    mlir::memref::GlobalOp global_memref = global_buffers_.get(op->name);
    buffer = builder_.create<mlir::memref::GetGlobalOp>(builder_.getUnknownLoc(),
                                                        global_memref.type(),
                                                        global_memref.getName());
    TablePush<VariableT>(op->name, {buffer, VarType::kNone});
  }

  mlir::Value index = Codegen(op->index);

  if (!index.getType().isIndex()) {
    if (!MaybeRegenerateToIndexOrInteger(index)) {
      index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                 index,
                                                 builder_.getIndexType());
    }
  }

  value_ = builder_.create<mlir::memref::LoadOp>(builder_.getUnknownLoc(),
                                                 buffer,
                                                 index);
}

void CodeGen_MLIR::visit(const Ramp* op) {
  if (is_const(op->stride) && !is_const(op->base)) {
    // If the stride is const and the base is not (e.g. ramp(x, 1,
    // 4)), we can lift out the stride and broadcast the base so
    // we can do a single vector broadcast and add instead of
    // repeated insertion
    Expr broadcast = Broadcast::make(op->base, op->lanes);
    Expr ramp = Ramp::make(make_zero(op->base.type()), op->stride, op->lanes);
    value_ = Codegen(broadcast + ramp);
  } else if (!is_const(op->stride)) {
    Expr broadcast_base = Broadcast::make(op->base, op->lanes);
    Expr broadcast_stride = Broadcast::make(op->stride, op->lanes);
    Expr ramp = Ramp::make(make_zero(op->base.type()), make_one(op->base.type()), op->lanes);
    value_ = Codegen(broadcast_base + broadcast_stride * ramp);
  } else {
    assert(is_const(op->base) && is_const(op->stride));
    // At this point base and stride should be constant. Generate
    // an insert element sequence. The code will be lifted to a
    // constant vector stored in .rodata or similar.
    mlir::Value base = Codegen(op->base);
    mlir::Value stride = Codegen(op->stride);

    mlir::Type type = ConvertToMlirType(op->type);
    value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), type, builder_.getZeroAttr(type));
    for (int i = 0; i < op->type.lanes(); i++) {
      if (i > 0) {
        if (op->type.is_float()) {
          base = builder_.create<mlir::AddFOp>(builder_.getUnknownLoc(), base.getType(), base, stride);
        } else {
          base = builder_.create<mlir::AddIOp>(builder_.getUnknownLoc(), base.getType(), base, stride);
        }
      }
      value_ = builder_.create<mlir::vector::InsertElementOp>(builder_.getUnknownLoc(), base, value_, i);
    }
  }
}

void CodeGen_MLIR::visit(const Broadcast* op) {
  DEBUG() << "broadcast" << std::endl;

  mlir::Type elem_type = ConvertToMlirType(op->value.type());
  mlir::VectorType vec_type = mlir::VectorType::get((int64_t)op->lanes,
                                                    elem_type);
  mlir::Value val = Codegen(op->value);
  if (val.getType() != elem_type) {
    if (!MaybeRegenerateToIndexOrInteger(val, true, elem_type)) {
      if (val.getType().isIndex()) {
        val = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), elem_type, val);
      } else {
        assert(false && "unimplemented type comparison(currently only int and index)");
      }
    }
  }
  value_ = builder_.create<mlir::vector::BroadcastOp>(builder_.getUnknownLoc(), vec_type, val);
}

void CodeGen_MLIR::visit(const Call* op) {
  DEBUG() << "call: " << op->name << std::endl;

  if (op->is_intrinsic(Call::reinterpret)) {
    DEBUG() << "reinterpret: " << op->name << std::endl;
    value_ = Codegen(op->args[0]);
    if (value_.getType().isIndex()) {
      value_ = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                  value_,
                                                  ConvertToMlirType(op->args[0].type()));
    }
    value_ = builder_.create<mlir::LLVM::BitcastOp>(builder_.getUnknownLoc(), ConvertToMlirType(op->type), value_);
  } else if (op->is_intrinsic(Call::abs)) {
    std::string var_name = unique_name("var_name");
    Expr var = Variable::make(op->args[0].type(), var_name);

    value_ = Codegen(Let::make(var_name, op->args[0], select(var >= 0, var, -var)));
  } else if (op->is_intrinsic(Call::absd)) {
    std::string var_name1 = unique_name("var_name1");
    std::string var_name2 = unique_name("var_name2");

    Expr var1 = Variable::make(op->args[0].type(), var_name1);
    Expr var2 = Variable::make(op->args[1].type(), var_name2);

    value_ = Codegen(Let::make(var_name1, op->args[0],
                     Let::make(var_name2, op->args[1],
                     Select::make(var1 > var2, var1 - var2, var2 - var1))));
  } else if (op->is_intrinsic(Call::make_struct)) {
    return;
  } else if (op->is_intrinsic(Call::mux)) {
    value_ = Codegen(lower_mux(op));
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    assert(op->args.size() == 2);
    mlir::Value a = Codegen(op->args[0]);
    mlir::Value b = Codegen(op->args[1]);
    mlir::Type t = ConvertToMlirType(op->args[0].type());

    if (a.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(a, true, t)) {
        a = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, a);
      }
    }
    if (b.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(b, true, t)) {
        b = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, b);
      }
    }

    value_ = builder_.create<mlir::OrOp>(builder_.getUnknownLoc(),
                                         a.getType(), a, b);
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    assert(op->args.size() == 2);
    mlir::Value a = Codegen(op->args[0]);
    mlir::Value b = Codegen(op->args[1]);
    mlir::Type t = ConvertToMlirType(op->args[0].type());

    if (a.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(a, true, t)) {
        a = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, a);
      }
    }
    if (b.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(b, true, t)) {
        b = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, b);
      }
    }

    value_ = builder_.create<mlir::XOrOp>(builder_.getUnknownLoc(),
                                          a.getType(), a, b);
  } else if (op->is_intrinsic(Call::bitwise_and)) {
    assert(op->args.size() == 2);
    mlir::Value a = Codegen(op->args[0]);
    mlir::Value b = Codegen(op->args[1]);
    mlir::Type t = ConvertToMlirType(op->args[0].type());

    if (a.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(a, true, t)) {
        a = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, a);
      }
    }
    if (b.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(b, true, t)) {
        b = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, b);
      }
    }

    value_ = builder_.create<mlir::AndOp>(builder_.getUnknownLoc(),
                                         a.getType(), a, b);
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    mlir::Value a = Codegen(op->args[0]);
    Expr temp = make_const(UInt(op->args[0].type().bits()), -1);
    if (op->args[0].type().lanes() > 1) {
      temp = Broadcast::make(temp, op->args[0].type().lanes());
    }
    mlir::Value b = Codegen(temp);
    value_ = builder_.create<mlir::XOrOp>(builder_.getUnknownLoc(), a.getType(), a, b);
  } else if (op->is_intrinsic(Call::shift_left)) {
    assert(op->args.size() == 2);

    if (op->args[1].type().is_uint()) {
      mlir::Value a = Codegen(op->args[0]);
      mlir::Value b = Codegen(op->args[1]);
      mlir::Type t = ConvertToMlirType(op->args[0].type());

      if (a.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(a, true, t)) {
          a = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, a);
        }
      }
      if (b.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(b, true, t)) {
          b = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, b);
        }
      }

      value_ = builder_.create<mlir::ShiftLeftOp>(builder_.getUnknownLoc(),
                                                  ConvertToMlirType(op->args[0].type()), a, b);
    } else {
      value_ = Codegen(lower_signed_shift_left(op->args[0], op->args[1]));
    }
  } else if (op->is_intrinsic(Call::shift_right)) {
    assert(op->args.size() == 2);

    if (op->args[1].type().is_uint()) {
      mlir::Value a = Codegen(op->args[0]);
      mlir::Value b = Codegen(op->args[1]);
      mlir::Type t = ConvertToMlirType(op->args[0].type());

      if (a.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(a, true, t)) {
          a = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, a);
        }
      }
      if (b.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(b, true, t)) {
          b = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, b);
        }
      }

      if (op->type.is_int()) {
        value_ = builder_.create<mlir::SignedShiftRightOp>(builder_.getUnknownLoc(), t, a, b);
      } else {
        value_ = builder_.create<mlir::UnsignedShiftRightOp>(builder_.getUnknownLoc(), t, a, b);
      }
    } else {
      value_ = Codegen(lower_signed_shift_right(op->args[0], op->args[1]));
    }
  } else if (op->is_intrinsic(Call::lerp)) {
      assert(op->args.size() == 3);
      Expr e = lower_lerp(op->args[0],
                          op->args[1],
                          op->args[2]);
      e = cast(op->type, e);
      Codegen(e);
  } else if (op->is_intrinsic(Call::rounding_halving_add)) {
    Expr lowered = lower_intrinsic(op);
    assert(lowered.defined() && "undefined lowered intrinsic");
    value_ = Codegen(lowered);
  } else if (op->is_intrinsic(Call::widening_sub)) {
    Expr lowered = lower_intrinsic(op);
    assert(lowered.defined() && "undefined lowered intrinsic");
    value_ = Codegen(lowered);
  } else if (op->is_intrinsic(Call::widening_mul)) {
    Expr lowered = lower_intrinsic(op);
    assert(lowered.defined() && "undefined lowered intrinsic");
    value_ = Codegen(lowered);
  } else if (op->is_intrinsic(Call::widening_add)) {
    Expr lowered = lower_intrinsic(op);
    assert(lowered.defined() && "undefined lowered intrinsic");
    value_ = Codegen(lowered);
  } else if (op->is_intrinsic(Call::mul_shift_right)) {
    Expr lowered = lower_intrinsic(op);
    assert(lowered.defined() && "undefined lowered intrinsic");
    value_ = Codegen(lowered);
  } else if (op->is_intrinsic(Call::alloca)) {
    assert(false);
  } else if (op->is_intrinsic(Call::div_round_to_zero)) {
    assert(op->args.size() == 2);
    mlir::Value a = Codegen(op->args[0]);
    mlir::Value b = Codegen(op->args[1]);
    mlir::Type t = ConvertToMlirType(op->args[0].type());

    if (op->type.lanes() == 1) {
      if (a.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(a, true, t)) {
          a = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, a);
        }
      }
      if (b.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(b, true, t)) {
          b = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, b);
        }
      }
    }

    if (op->type.is_int()) {
      value_ = builder_.create<mlir::SignedDivIOp>(builder_.getUnknownLoc(), a.getType(), a, b);
    } else if (op->type.is_uint()) {
      value_ = builder_.create<mlir::UnsignedDivIOp>(builder_.getUnknownLoc(), a.getType(), a, b);
    } else {
      assert(false && "can only be integer");
    }
  } else if (op->is_intrinsic(Call::mod_round_to_zero)) {
    assert(op->args.size() == 2);
    mlir::Value a = Codegen(op->args[0]);
    mlir::Value b = Codegen(op->args[1]);
    mlir::Type t = ConvertToMlirType(op->args[0].type());

    if (op->type.lanes() == 1) {
      if (a.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(a, true, t)) {
          a = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, a);
        }
      }
      if (b.getType().isIndex()) {
        if (!MaybeRegenerateToIndexOrInteger(b, true, t)) {
          b = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), t, b);
        }
      }
    }

    if (op->type.is_int()) {
      value_ = builder_.create<mlir::SignedRemIOp>(builder_.getUnknownLoc(), a.getType(), a, b);
    } else if (op->type.is_uint()) {
      value_ = builder_.create<mlir::UnsignedRemIOp>(builder_.getUnknownLoc(), a.getType(), a, b);
    } else {
      assert(false && "can only be integer");
    }
  } else if (op->call_type == Call::PureExtern && op->name == "pow_f32") {
    assert(op->args.size() == 2);

    Expr x = op->args[0];
    Expr y = op->args[1];

    Halide::Expr abs_x_pow_y = Internal::halide_exp(Internal::halide_log(abs(x)) * y);
    Halide::Expr nan_expr = Call::make(x.type(), "nan_f32", {}, Call::PureExtern);
    Expr iy = floor(y);
    Expr one = make_one(x.type());
    Expr zero = make_zero(x.type());
    Expr e = select(x > 0, abs_x_pow_y,        // Strictly positive x
                    y == 0.0f, one,            // x^0 == 1
                    x == 0.0f, zero,           // 0^y == 0
                    y != iy, nan_expr,         // negative x to a non-integer power
                    iy % 2 == 0, abs_x_pow_y,  // negative x to an even power
                    -abs_x_pow_y);             // negative x to an odd power

    e = common_subexpression_elimination(e);
    value_ = Codegen(e);
  } else if (op->call_type == Call::PureExtern && op->name == "floor_f32") {
    value_ = builder_.create<mlir::FloorFOp>(builder_.getUnknownLoc(),
                                             ConvertToMlirType(op->args[0].type()),
                                             Codegen(op->args[0]));
  } else if (op->call_type == Call::PureExtern && op->name == "inf_f32") {
    value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                               builder_.getF32Type(),
                                               builder_.getF32FloatAttr(std::numeric_limits<float>::max()));
  } else if (op->call_type == Call::PureExtern && op->name == "neg_inf_f32") {
    value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                               builder_.getF32Type(),
                                               builder_.getF32FloatAttr(std::numeric_limits<float>::lowest()));
  } else if (op->call_type == Call::PureExtern && op->name == "nan_f32") {
    value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                               builder_.getF32Type(),
                                               builder_.getF32FloatAttr(std::numeric_limits<float>::quiet_NaN()));
  } else if (op->call_type == Call::PureExtern && op->name == "exp_f32") {
    assert(op->args.size() == 1);
    Expr e = Internal::halide_exp(op->args[0]);
    value_ = Codegen(e);
  } else if (op->is_intrinsic()) {
    DEBUG() << "is_intrinsic: " << op->name << std::endl;
    assert(false);
    assert(TableContains<mlir::LLVM::LLVMFuncOp>(op->name));
  } else if (op->call_type == Call::Image || op->call_type == Call::Halide) {
    Expr load = BufferLoad::make(op->type,
                                 op->name,
                                 op->args,
                                 op->image,
                                 op->param,
                                 IntImm::make(Int(32), 1),
                                 ModulusRemainder());

		value_ = Codegen(load);
  } else if (op->is_pure()) {
    DEBUG() << "is_pure: " << op->name << std::endl;
    assert(false);
  } else if (op->is_extern()) {
    DEBUG() << "is_extern: " << op->name << std::endl;
    value_ = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getBoolAttr(true));
    return;
  } else {
    DEBUG() << "unknown call type: " << op->call_type << std::endl;
    assert(false);
  }
}

void CodeGen_MLIR::visit(const ProducerConsumer* op) {
  std::string name;
  if (op->is_producer) {
    name = "produce: " + op->name;
  } else {
    name = "consume: " + op->name;
  }
  DEBUG() << name << std::endl;

  op->body.accept(this);
}

void CodeGen_MLIR::visit(const Acquire *) {
  assert(false && "Acquire");
}

// FIXME: predicate store is not implemented yet
void CodeGen_MLIR::visit(const BufferStore* op) {
  DEBUG() << "buffer store name:" << op->name << " dimension: " << op->index.size() << std::endl;
  assert(TableContains<VariableT>(op->name) || global_buffers_.contains(op->name));

  mlir::Value buffer, val_to_store;
  std::vector<mlir::Value> indices;

  if (TableContains<VariableT>(op->name)) {
    VariableT var = TableGet<VariableT>(op->name);
    buffer = var.value;
  } else {
    mlir::memref::GlobalOp global_memref = global_buffers_.get(op->name);
    buffer = builder_.create<mlir::memref::GetGlobalOp>(builder_.getUnknownLoc(),
                                                        global_memref.type(),
                                                        global_memref.getName());
    TablePush<VariableT>(op->name, {buffer, VarType::kNone});
  }

  mlir::MemRefType buf_type = buffer.getType().cast<mlir::MemRefType>();
  if (buf_type.getShape().size() > op->index.size()) {
    std::vector<int64_t> shape_dim(op->index.size(), -1);
    mlir::MemRefType type = mlir::MemRefType::get(shape_dim, buf_type.getElementType());
    mlir::MemRefType shape_type = mlir::MemRefType::get({(int)op->index.size()}, builder_.getI32Type());
    mlir::IntegerAttr alignment = builder_.getI64IntegerAttr(8);
    mlir::Value new_shape = builder_.create<mlir::memref::AllocaOp>(builder_.getUnknownLoc(), shape_type, alignment);
    for (size_t i = 0; i < op->index.size(); ++i) {
      mlir::Value temp = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getI32Type(), builder_.getI32IntegerAttr(-1));
      mlir::Value index = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), builder_.getIndexType(), builder_.getIndexAttr(i));
      builder_.create<mlir::memref::StoreOp>(builder_.getUnknownLoc(), temp, new_shape, mlir::ValueRange({index}));
    }
    buffer = builder_.create<mlir::memref::ReshapeOp>(builder_.getUnknownLoc(), type, buffer, new_shape);
  } else if (buf_type.getShape().size() < op->index.size()) {
    assert(false);
  }

  val_to_store = Codegen(op->value);
  if (val_to_store.getType().isIndex()) {
    val_to_store = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), ConvertToMlirType(op->value.type()), val_to_store);
  }

  if (op->value.type().is_scalar()) {
    int valid_affine_index = 1;
    for (auto ind : op->index) {
      valid_affine_index = valid_affine_index && IsValidAffineExpr(ind);
    }

    if (valid_affine_index) {
      AffineHelper H(*this);
      mlir::AffineMap affine_map = H.GetAffineMap(op->index);
      llvm::SmallVector<mlir::Value, 4> operands = H.GetAffineOperands();

      builder_.create<mlir::AffineStoreOp>(builder_.getUnknownLoc(),
                                           val_to_store,
                                           buffer,
                                           affine_map,
                                           operands);
    } else {
      for (auto& index : op->index) {
        mlir::Value temp = Codegen(index);
        if (!temp.getType().isIndex()) {
          if (!MaybeRegenerateToIndexOrInteger(temp)) {
            temp = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                      temp,
                                                      builder_.getIndexType());
          }
        }
        indices.push_back(temp);
      }
      std::reverse(indices.begin(), indices.end());

      if (mlir::Type elem_type = buffer.getType().cast<mlir::MemRefType>().getElementType()) {
        if (val_to_store.getType() != elem_type) {
          if (val_to_store.getType().isIndex()) {
            val_to_store = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), val_to_store, elem_type);
          }
        }
      }

      builder_.create<mlir::memref::StoreOp>(builder_.getUnknownLoc(),
                                             val_to_store,
                                             buffer,
                                             indices);
    }
  } else {
    auto ramp = op->index[0].as<Ramp>();
    std::vector<bool> is_vectorized_dimension(op->index.size(), false);
    int vectorized_dimensions_count = 0;
    for (size_t i = 0; i < op->index.size(); ++i) {
      if (op->index[i].type().lanes() > 1) {
        is_vectorized_dimension[i] = true;
        vectorized_dimensions_count++;
      }
    }

    std::vector<Expr> halide_index = op->index;

    if (ramp && is_const_one(ramp->stride) && vectorized_dimensions_count == 1) {
      halide_index[0] = ramp->base;
      int valid_affine_index = 1;
      for (auto ind : halide_index) {
        valid_affine_index = valid_affine_index && IsValidAffineExpr(ind);
      }

      assert(256 >= op->value.type().bits());

      if (valid_affine_index) {
        AffineHelper H(*this);
        mlir::AffineMap affine_map = H.GetAffineMap(halide_index);
        llvm::SmallVector<mlir::Value, 4> operands = H.GetAffineOperands();

        builder_.create<mlir::AffineVectorStoreOp>(builder_.getUnknownLoc(),
                                                   val_to_store,
                                                   buffer,
                                                   affine_map,
                                                   operands);
      } else {
        for (size_t i = 0; i < op->index.size(); ++i) {
          indices.push_back(Codegen(halide_index[i]));
        }
        std::reverse(indices.begin(), indices.end());

        for (mlir::Value& index : indices) {
          if (!index.getType().isIndex()) {
            if (!MaybeRegenerateToIndexOrInteger(index)) {
              index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                         index,
                                                         builder_.getIndexType());
            }
          }
        }
        builder_.create<mlir::vector::StoreOp>(builder_.getUnknownLoc(), val_to_store, buffer, indices);
      }
    } else {
      // "non dense(stride 1)" ramp store and vector of index store
      std::vector<mlir::Value> base_indices;
      mlir::Value zero_offset = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(),
                                                                  builder_.getIndexAttr(0));

      if (vectorized_dimensions_count == 1 && op->index[0].type().lanes() > 1) {
        mlir::Value offset_vec;

        if (auto ramp = op->index[0].as<Ramp>()) {
          // gather with ramp
          halide_index[0] = ramp->base;
          for (size_t i = 0; i < halide_index.size(); ++i) {
            mlir::Value temp_index = Codegen(halide_index[i]);
            if (!temp_index.getType().isIndex()) {
              if (!MaybeRegenerateToIndexOrInteger(temp_index)) {
                temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                                temp_index,
                                                                builder_.getIndexType());
              }
            }
            base_indices.push_back(temp_index);
          }

          mlir::Type index_type = ConvertToMlirType(op->index[0].type());
          mlir::Attribute index_attr = builder_.getZeroAttr(index_type);
          offset_vec = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), index_type, index_attr);
          auto stride = ramp->stride;
          for (int i = 0; i < ramp->lanes; ++i) {
            mlir::Value temp_index = Codegen(simplify(i * stride));
            if (temp_index.getType().isIndex()) {
              temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), builder_.getI32Type(), temp_index);
            }
            offset_vec = builder_.create<mlir::vector::InsertElementOp>(builder_.getUnknownLoc(), temp_index, offset_vec, i);
          }
        } else {
          // gather with vector of index
          base_indices.push_back(zero_offset);
          for (size_t i = 1; i < op->index.size(); ++i) {
            mlir::Value temp_index = Codegen(op->index[i]);
            if (!temp_index.getType().isIndex()) {
              if (!MaybeRegenerateToIndexOrInteger(temp_index)) {
                temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                                temp_index,
                                                                builder_.getIndexType());
              }
            }
            base_indices.push_back(temp_index);
          }
          offset_vec = Codegen(op->index[0]);
        }

        int val_lanes = op->value.type().lanes();
        llvm::SmallVector<bool, 4> mask_(val_lanes, true);
        mlir::Attribute mask_attr = builder_.getBoolVectorAttr(mask_);
        mlir::Value mask = builder_.create<mlir::ConstantOp>(builder_.getUnknownLoc(), 
                                                             mlir::VectorType::get(val_lanes, builder_.getI1Type()),
                                                             mask_attr);

        std::reverse(base_indices.begin(), base_indices.end());
        builder_.create<mlir::vector::ScatterOp>(builder_.getUnknownLoc(), buffer, base_indices, offset_vec, mask, val_to_store);
      } else {
        // the vectorized dimension is not the lowest one, or there are more than one vectorized dimensions, so we have to do the store separately
        std::vector<mlir::Value> indices(halide_index.size());
        std::map<int, mlir::Value> index_vecs;

        std::reverse(halide_index.begin(), halide_index.end());
        for (size_t i = 0; i < halide_index.size(); ++i) {
          if (halide_index[i].type().lanes() == 1) {
            mlir::Value temp_index = Codegen(halide_index[i]);
            if (!temp_index.getType().isIndex()) {
              if (!MaybeRegenerateToIndexOrInteger(temp_index)) {
                temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                                temp_index,
                                                                builder_.getIndexType());
              }
            }
            indices[i] = temp_index;
          } else if (!halide_index[i].as<Ramp>()) {
            index_vecs[i] = Codegen(halide_index[i]);
          }
        }

        for (int i = 0; i < op->value.type().lanes(); ++i) {
          for (size_t j = 0; j < halide_index.size(); ++j) {
            if (halide_index[j].type().lanes() > 1) {
              mlir::Value temp_index;
              if (auto ramp = halide_index[j].as<Ramp>()) {
                temp_index = Codegen(simplify(ramp->base + i * ramp->stride));
              } else if (index_vecs.count(j)) {
                temp_index = builder_.create<mlir::vector::ExtractElementOp>(builder_.getUnknownLoc(), index_vecs[j], i);
              } else {
                assert(false && "unrecognized");
              }
              if (!temp_index.getType().isIndex()) {
                if (!MaybeRegenerateToIndexOrInteger(temp_index)) {
                  temp_index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                                  temp_index,
                                                                  builder_.getIndexType());
                }
              }
              indices[j] = temp_index;
            }
          }
          mlir::Value temp_val = builder_.create<mlir::vector::ExtractElementOp>(builder_.getUnknownLoc(), val_to_store, i);
          builder_.create<mlir::memref::StoreOp>(builder_.getUnknownLoc(), temp_val, buffer, indices);
        }
      }
    }
  }
  return;
}

void CodeGen_MLIR::visit(const Store* op) {
  mlir::Value buffer; 

  if (TableContains<VariableT>(op->name)) {
    VariableT var = TableGet<VariableT>(op->name);
    buffer = var.value;
  } else {
    mlir::memref::GlobalOp global_memref = global_buffers_.get(op->name);
    buffer = builder_.create<mlir::memref::GetGlobalOp>(builder_.getUnknownLoc(),
                                                        global_memref.type(),
                                                        global_memref.getName());
    TablePush<VariableT>(op->name, {buffer, VarType::kNone});
  }

  mlir::Value index = Codegen(op->index);
  mlir::Value val_to_store = Codegen(op->value);
  if (val_to_store.getType().isIndex()) {
    val_to_store = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(), ConvertToMlirType(op->value.type()), val_to_store);
  }

  if (!index.getType().isIndex()) {
    if (!MaybeRegenerateToIndexOrInteger(index)) {
      index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                 index,
                                                 builder_.getIndexType());
    }
  }

  builder_.create<mlir::memref::StoreOp>(builder_.getUnknownLoc(),
                                         val_to_store,
                                         buffer,
                                         index);
}

void CodeGen_MLIR::visit(const Fork *) {
  assert(false && "Fork");
}

void CodeGen_MLIR::visit(const Evaluate* op) {
  DEBUG() << "evaluate" << std::endl;

  Codegen(op->value);
}

mlir::Value CodeGen_MLIR::CreateShuffle(mlir::Value a, mlir::Value b, mlir::ArrayRef<int64_t> indices) {
  assert(a.getType() == b.getType());
  if (!a.getType().dyn_cast_or_null<mlir::VectorType>()) {
    mlir::Type vec_type = mlir::VectorType::get(1, a.getType());
    a = builder_.create<mlir::vector::BroadcastOp>(builder_.getUnknownLoc(), vec_type, a);
    b = builder_.create<mlir::vector::BroadcastOp>(builder_.getUnknownLoc(), vec_type, b);
  }

  mlir::Type result_type = mlir::VectorType::get(indices.size(), a.getType().cast<mlir::VectorType>().getElementType());
  mlir::ArrayAttr array_attr = builder_.getI64ArrayAttr(indices);

  return builder_.create<mlir::vector::ShuffleOp>(builder_.getUnknownLoc(), result_type, a, b, array_attr);
}

mlir::Value CodeGen_MLIR::ConcatVector(mlir::ArrayRef<mlir::Value> vectors) {
  assert(vectors.size());

  if (vectors.size() == 1) {
    return vectors[0];
  }

  while (vectors.size() != 1) {
    int num_vectors = vectors.size();
    std::vector<mlir::Value> temp_vectors;
    for (int i = 0; i < num_vectors - 1; i += 2) {
      mlir::Value a = vectors[i];
      mlir::Value b = vectors[i + 1];

      int len_a = a.getType().cast<mlir::VectorType>().getShape()[0];
      int len_b = b.getType().cast<mlir::VectorType>().getShape()[0];
      int vec_len = std::max(len_a, len_b);
      if (len_a > len_b) {
        a = SliceVector(a, 0, len_b);
      } else if (len_a < len_b) {
        b = SliceVector(b, 0, len_a);
      }

      std::vector<int64_t> indices;
      for (int i = 0; i < len_a; ++i) {
        indices.push_back(i);
      }
      for (int i = vec_len; i < vec_len + len_b; ++i) {
        indices.push_back(i);
      }

      temp_vectors.push_back(CreateShuffle(a, b, indices));
    }

    if (num_vectors & 1) {
      temp_vectors.push_back(vectors[num_vectors - 1]);
    }

    vectors = temp_vectors;
  }

  return vectors[0];
}

mlir::Value CodeGen_MLIR::SliceVector(mlir::Value a, int start, int length) {
  std::vector<int64_t> indices(length, -1);
  for (int i = 0; i + start < std::min((int)a.getType().cast<mlir::VectorType>().getShape()[0], length); ++i) {
    indices[i] = i + start;
  }

  return CreateShuffle(a, a, indices);
}

void CodeGen_MLIR::visit(const Shuffle* op) {
  DEBUG() << "shuffle called" << std::endl;
  std::vector<mlir::Value> vectors;
  for (auto v : op->vectors) {
    vectors.push_back(Codegen(v));
  }
  if (vectors.size() == 1) {
    vectors.push_back(vectors.back());
  }

  std::vector<mlir::Value> vectors1;
  std::vector<mlir::Value> vectors2;
  for (size_t i = 0; i < vectors.size() / 2; ++i) {
    vectors1.push_back(vectors[i]);
  }
  for (size_t i = vectors.size() / 2; i < vectors.size(); ++i) {
    vectors2.push_back(vectors[i]);
  }

  mlir::Value concat1 = ConcatVector(vectors1);
  mlir::Value concat2 = ConcatVector(vectors2);

  std::vector<int64_t> indices(op->indices.begin(), op->indices.end());
  value_ = CreateShuffle(concat1, concat2, indices);
  DEBUG() << "shuffle end" << std::endl;
}

void CodeGen_MLIR::visit(const VectorReduce *) {
  assert(false && "VectorReduce");
}
void CodeGen_MLIR::visit(const Prefetch *) {
  assert(false && "Prefetch");
}
void CodeGen_MLIR::visit(const Atomic *) {
  assert(false && "Atomic");
}

void CodeGen_MLIR::visit(const Allocate* op) {
  DEBUG() << "Allocate: " << op->name << " dimension: " << op->extents.size() << std::endl;

  std::vector<int64_t> shape;
  std::vector<mlir::Value> dynamic_sizes;
  for (auto extent : op->extents) {
    if (auto intimm = extent.as<IntImm>()) {
      shape.push_back(intimm->value);
    } else if (auto var = extent.as<Variable>()) {
      if (constants_int_.count(var->name)) {
        shape.push_back(constants_int_[var->name]);
      } else {
        shape.push_back(-1);
        dynamic_sizes.push_back(Codegen(var));
      }
    } else {
      shape.push_back(-1);
      dynamic_sizes.push_back(Codegen(extent));
    }
  }

  for (auto& index : dynamic_sizes) {
    if (!index.getType().isIndex()) {
      if (!MaybeRegenerateToIndexOrInteger(index)) {
        index = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                   index,
                                                   builder_.getIndexType());
      }
    }
  }
  std::reverse(dynamic_sizes.begin(), dynamic_sizes.end());
  std::reverse(shape.begin(), shape.end());

  mlir::IntegerAttr alignment = builder_.getI64IntegerAttr(8);
  mlir::MemRefType memref_type = mlir::MemRefType::get(shape, ConvertToMlirType(op->type));
  if (op->memory_type == MemoryType::Stack) {
    // allocas_.insert(op->name);
    // value_ = builder_.create<mlir::memref::AllocaOp>(builder_.getUnknownLoc(), memref_type, dynamic_sizes, alignment);
    value_ = builder_.create<mlir::memref::AllocOp>(builder_.getUnknownLoc(), memref_type, dynamic_sizes, alignment);
  } else {
    value_ = builder_.create<mlir::memref::AllocOp>(builder_.getUnknownLoc(), memref_type, dynamic_sizes, alignment);
  }

  TablePush(op->name, VariableT({value_, VarType::kNone}));
  
  Codegen(op->body);
}

void CodeGen_MLIR::visit(const Free* op) {
  DEBUG() << "free: " << op->name << std::endl;
  if (allocas_.count(op->name)) {
    return;
  }
  assert(TableContains<VariableT>(op->name) && "variable should exist before it's freed");
  mlir::Value to_be_freed = TableGet<VariableT>(op->name).value;
  builder_.create<mlir::memref::DeallocOp>(builder_.getUnknownLoc(), to_be_freed);
}

} // namespace Internal
} // namespace Halide
