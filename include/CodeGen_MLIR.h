#ifndef CODEGEN_MLIR_H
#define CODEGEN_MLIR_H

#include <map>
#include <tuple>
#include <type_traits>
#include <vector>
#include <string>
#include <atomic>

#include "mlir/Parser.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/LoopUtils.h"

#include "llvm/Support/Casting.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Linker/Linker.h"


#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/IR/Module.h"

#include "Halide.h"
#include "Simplifier.h"


namespace Utils {
class MainFuncGenerator;
}

namespace Halide {
namespace Internal {

class CodeGen_MLIR : public IRVisitor {
public:
  CodeGen_MLIR(const Target&);
  // CodeGen_MLIR(const CodeGen_MLIR&) = default;
  // CodeGen_MLIR(CodeGen_MLIR&&) = default;
  ~CodeGen_MLIR() override = default;

  enum class VarType {kAffineSym,
                      kAffineDim,
                      kScfDim,
                      kInputArg,
                      kNormal,
                      kNone};

  using VariableT = struct {mlir::Value value;
                            VarType var_type = VarType::kNone;
                            mlir::Type type = nullptr;
                            uint8_t dimensions = 0;
                            operator bool() {return var_type != VarType::kNone;}};

  template <typename T>
  void TablePush(const std::string&, T) noexcept;

  template <typename T>
  void TablePop(const std::string&) noexcept;

  template <typename T>
  T TableGet(const std::string&) noexcept;

  template <typename T>
  bool TableContains(const std::string&) noexcept;

  mlir::OpBuilder& GetBuilder() noexcept;
  mlir::MLIRContext& GetContext() noexcept;

  mlir::LLVM::LLVMStructType CreateLlvmStructType(std::string) noexcept;
  void DeclareHalideFunction(const std::string&, mlir::Type, std::vector<mlir::Type>) noexcept;
  mlir::Type ConvertToMlirType(const Type&) noexcept;
  mlir::Type ConvertHalideTypeTToMlirType(const halide_type_t&) noexcept;

  void compile(const Module& M);

protected:
  virtual void CompileFunction(const LoweredFunc&) noexcept;
  virtual void CompileBuffer(const Buffer<>&) noexcept;

  mlir::LLVM::LLVMStructType halide_buffer_t_type,
                             halide_type_t_type,
                             halide_dimension_t_type,
                             halide_metadata_t_type,
                             halide_argument_t_type,
                             halide_scalar_value_t_type,
                             halide_device_interface_t_type,
                             halide_pseudostack_slot_t_type,
                             halide_semaphore_t_type,
                             halide_semaphore_acquire_t_type,
                             halide_parallel_task_t_type;

  template<typename T>
  void GenerateCmpOp(const T*, mlir::CmpFPredicate, mlir::CmpIPredicate) noexcept;

  template<typename T, typename, typename>
  void GenerateArithOp(const T*) noexcept;

  mlir::Value Codegen(const Expr&) noexcept;
  void Codegen(const Stmt&) noexcept;

  bool MaybeRegenerateToIndexOrInteger(mlir::Value&, mlir::Value&, bool to_integer = false) noexcept;
  bool MaybeRegenerateToIndexOrInteger(mlir::Value& v, bool to_integer = false, mlir::Type = nullptr) noexcept;

  int IsValidAffineExpr(const Expr&);
  bool IsValidAffineVariable(mlir::Value v);

  mlir::Value CreateShuffle(mlir::Value a, mlir::Value b, mlir::ArrayRef<int64_t> indices);
  mlir::Value ConcatVector(mlir::ArrayRef<mlir::Value> vectors);
  mlir::Value SliceVector(mlir::Value a, int start, int length);

  using IRVisitor::visit;
  void visit(const IntImm*) override;
  void visit(const UIntImm*) override;
  void visit(const FloatImm*) override;
  void visit(const StringImm*) override;
  void visit(const Variable*) override;

  void visit(const IfThenElse*) override;
  void visit(const For*) override;

  void visit(const Add*) override;
  void visit(const Sub*) override;
  void visit(const Mul*) override;
  void visit(const Div*) override;
  void visit(const Mod*) override;

  void visit(const EQ*) override;
  void visit(const NE*) override;
  void visit(const LT*) override;
  void visit(const LE*) override;
  void visit(const GT*) override;
  void visit(const GE*) override;

  void visit(const Cast*) override;

  void visit(const Min*) override;
  void visit(const Max*) override;
  void visit(const And*) override;
  void visit(const Or*) override;
  void visit(const Not*) override;
  void visit(const Select*) override;
  void visit(const Load*) override;
  void visit(const BufferLoad*) override;
  void visit(const Ramp*) override;
  void visit(const Broadcast*) override;
  void visit(const Call*) override;
  void visit(const Let*) override;
  void visit(const LetStmt*) override;
  void visit(const AssertStmt*) override;
  void visit(const ProducerConsumer*) override;
  void visit(const Acquire*) override;
  void visit(const Store*) override;
  void visit(const BufferStore*) override;

  void visit(const Allocate*) override;
  void visit(const Free*) override;

  void visit(const Block*) override;
  void visit(const Fork*) override;
  void visit(const Evaluate*) override;
  void visit(const Shuffle*) override;
  void visit(const VectorReduce*) override;
  void visit(const Prefetch*) override;
  void visit(const Atomic*) override;

private:
  void Init() noexcept;
  void InitLlvmModules() noexcept;
  void InitDialect() noexcept;
  void InitHalideStructs() noexcept;
  void InitHalideFunctions() noexcept;

  Target target_;

  Scope<VariableT> symbol_table_;
  Scope<mlir::Type> type_table_;
  Scope<mlir::LLVM::LLVMFuncOp> func_table_;
  Scope<mlir::memref::GlobalOp> global_buffers_;
  // store the expressions that are not evaluated until they're used
  Scope<Expr> lazy_evals_;
  std::vector<std::string> func_names_;
  std::map<std::string, std::vector<mlir::Type>> func_args_;

  std::map<std::string, int64_t> constants_int_;
  std::map<std::string, double> constants_float_;

  std::set<std::string> allocas_;

  mlir::Value value_;
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;

friend class AffineHelper;
friend class Simplifier;
friend class Utils::MainFuncGenerator;
};

} // namespace Internal
} // namespace Halide

#endif
