#include "AffineHelper.h"
#include "Debug.h"

namespace Halide {
namespace Internal {

AffineHelper::AffineHelper(CodeGen_MLIR& Codegen_mlir)
    : IRPrinter(DEBUG()), cg_mlir_(Codegen_mlir), builder_(cg_mlir_.GetBuilder()) {}

mlir::AffineMap AffineHelper::GetAffineMap(llvm::ArrayRef<Expr> exprs) {
  llvm::SmallVector<mlir::AffineExpr, 4> affine_exprs;
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    affine_exprs.push_back(Codegen(*it));
  }

  DEBUG() << "dimensions: " << dimensions_.size() << std::endl;
  DEBUG() << "symbols: " << symbols_.size() << std::endl;
  operands_.append(dimensions_.begin(), dimensions_.end());
  operands_.append(symbols_.begin(), symbols_.end());

  return mlir::AffineMap::get(dimensions_.size(),
                              symbols_.size(),
                              affine_exprs,
                              &cg_mlir_.GetContext());
}

llvm::SmallVector<mlir::Value, 4> AffineHelper::GetAffineOperands() {
  for (mlir::Value& op : operands_) {
    if (op && !op.getType().isIndex()) {
      if (!cg_mlir_.MaybeRegenerateToIndexOrInteger(op)) {
        op = builder_.create<mlir::IndexCastOp>(builder_.getUnknownLoc(),
                                                op,
                                                builder_.getIndexType());
      }
    }
  }
  return operands_;
}

void AffineHelper::VarPush(const std::string& name, mlir::AffineExpr expr) {
  symbol_table_.push(name, expr);
}

void AffineHelper::VarPop(const std::string& name) {
  // symbol_table_.pop(name);
}

mlir::AffineExpr AffineHelper::VarGet(const std::string& name) {
  if (symbol_table_.count(name)) {
    return symbol_table_.get(name);
  }
  
  return nullptr;
}

bool AffineHelper::VarContains(const std::string& name) {
  return symbol_table_.count(name);
}

mlir::AffineExpr AffineHelper::Codegen(const Expr& e) {
  assert(e.defined());

  value_ = nullptr;

  e.accept(this);

  return value_;
}

void AffineHelper::Codegen(const Stmt& s) {
  s.accept(this);
}

void AffineHelper::visit(const IntImm* op) {
  DEBUG() << "affine intimm: " << op->value << std::endl;
  value_ = builder_.getAffineConstantExpr(op->value);
}

void AffineHelper::visit(const UIntImm* op) {
  DEBUG() << "affine uintimm: " << op->value << std::endl;
  value_ = builder_.getAffineConstantExpr(op->value);
}

void AffineHelper::visit(const FloatImm* op) {
  DEBUG() << "affine floatimm: " << op->value << std::endl;
  value_ = builder_.getAffineSymbolExpr(symbols_.size());
  cg_mlir_.visit(op);
  symbols_.push_back(cg_mlir_.value_);
}

void AffineHelper::visit(const Variable* op) {
  DEBUG() << "affine variable: " << op->name << std::endl;
  if (VarContains(op->name)) {
    value_ = VarGet(op->name);
    return;
  }

  if (cg_mlir_.TableContains<CodeGen_MLIR::VariableT>(op->name) && !cg_mlir_.lazy_evals_.contains(op->name)) {
    CodeGen_MLIR::VariableT var = cg_mlir_.TableGet<CodeGen_MLIR::VariableT>(op->name);
    if (var.var_type == CodeGen_MLIR::VarType::kAffineSym) {
			mlir::Value v = var.value;
      if (!v.getType().isIndex()) {
        if (!cg_mlir_.MaybeRegenerateToIndexOrInteger(v)) {
          assert(false);
        }
      }

      value_ = builder_.getAffineSymbolExpr(symbols_.size());
      symbols_.push_back(v);
    } else {
      value_ = builder_.getAffineDimExpr(dimensions_.size());
      dimensions_.push_back(var.value);
    }
  } else {
    Expr var = cg_mlir_.lazy_evals_.get(op->name);
		if (var.as<Variable>() ||
        var.as<Add>() ||
        var.as<Sub>() ||
        var.as<Mul>() ||
        var.as<Div>() ||
        var.as<Mod>() ||
        var.as<IntImm>() ||
        var.as<UIntImm>()) {
			value_ = Codegen(var);
		} else {
      DEBUG() << "lazy: " << var << std::endl;
			mlir::Value v = cg_mlir_.Codegen(var);
      if (!v.getType().isIndex()) {
        if (!cg_mlir_.MaybeRegenerateToIndexOrInteger(v)) {
          assert(false);
        }
      }

			value_ = builder_.getAffineSymbolExpr(symbols_.size());
			symbols_.push_back(v);
		}
  }

  VarPush(op->name, value_);
}

void AffineHelper::visit(const Add* op) {
  DEBUG() << "affine add: " << op->a << " + " << op->b << std::endl;
  assert(op->a.defined() && op->b.defined());

  value_ = Codegen(op->a) + Codegen(op->b);
  DEBUG() << "affine add done: " << op->a << " + " << op->b << std::endl;
}

void AffineHelper::visit(const Sub* op) {
  DEBUG() << "affine sub: " << op->a << " - " << op->b << std::endl;
  assert(op->a.defined() && op->b.defined());

  value_ = Codegen(op->a) - Codegen(op->b);
}

void AffineHelper::visit(const Mul* op) {
  DEBUG() << "affine mul: " << op->a << " * " << op->b << std::endl;
  assert(op->a.defined() && op->b.defined());

  value_ = Codegen(op->a) * Codegen(op->b);
  DEBUG() << "affine mul done: " << op->a << " * " << op->b << std::endl;
}

void AffineHelper::visit(const Div* op) {
  DEBUG() << "affine div: " << op->a << " / " << op->b << std::endl;
  assert(op->a.defined() && op->b.defined());

  value_ = Codegen(op->a).floorDiv(Codegen(op->b));
  DEBUG() << "affine div done: " << op->a << " / " << op->b << std::endl;
}

void AffineHelper::visit(const Mod* op) {
  DEBUG() << "affine mod: " << op->a << " % " << op->b << std::endl;
  assert(op->a.defined() && op->b.defined());

  value_ = Codegen(op->a) % (Codegen(op->b));
  DEBUG() << "affine mod done: " << op->a << " % " << op->b << std::endl;
}

void AffineHelper::visit(const Min* op) {
  assert(false);
  DEBUG() << "affine min" << std::endl;
  mlir::AffineMap affine_map;
  llvm::SmallVector<mlir::Value, 4> operands;
  AffineHelper H(cg_mlir_);
  affine_map = H.GetAffineMap({op->a, op->b});
  operands = H.GetAffineOperands();
  DEBUG() << "affine minnnnnnnnnnnnnnnn" << std::endl;

  value_ = builder_.getAffineSymbolExpr(symbols_.size());
  symbols_.push_back(builder_.create<mlir::AffineMinOp>(builder_.getUnknownLoc(), affine_map, operands));
  DEBUG() << "affine min done" << std::endl;
}

void AffineHelper::visit(const Max* op) {
  assert(false);
  DEBUG() << "affine max" << std::endl;
  mlir::AffineMap affine_map;
  llvm::SmallVector<mlir::Value, 4> operands;
  AffineHelper H(cg_mlir_);
  affine_map = H.GetAffineMap({op->a, op->b});
  operands = H.GetAffineOperands();

  value_ = builder_.getAffineSymbolExpr(symbols_.size());
  symbols_.push_back(builder_.create<mlir::AffineMaxOp>(builder_.getUnknownLoc(), affine_map, operands));
  DEBUG() << "affine max done" << std::endl;
}

void AffineHelper::visit(const StringImm* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const IfThenElse* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const For* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const EQ* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const NE* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const LT* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const LE* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const GT* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const GE* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const And* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Or* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Not* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Load* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const BufferLoad* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Ramp* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Broadcast* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const AssertStmt* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const ProducerConsumer* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Acquire* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Store* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const BufferStore* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Allocate* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Free* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Block* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Fork* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Evaluate* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Shuffle* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const VectorReduce* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Prefetch* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Atomic* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Let* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const LetStmt* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Select* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Call* op) {
  assert(false && "invalid operation for affine expression");
}

void AffineHelper::visit(const Cast* op) {
  assert(false && "invalid operation for affine expression");
}


} // namespace Internal
} // namespace Halide
