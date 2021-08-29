#ifndef SIMPLIFIER_H
#define SIMPLIFIER_H

#include "CodeGen_MLIR.h"

namespace Halide {
namespace Internal {

class CodeGen_MLIR;

class Simplifier : public VariadicVisitor<Simplifier, Expr, Stmt> {
public:
  Simplifier(CodeGen_MLIR& cg_mlir);
  // Simplifier() = default;
  ~Simplifier() = default;

  Expr mutate(const Expr& e) {
      return Super::dispatch(e);
  }

  Stmt mutate(const Stmt& s) {
      return Super::dispatch(s);
  }

  Expr visit(const IntImm* op);
  Expr visit(const UIntImm* op);
  Expr visit(const FloatImm* op);
  Expr visit(const StringImm* op);

  Expr visit(const Broadcast* op);
  Expr visit(const Cast* op);

  Expr visit(const Variable* op);
  Expr visit(const Add* op);
  Expr visit(const Sub* op);
  Expr visit(const Mul* op);
  Expr visit(const Div* op);
  Expr visit(const Mod* op);
  Expr visit(const Min* op);
  Expr visit(const Max* op);
  Expr visit(const EQ* op);
  Expr visit(const NE* op);
  Expr visit(const LT* op);
  Expr visit(const LE* op);
  Expr visit(const GT* op);
  Expr visit(const GE* op);
  Expr visit(const And* op);
  Expr visit(const Or* op);

  Expr visit(const Not* op);
  Expr visit(const Select* op);
  Expr visit(const Ramp* op);
  Stmt visit(const IfThenElse* op);
  Expr visit(const Load* op);
  Expr visit(const BufferLoad* op);
  Expr visit(const Call* op);
  Expr visit(const Shuffle* op);
  Expr visit(const VectorReduce* op) {assert(false);}
  Expr visit(const Let* op);
  Stmt visit(const LetStmt* op);
  Stmt visit(const AssertStmt* op);
  Stmt visit(const For* op);
  Stmt visit(const Provide* op) {assert(false);}
  Stmt visit(const Store* op);
  Stmt visit(const BufferStore* op);
  Stmt visit(const Allocate* op);
  Stmt visit(const Evaluate* op);
  Stmt visit(const ProducerConsumer* op);
  Stmt visit(const Block* op);
  Stmt visit(const Realize* op) {assert(false);}
  Stmt visit(const Prefetch* op) {assert(false);}
  Stmt visit(const Free* op);
  Stmt visit(const Acquire* op) {assert(false);}
  Stmt visit(const Fork* op) {assert(false);}
  Stmt visit(const Atomic* op) {assert(false);}
private:
  using Super = VariadicVisitor<Simplifier, Expr, Stmt>;
  CodeGen_MLIR& cg_mlir_;
  std::map<std::string, Expr> variables_;
};


} // namespace Internal
} // namespace Halide

#endif
