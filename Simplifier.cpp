#include <math.h>

#include "Simplifier.h"
#include "Debug.h"

namespace Halide {
namespace Internal {

Simplifier::Simplifier(CodeGen_MLIR& cg_mlir) : cg_mlir_(cg_mlir) {}

Expr Simplifier::visit(const IntImm* op) {
  return IntImm::make(op->type, op->value);
}

Expr Simplifier::visit(const UIntImm* op) {
  return UIntImm::make(op->type, op->value);
}

Expr Simplifier::visit(const FloatImm* op) {
  return FloatImm::make(op->type, op->value);
}

Expr Simplifier::visit(const StringImm* op) {
  return StringImm::make(op->value);
}

Expr Simplifier::visit(const Cast* op) {
  DEBUG() << "simpifying cast" << std::endl;
  Expr value = mutate(op->value);
  if (auto val = value.as<IntImm>()) {
    if (op->type.is_float()) {
      return FloatImm::make(op->type, val->value);
    } else if (op->type.is_int()) {
      return IntImm::make(op->type, (int64_t)val->value);
    } else if (op->type.is_uint()) {
      return UIntImm::make(op->type, (uint64_t)val->value);
    }
  } else if (auto val = value.as<UIntImm>()) {
    if (op->type.is_float()) {
      return FloatImm::make(op->type, val->value);
    } else if (op->type.is_int()) {
      return IntImm::make(op->type, (int64_t)val->value);
    } else if (op->type.is_uint()) {
      return UIntImm::make(op->type, (uint64_t)val->value);
    }
  } else if (auto val = value.as<FloatImm>()) {
    if (op->type.is_float()) {
      return FloatImm::make(op->type, val->value);
    } else if (op->type.is_int()) {
      return IntImm::make(op->type, (int64_t)val->value);
    } else if (op->type.is_uint()) {
      return UIntImm::make(op->type, (uint64_t)val->value);
    }
  }

  return Cast::make(op->type, value);
}

Expr Simplifier::visit(const Variable* op) {
  DEBUG() << "simpifying variable: " << op->name << std::endl;
  if (cg_mlir_.constants_int_.count(op->name)) {
    return IntImm::make(op->type, cg_mlir_.constants_int_[op->name]);
  } else if (cg_mlir_.constants_float_.count(op->name)) {
    return FloatImm::make(op->type, cg_mlir_.constants_float_[op->name]);
  } else if (variables_.count(op->name)) {
    if (auto val = variables_[op->name].as<IntImm>()) {
      return IntImm::make(val->type, val->value);
    } else if (auto val = variables_[op->name].as<FloatImm>()) {
      return FloatImm::make(val->type, val->value);
    } else {
      return variables_[op->name];
    }
  }

  return op;
}


Expr Simplifier::visit(const Add* op) {
  DEBUG() << "simpifying add: " << op->a << " + " << op->b << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return Add::make(a, b);
}

Expr Simplifier::visit(const Sub* op) {
  DEBUG() << "simpifying sub: " << op->a << " - " << op->b << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return Sub::make(a, b);
}

Expr Simplifier::visit(const Mul* op) {
  DEBUG() << "simpifying mul: " << op->a << " * " << op->b << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return Mul::make(a, b);
}

Expr Simplifier::visit(const Div* op) {
  DEBUG() << "simpifying div: " << op->a << " / " << op->b << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return Div::make(a, b);
}

Expr Simplifier::visit(const Mod* op) {
  DEBUG() << "simpifying mod: " << op->a << " % " << op->b << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return Mod::make(a, b);
}

Expr Simplifier::visit(const Min* op) {
  DEBUG() << "simpifying min" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return Min::make(a, b);
}

Expr Simplifier::visit(const Max* op) {
  DEBUG() << "simpifying max" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return Max::make(a, b);
}

Expr Simplifier::visit(const EQ* op) {
  DEBUG() << "simpifying eq" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return EQ::make(a, b);
}

Expr Simplifier::visit(const NE* op) {
  DEBUG() << "simpifying ne" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return NE::make(a, b);
}

Expr Simplifier::visit(const LT* op) {
  DEBUG() << "simpifying lt" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return LT::make(a, b);
}

Expr Simplifier::visit(const LE* op) {
  DEBUG() << "simpifying le" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return LE::make(a, b);
}

Expr Simplifier::visit(const GT* op) {
  DEBUG() << "simpifying gt" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return GT::make(a, b);
}

Expr Simplifier::visit(const GE* op) {
  DEBUG() << "simpifying ge" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return GE::make(a, b);
}

Expr Simplifier::visit(const And* op) {
  DEBUG() << "simpifying and" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return And::make(a, b);
}

Expr Simplifier::visit(const Or* op) {
  DEBUG() << "simpifying or" << std::endl;
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  return Or::make(a, b);
}

Expr Simplifier::visit(const Not* op) {
  DEBUG() << "simpifying not" << std::endl;
  return Not::make(op->a);
}

Expr Simplifier::visit(const Select* op) {
  DEBUG() << "simpifying select" << std::endl;
  Expr t = mutate(op->true_value);
  Expr f = mutate(op->false_value);
  Expr c = mutate(op->condition);

  auto condition = c.as<IntImm>();
  if (condition) {
    if (condition->value) {
      return t;
    } else {
      return f;
    }
  }

  return Select::make(c, t, f);
}

Expr Simplifier::visit(const Let* op) {
  DEBUG() << "simpifying let" << std::endl;
  Expr val = mutate(op->value);
  Expr body = mutate(op->body);
  // variables_[op->name] = val;

  return Let::make(op->name, val, body);
}

Stmt Simplifier::visit(const LetStmt* op) {
  DEBUG() << "simpifying letstmt" << std::endl;
  Expr val = mutate(op->value);
  Stmt body = mutate(op->body);
  // variables_[op->name] = val;

  return LetStmt::make(op->name, val, body);
}

Stmt Simplifier::visit(const Free* op) {
  DEBUG() << "simpifying free" << std::endl;
  return Free::make(op->name);
}

Stmt Simplifier::visit(const Block* op) {
  DEBUG() << "simpifying block" << std::endl;
  return Block::make(mutate(op->first), mutate(op->rest));
}

Stmt Simplifier::visit(const ProducerConsumer* op) {
  DEBUG() << "simpifying producerconsumer" << std::endl;
  return ProducerConsumer::make(op->name, op->is_producer, mutate(op->body));
}

Stmt Simplifier::visit(const Allocate* op) {
  DEBUG() << "simpifying allocate: " << op->name << std::endl;
  std::vector<Expr> extents;
  for (auto ext : op->extents) {
    extents.push_back(mutate(ext));
  }

  return Allocate::make(op->name, op->type, op->memory_type, std::move(extents), mutate(op->condition), mutate(op->body));
}

Expr Simplifier::visit(const BufferLoad* op) {
  DEBUG() << "simpifying bufferload: " << op->name << std::endl;
  std::vector<Expr> indices;
  for (auto index : op->index) {
    indices.push_back(mutate(index));
  }

  return BufferLoad::make(op->type, op->name, std::move(indices), op->image, op->param, mutate(op->predicate), op->alignment);
}

Stmt Simplifier::visit(const BufferStore* op) {
  DEBUG() << "simpifying bufferstore: " << op->name << std::endl;
  std::vector<Expr> indices;
  for (auto index : op->index) {
    indices.push_back(mutate(index));
  }

  return BufferStore::make(op->name, mutate(op->value), std::move(indices), op->param, mutate(op->predicate), op->alignment);
}

Stmt Simplifier::visit(const For* op) {
  DEBUG() << "simpifying for: " << op->name << std::endl;
  return For::make(op->name, mutate(op->min), mutate(op->extent), op->for_type, op->device_api, mutate(op->body));
}

Expr Simplifier::visit(const Call* op) {
  DEBUG() << "simpifying call: " << op->name << std::endl;
  if (op->call_type == Call::Extern) {
    return Call::make(op->type, op->name, op->args, op->call_type, op->func, op->value_index, op->image, op->param);
  }

  std::vector<Expr> args;
  for (auto arg : op->args) {
    args.push_back(mutate(arg));
  }

  return Call::make(op->type, op->name, std::move(args), op->call_type, op->func, op->value_index, op->image, op->param);
}

Stmt Simplifier::visit(const IfThenElse* op) {
  DEBUG() << "simpifying if then else: " << std::endl;
  return IfThenElse::make(mutate(op->condition), mutate(op->then_case), mutate(op->else_case));
}

Stmt Simplifier::visit(const AssertStmt* op) {
  DEBUG() << "simpifying assertstmt" << std::endl;
  return AssertStmt::make(op->condition, op->message);
}

Stmt Simplifier::visit(const Evaluate* op) {
  DEBUG() << "simpifying evaluate" << std::endl;
  return Evaluate::make(mutate(op->value));
}

Expr Simplifier::visit(const Ramp* op) {
  DEBUG() << "simpifying Ramp" << std::endl;
  return Ramp::make(mutate(op->base), mutate(op->stride), op->lanes);
}

Expr Simplifier::visit(const Broadcast* op) {
  DEBUG() << "simpifying Broadcast" << std::endl;
  return Broadcast::make(mutate(op->value), op->lanes);
}

Expr Simplifier::visit(const Shuffle* op) {
  std::vector<Expr> vectors;
  for (size_t i = 0; i < op->vectors.size(); ++i) {
    vectors.push_back(mutate(op->vectors[i]));
  }

  return Shuffle::make(vectors, op->indices);
}

Expr Simplifier::visit(const Load* op) {
  return op;
}

Stmt Simplifier::visit(const Store* op) {
  return op;
}

} // namespace Halide
} // namespace Internal
