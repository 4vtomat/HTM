#ifndef MAIN_FUNC_GENERATOR_H
#define MAIN_FUNC_GENERATOR_H

#include "CodeGen_MLIR.h"

namespace Utils {

enum class ArgType {
  kInt,
  kUInt,
  kFloat,
  kBuffer
};

struct BaseValue {
  const ArgType type;
  BaseValue(ArgType t) : type(t) {}
  bool is(ArgType t) {
    return t == type;
  }
};

struct IntArg : BaseValue {
  int arg_bits;
  int64_t value;
  IntArg(int bits, int64_t val)
      : BaseValue(ArgType::kInt), arg_bits(bits), value(val) {}
};

struct UIntArg : BaseValue {
  int arg_bits;
  uint64_t value;
  UIntArg(int bits, uint64_t val)
      : BaseValue(ArgType::kUInt), arg_bits(bits), value(val) {}
};

struct FloatArg : BaseValue {
  int arg_bits;
  double value;
  FloatArg(int bits, double val)
      : BaseValue(ArgType::kFloat), arg_bits(bits), value(val) {}
};

struct BufferArg : BaseValue {
  halide_buffer_t* value;
  std::vector<int64_t> shape;
  ArgType elem_type;
  int elem_bits;
  mlir::memref::GlobalOp global_memref;

  BufferArg(halide_buffer_t* val, ArgType type)
      : BaseValue(ArgType::kBuffer), value(val), elem_type(type) {}

  template <typename T>
  BufferArg(halide_buffer_t* val, std::vector<T>&& shape, ArgType type)
      : BaseValue(ArgType::kBuffer), value(val), shape(std::move(shape)), elem_type(type) {
    for (auto s : shape) {
      this->shape.push_back(s);
    }
  }
};

struct ArgumentBase {
  ArgType arg_type;
  BaseValue* base_val;
  bool is_input;

  ArgumentBase(ArgType type, int bits, int64_t val, bool is_input = true)
      : arg_type(type), is_input(is_input) {
    assert(type == ArgType::kInt);
    base_val = new IntArg{bits, val};
  }

  ArgumentBase(ArgType type, int bits, uint64_t val, bool is_input = true)
      : arg_type(type), is_input(is_input) {
    assert(type == ArgType::kUInt);
    base_val = new UIntArg{bits, val};
  }

  ArgumentBase(ArgType type, int bits, double val, bool is_input = true)
      : arg_type(type), is_input(is_input) {
    assert(type == ArgType::kFloat);
    base_val = new FloatArg{bits, val};
  }

  template <typename T>
  ArgumentBase(ArgType type, halide_buffer_t* val = nullptr, std::vector<T> shape = {}, ArgType elem_type = ArgType::kBuffer, bool is_input = true)
      : arg_type(type), is_input(is_input) {
    assert(type == ArgType::kBuffer);
    base_val = new BufferArg{val, std::move(shape), elem_type};
  }

  ArgumentBase(ArgType type, halide_buffer_t* val = nullptr, std::initializer_list<int64_t> shape = {}, ArgType elem_type = ArgType::kBuffer, bool is_input = true)
      : arg_type(type), is_input(is_input) {
    assert(type == ArgType::kBuffer);
    base_val = new BufferArg{val, elem_type};
    ((BufferArg*)base_val)->shape.insert(((BufferArg*)base_val)->shape.end(), shape.begin(), shape.end());
  }
};

template <ArgType type, int bits = -1, typename = void>
struct InputArgument : public ArgumentBase {};

template <ArgType type, int bits = -1, typename = void>
struct OutputArgument : public ArgumentBase {};

template <int bits>
struct InputArgument<ArgType::kInt, bits> : public ArgumentBase {
  InputArgument(int64_t val)
      : ArgumentBase(ArgType::kInt, bits, val) {}
};

template <int bits>
struct InputArgument<ArgType::kUInt, bits> : public ArgumentBase {
  InputArgument(uint64_t val)
      : ArgumentBase(ArgType::kUInt, bits, val) {}
};

template <int bits>
struct InputArgument<ArgType::kFloat, bits> : public ArgumentBase {
  InputArgument(double val)
      : ArgumentBase(ArgType::kFloat, bits, val) {}
};

template <int bits, typename T>
struct InputArgument<ArgType::kBuffer, bits, T> : public ArgumentBase {
  InputArgument(Halide::Buffer<T> val, std::vector<int64_t> shape = {})
      : ArgumentBase(ArgType::kBuffer, val.raw_buffer(), shape) {}
};

template <int bits>
struct OutputArgument<ArgType::kInt, bits> : public ArgumentBase {
  OutputArgument()
      : ArgumentBase(ArgType::kInt, bits, (int64_t)1, false) {}
};

template <int bits>
struct OutputArgument<ArgType::kUInt, bits> : public ArgumentBase {
  OutputArgument()
      : ArgumentBase(ArgType::kUInt, bits, (uint64_t)1, false) {}
};

template <int bits>
struct OutputArgument<ArgType::kFloat, bits> : public ArgumentBase {
  OutputArgument()
      : ArgumentBase(ArgType::kFloat, bits, (double)1, false) {}
};

template <int bits, typename T>
struct OutputArgument<ArgType::kBuffer, bits, T> : public ArgumentBase {

  template <typename S,
            typename std::enable_if<std::is_integral<S>::value, bool>::type = true>
  OutputArgument(ArgType elem_type, std::vector<S> shape)
      : ArgumentBase(ArgType::kBuffer, nullptr, std::move(shape), elem_type, false) {
    assert(~bits && "Buffer element width must be specified");
    auto buf = (BufferArg*)base_val;
    buf->elem_bits = bits;
  }

  OutputArgument(ArgType elem_type, std::initializer_list<int64_t> shape)
      : ArgumentBase(ArgType::kBuffer, nullptr, std::move(shape), elem_type, false) {
    assert(~bits && "Buffer element width must be specified");
    auto buf = (BufferArg*)base_val;
    buf->elem_bits = bits;
  }

  OutputArgument(Halide::Buffer<T> val)
      : ArgumentBase(ArgType::kBuffer, nullptr, std::vector<int64_t>{}, ArgType::kBuffer, false) {
    auto buf = (BufferArg*)base_val;
    for (int i = val.dimensions() - 1; i >= 0; --i) {
      buf->shape.push_back(val.dim(i).extent());
    }

    buf->elem_bits = val.type().bits();
    switch (val.type().code()) {
      case halide_type_code_t::halide_type_int:
        buf->elem_type = ArgType::kInt;
        break;
      case halide_type_code_t::halide_type_uint:
        buf->elem_type = ArgType::kUInt;
        break;
      case halide_type_code_t::halide_type_float:
        buf->elem_type = ArgType::kFloat;
        break;
      default:
        assert(0);
    }
  }
};

class MainFuncGenerator {
public:
  MainFuncGenerator(Halide::Internal::CodeGen_MLIR& cg);
  void GenerateMainFunc() noexcept;

  template <typename Arg = ArgumentBase, typename... Remain>
  static void PushArgs(Arg a, Remain... r) {
    input_count += a.is_input? 1 : 0;
    arguments_.push_back(a);
    PushArgs(r...);
  }
  template <typename Arg = ArgumentBase>
  static void PushArgs(Arg a) {
    input_count += a.is_input? 1 : 0;
    arguments_.push_back(a);
  }

  static bool print_result;

protected:
  void DeclareUtilities() noexcept;
  mlir::memref::GlobalOp CreateGlobalBuffer(mlir::Type buf_type, mlir::Attribute&& buf_attr) noexcept;
  void ProcessInputBuffers() noexcept;
  void GenerateBody() noexcept;
  void GeneratePrintMemref(mlir::Value memref, ArgType type) noexcept;
  void GeneratePrintMemref(std::vector<mlir::Value>& memrefs) noexcept;
  mlir::Type ConvertToMlirType(ArgumentBase&) noexcept;
  std::string GenerateRandomName() noexcept;

private:
  Halide::Internal::CodeGen_MLIR& cg_mlir_;
  mlir::MLIRContext& context_;
  mlir::OpBuilder& builder_;
  mlir::ModuleOp& module_;
  static std::vector<ArgumentBase> arguments_;
  static int input_count;

template <ArgType type, int bits, typename T>
friend struct Argument;
};

template <typename Arg = ArgumentBase, typename... Remain>
void PushArgs(Arg a, Remain... r) {
  MainFuncGenerator::PushArgs(a, r...);
}

template <typename Arg = ArgumentBase>
void PushArgs(Arg a) {
  MainFuncGenerator::PushArgs(a);
}

template <typename T>
void Generate(bool print_result = true) {
  MainFuncGenerator::print_result = print_result;
  std::map<Halide::Output, std::string> single = {{Halide::Output::mlir, "mlir"}};
  auto gen = T::create(Halide::GeneratorContext(Halide::get_target_from_environment()));
  auto m = gen->build_module();
  m.compile(single);
}

}

#endif
