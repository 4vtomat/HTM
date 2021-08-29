#include "MainFuncGenerator.h"
#include "Halide.h"
#include "tools/halide_image_io.h"

#include "../app_generators/iir_blur_generator.h"

using namespace Utils;

int main(int argc, char* argv[])
{
  std::vector<int64_t> buffer_size = {3, 2560, 1536};
  Halide::Buffer<float> buffer_ = Halide::Tools::load_and_convert_image("rgba.png");
  InputArgument<ArgType::kBuffer> buffer{buffer_, buffer_size};
  InputArgument<ArgType::kFloat, 32> alpha{0.5f};
  OutputArgument<ArgType::kBuffer, 32> output{ArgType::kFloat, buffer_size};
  PushArgs(buffer, alpha, output);

  Utils::Generate<IirBlur>(true /*print out result*/);

  std::cout << "compile done" << std::endl;
}
