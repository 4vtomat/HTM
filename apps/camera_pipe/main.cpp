#include "MainFuncGenerator.h"
#include "Halide.h"
#include "tools/halide_image_io.h"

#include "../app_generators/camera_pipe_generator.h"

using namespace Utils;

int main(int argc, char* argv[])
{
  std::vector<int64_t> buffer_size = {1968, 2592};
  Halide::Buffer<uint16_t> buffer_ = Halide::Tools::load_and_convert_image("bayer_raw.png");
  float _matrix_3200[3][4] = {{1.6697f, -0.2693f, -0.4004f, -42.4346f},
                             {-0.3576f, 1.0615f, 1.5949f, -37.1158f},
                             {-0.2175f, -1.8751f, 6.9640f, -26.6970f}};

  float _matrix_7000[3][4] = {{2.2997f, -0.4478f, 0.1706f, -39.0923f},
                             {-0.3826f, 1.5906f, -0.2080f, -25.4311f},
                             {-0.0888f, -0.7344f, 2.2832f, -20.0826f}};

  InputArgument<ArgType::kBuffer> buffer{buffer_};
  Halide::Buffer m3200_{_matrix_3200};
  InputArgument<ArgType::kBuffer> m3200{m3200_};
  Halide::Buffer m7000_{_matrix_7000};
  InputArgument<ArgType::kBuffer> m7000{m7000_};
  InputArgument<ArgType::kFloat, 32> color_temp{3700};
  InputArgument<ArgType::kFloat, 32> gamma{2};
  InputArgument<ArgType::kFloat, 32> contrast{50};
  InputArgument<ArgType::kFloat, 32> sharpen{1};
  InputArgument<ArgType::kInt, 32> black_level{25};
  InputArgument<ArgType::kInt, 32> white_level{1023};
  Halide::Buffer<uint8_t> output_{2560, 1920, 3};
  OutputArgument<ArgType::kBuffer> output{output_};
  PushArgs(buffer, m3200, m7000, color_temp, gamma, contrast, sharpen, black_level, white_level, output);

  Generate<CameraPipe>();

  std::cout << "compile done" << std::endl;
}
