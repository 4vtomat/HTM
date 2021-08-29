#ifndef DEBUG_H
#define DEBUG_H
#include <fstream>

namespace {
std::ofstream ofs("/dev/null", std::ofstream::out);
#ifdef DEBUG_MODE
#define DEBUG() std::cerr
#else
#define DEBUG() ofs
#endif
}

#endif
