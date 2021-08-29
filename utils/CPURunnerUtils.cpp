#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

extern "C" void printI32(int32_t i) { fprintf(stdout, "%" PRId32, i); }
extern "C" void printIU32(uint32_t i) { fprintf(stdout, "%" PRIu32, i); }
extern "C" void printI16(int16_t i) { fprintf(stdout, "%" PRId16, i); }
extern "C" void printIU16(uint16_t i) { fprintf(stdout, "%" PRIu16, i); }
extern "C" void printI8(int8_t i) { fprintf(stdout, "%" PRId8, i); }
extern "C" void printIU8(uint8_t i) { fprintf(stdout, "%" PRIu8, i); }
extern "C" void printSpace() { fputc(' ', stdout); }

extern "C" void print_time_elapsed(double elapsed) {
  fprintf(stderr, "time elapsed: %lf \n", elapsed);
}
