#include "FFTConfig.h"
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal host-only stubs needed by FFTConfig.cpp (see type4_pfa9_plan_test.cpp).
void log(const char*, ...) {}
std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, delimiter)) result.push_back(part);
  return result;
}

static void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

int main() {
  // An explicit ":0" spec requests a 32-bit carry for any FFT/NTT type.  The
  // kernels cannot hold the carries once EXP / NWORDS reaches 19 (the build
  // fails with #error in carryutil.cl); the host refuses it earlier.
  for (const char* base : {"512:8:512:101", "1:512:8:512:101", "4:512:8:512:202"}) {
    FFTConfig c32{std::string(base) + ":0"};
    FFTConfig c64{std::string(base) + ":1"};
    FFTConfig cauto{std::string(base)};
    require(c32.carry == CARRY_32 && c64.carry == CARRY_64 && cauto.carry == CARRY_AUTO, "carry spec parse");
    const u64 n = c32.size();
    require(!c32.carry32TooWide(n * 18 + n - 1), "carry32 must be allowed up to 18 bits per word");
    require(c32.carry32TooWide(n * 19), "carry32 must be refused at 19 bits per word");
    require(c32.carry32TooWide(n * 25), "carry32 must be refused at 25 bits per word");
    require(!c64.carry32TooWide(n * 25), "carry64 is not subject to the carry32 limit");
    require(!cauto.carry32TooWide(n * 25), "auto carry is not subject to the carry32 limit");
    std::cout << base << " ok\n";
  }
  std::cout << "ok\n";
  return 0;
}
