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
  // An explicit ":0" spec requests a 32-bit carry for any FFT/NTT type.  The host refuses it exactly where CARRY_AUTO
  // would switch to a 64-bit carry (FFTShape::needsLargeCarry, bits per word above carry32BPW()), which can be well
  // below 19 bpw; the kernels additionally fail the build with #error at EXP / NWORDS >= 19 (carryutil.cl).
  for (const char* base : {"512:8:512:101", "1:512:8:512:101", "4:512:8:512:202"}) {
    FFTConfig c32{std::string(base) + ":0"};
    FFTConfig c64{std::string(base) + ":1"};
    FFTConfig cauto{std::string(base)};
    require(c32.carry == CARRY_32 && c64.carry == CARRY_64 && cauto.carry == CARRY_AUTO, "carry spec parse");
    const u64 n = c32.size();
    const double limit = c32.shape.carry32BPW();
    require(limit > 18.0 && limit < 19.0, "test shapes must have carry32BPW() between 18 and 19");
    const u64 atLimit = u64(limit * n);
    require(!c32.shape.needsLargeCarry(atLimit) && !c32.carry32TooWide(atLimit), "carry32 must be allowed at carry32BPW");
    require(!c32.carry32TooWide(n * 18 - 1), "carry32 must be allowed below 18 bits per word");
    // Above carry32BPW() but below 19 bpw: AUTO needs CARRY64, so the explicit 32-bit carry must be refused too.
    const u64 between = u64((limit + 19.0) / 2 * n);
    require(between / n == 18, "regression exponent must be in the 18.x bpw range");
    require(c32.shape.needsLargeCarry(between), "AUTO must require CARRY64 between carry32BPW and 19 bpw");
    require(c32.carry32TooWide(between), "carry32 must be refused above carry32BPW even below 19 bpw");
    require(c32.carry32TooWide(atLimit + n / 100), "carry32 must be refused just above carry32BPW");
    require(c32.carry32TooWide(n * 19), "carry32 must be refused at 19 bits per word");
    require(c32.carry32TooWide(n * 25), "carry32 must be refused at 25 bits per word");
    require(!c64.carry32TooWide(n * 25), "carry64 is not subject to the carry32 limit");
    require(!cauto.carry32TooWide(n * 25), "auto carry is not subject to the carry32 limit");
    // The host gate must never be looser than the kernel #error (EXP / NWORDS >= 19, integer division).
    for (u64 e : {n * 19 - 1, n * 19, n * 19 + 1, n * 20}) {
      require(c32.carry32TooWide(e) || e / n < 19, "host gate must cover the kernel #error");
    }
    std::cout << base << " ok\n";
  }
  // Reproducer: 4M words (1:512:8:512:101); exponent 78852915 is ~18.80 bpw, above carry32BPW() ~18.70 but below 19.
  {
    FFTConfig c32{std::string("1:512:8:512:101:0")};
    require(c32.size() == 4194304, "reproducer shape size");
    require(c32.shape.needsLargeCarry(78852915), "reproducer: AUTO requires CARRY64");
    require(c32.carry32TooWide(78852915), "reproducer: explicit :0 must be refused");
    require(!c32.carry32TooWide(78000000), "reproducer: 18.6 bpw is still fine for :0");
  }
  // Every maxExp() the tuner may time with an explicit 32-bit carry (FP64 FFTs) must pass the gate.
  for (const FFTShape& shape : FFTShape::allShapes()) {
    if (shape.fft_type != FFT64) continue;
    FFTConfig c{shape, 101, CARRY_32};
    require(!c.carry32TooWide(c.maxExp()), "maxExp() of an explicit 32-bit FP64 FFT must pass the gate");
  }
  std::cout << "ok\n";
  return 0;
}
