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

static bool rejected(const std::string& spec) {
  try {
    FFTConfig fft{spec};
    (void) fft.maxBpw();
  } catch (...) {
    return true;
  }
  return false;
}

int main() {
  // The variant digits index the bpw table: W 0-2, M 0-1, H 0-2.  An out-of-range digit must be refused
  // when the spec is parsed (in NDEBUG builds it used to read past the end of the table).
  for (const char* bad : {"256:2:256:303", "256:2:256:222", "256:2:256:120", "256:2:256:1000"}) {
    if (!rejected(bad)) {
      std::cerr << "variant check: '" << bad << "' was accepted\n";
      return 1;
    }
  }
  // A carry other than 0 or 1 is not a carry.
  if (!rejected("256:2:256:212:2")) {
    std::cerr << "carry ':2' was accepted\n";
    return 1;
  }
  for (const char* good : {"256:2:256:101", "256:2:256:212", "256:2:256:212:0", "256:2:256:101:1", "1:512:8:512:202"}) {
    if (rejected(good)) {
      std::cerr << "valid spec '" << good << "' was rejected\n";
      return 1;
    }
  }
  // -tune / -info shapes (FFTShape::multiSpec) skip FFTConfig's spec checks.  A too-small shape drove middle to 0 in the
  // FFTShape bpw fallback, which then looped forever ("-tune -fft 128:2:128" hung); out-of-range shapes must be refused.
  for (const char* bad : {"128:2:128", "64:1:64", "2K:2:256", "256:0:256", "256:17:256", "256:2:128", "8K:2:256", "1:128:4:128"}) {
    bool refused = false;
    try { (void) FFTShape::multiSpec(bad); } catch (const char*) { refused = true; }
    if (!refused) {
      std::cerr << "shape check: '" << bad << "' was accepted\n";
      return 1;
    }
  }
  // Every shape FFTConfig accepts still builds, for every FFT type (the bpw fallback constructs intermediate shapes).
  for (int t : {0, 1, 2, 3, 4, 50, 51, 52, 53}) {
    for (u32 w : {256u, 512u, 1024u, 4096u}) {
      for (u32 m = 1; m <= 16; ++m) {
        for (u32 h : {256u, 512u, 1024u}) {
          try {
            FFTShape shape{(FFT_TYPES) t, w, m, h};
            if (shape.maxBpw() <= 0) { std::cerr << "shape " << shape.spec() << " has no bpw\n"; return 1; }
          } catch (...) {
            std::cerr << "valid shape " << t << ":" << w << ":" << m << ":" << h << " was rejected\n";
            return 1;
          }
        }
      }
    }
  }
  std::cout << "ok\n";
  return 0;
}
