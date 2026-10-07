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
  std::cout << "ok\n";
  return 0;
}
