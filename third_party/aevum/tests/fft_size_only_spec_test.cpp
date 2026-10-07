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
  // A bare FFT size is a documented explicit spec.  The 1-component branch of
  // the FFTConfig(spec) dispatch used to fall into the 3/4/5-component chain
  // and throw "FFT spec" after having parsed it.
  for (const char* size : {"4M", "256K", "1M"}) {
    try {
      FFTConfig fft{size};
      require(fft.size() == FFTShape::multiSpec(size).front().size(), "size-only spec resolved to the wrong shape");
      std::cout << size << " -> " << fft.spec() << "\n";
    } catch (...) {
      std::cerr << "size-only spec '" << size << "' threw\n";
      return 1;
    }
  }

  // The 3/4/5-component forms must keep working.
  require(FFTConfig{"256:2:256"}.size() == 256 * 2 * 256 * 2, "3-component spec");
  require(FFTConfig{"256:2:256:212"}.variant == 212, "4-component spec");
  require(FFTConfig{"256:2:256:212:1"}.carry == CARRY_64, "5-component spec");
  std::cout << "ok\n";
  return 0;
}
