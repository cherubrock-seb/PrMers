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

  // A type prefix on a size: the size is the part after the prefix, and the shape is of the requested type.
  // The whole spec used to be split on '-', so "1:8M" was read as the size 1 (times the trailing M) and the prefix was
  // dropped, so "4:2M" became an FFT3161 of 4M.
  struct Prefixed { const char* spec; FFT_TYPES type; u32 size; };
  for (const Prefixed& p : {Prefixed{"1:8M", FFT3161, 8u << 20}, Prefixed{"1:2M", FFT3161, 2u << 20},
                            Prefixed{"4:2M", FFT323161, 2u << 20}, Prefixed{"4:8M", FFT323161, 8u << 20},
                            Prefixed{"2:1M", FFT3261, 1u << 20}, Prefixed{"0:6.5M", FFT64, 6815744u}}) {
    try {
      FFTConfig fft{p.spec};
      std::cout << p.spec << " -> " << fft.spec() << "\n";
      require(fft.shape.fft_type == p.type, "prefixed size-only spec resolved to the wrong FFT type");
      require(fft.size() == p.size, "prefixed size-only spec resolved to the wrong size");
      for (const FFTShape& shape : FFTShape::multiSpec(p.spec)) {
        require(shape.fft_type == p.type && shape.size() == p.size, "multiSpec of a prefixed size returned another type or size");
      }
    } catch (const std::runtime_error&) {
      throw;
    } catch (...) {
      std::cerr << "prefixed size-only spec '" << p.spec << "' threw\n";
      return 1;
    }
  }
  // A prefixed range keeps the prefix's type and both bounds.
  {
    auto shapes = FFTShape::multiSpec("4:1M-4M");
    require(!shapes.empty(), "prefixed size range found nothing");
    for (const FFTShape& shape : shapes) {
      require(shape.fft_type == FFT323161 && shape.size() >= (1u << 20) && shape.size() <= (4u << 20),
              "prefixed size range returned another type or size");
    }
  }
  // An unprefixed size or range is unchanged: FFT3161 only.
  for (const FFTShape& shape : FFTShape::multiSpec("1M-4M")) {
    require(shape.fft_type == FFT3161, "unprefixed size range must stay FFT3161");
  }
  // A bare prefix is not a spec.
  for (const char* bad : {"1:", "4:"}) {
    bool refused = false;
    try { (void) FFTShape::multiSpec(bad); } catch (const char*) { refused = true; }
    require(refused, "a bare type prefix must be refused");
  }

  // The 3/4/5-component forms must keep working.
  require(FFTConfig{"256:2:256"}.size() == 256 * 2 * 256 * 2, "3-component spec");
  require(FFTConfig{"256:2:256:212"}.variant == 212, "4-component spec");
  require(FFTConfig{"256:2:256:212:1"}.carry == CARRY_64, "5-component spec");
  std::cout << "ok\n";
  return 0;
}
