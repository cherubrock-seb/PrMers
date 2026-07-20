#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
EXPONENT="${2:-175000039}"
ITERATIONS="${3:-2}"
cd "$ROOT"

make -j"$(nproc)" engine-lib
mkdir -p build-tests
${CXX:-c++} -O2 -std=c++20 tests/type4_pfa9_engine_compare.cpp -ldl \
  -o build-tests/aevum-type4-pfa9-engine-compare

build-tests/aevum-type4-pfa9-engine-compare \
  build-engine/libaevum_engine.so "$DEVICE" "$EXPONENT" "$ITERATIONS"

echo "Force-adaptive resolution:"
cat > build-tests/resolve-type4.cpp <<'CPP'
#include "FFTConfig.h"
#include "Args.h"
#include <iostream>
#include <sstream>
#include <vector>
void log(const char*, ...) {}
std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> out; std::stringstream ss(text); std::string part;
  while (std::getline(ss, part, delimiter)) out.push_back(part); return out;
}
int main(int argc, char** argv) {
  Args args(true); const auto e = static_cast<u64>(std::stoull(argv[1]));
  auto fast = FFTConfig::bestFit(args, e, "pfa9:4:512:9:512:202");
  auto full = FFTConfig::bestFit(args, e, "pfa9full:4:512:9:512:202");
  std::cout << "forced-type4=" << fast.spec() << " arithmetic-type=" << int(fast.shape.fft_type) << "\n";
  std::cout << "full-type4=" << full.spec() << " arithmetic-type=" << int(full.shape.fft_type) << "\n";
}
CPP
${CXX:-c++} -O2 -std=c++20 -ffunction-sections -fdata-sections -Isrc \
  build-tests/resolve-type4.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp \
  src/common.cpp src/fs.cpp src/File.cpp src/log.cpp src/timeutil.cpp \
  -Wl,--gc-sections -o build-tests/resolve-type4
build-tests/resolve-type4 "$EXPONENT"
