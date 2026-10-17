// carryFused is unstable below about log2(N)/2 + 0.5 bits per word (gpuowl cdf2ab1); the host must switch to the long
// carry there.  The old fixed cutoff of 10 bpw let a 4M-word FFT3161 plan at 10.7 bpw run carryFused, fail its Gerbicz
// check within 10000 iterations and give a wrong LL residue.
#include "FFTConfig.h"
#include "Args.h"
#include <cmath>
#include <cstdio>
#include <fstream>
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

static int failures = 0;

static void require(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

static void expectLong(const std::string& spec, u64 E, bool wantLong) {
  FFTConfig fft{spec};
  char buf[256];
  snprintf(buf, sizeof(buf), "%s at %llu (%.2f bpw, cutoff %.2f) must use %s carry", spec.c_str(), (unsigned long long) E,
           E / double(fft.size()), fft.shape.longCarryBPW(), wantLong ? "long" : "fused");
  require(fft.shape.needsLongCarry(E) == wantLong, buf);
}

int main() {
  // The cutoff is max(10, log2(N)/2 + 0.5).
  require(FFTConfig{"1:512:8:512:101"}.shape.longCarryBPW() == 11.5, "2^22 words: cutoff 11.5");
  require(FFTConfig{"1:1K:8:1K:101"}.shape.longCarryBPW() == 12.5, "2^24 words: cutoff 12.5");
  require(FFTConfig{"1:256:2:256:101"}.shape.longCarryBPW() == 10.0, "2^18 words: cutoff stays 10");
  require(FFTConfig{"1:256:4:256:101"}.shape.longCarryBPW() == 10.0, "2^19 words: cutoff stays 10");

  // 4M words (2^22): measured on an RX 7900 XTX, 10.73 bpw failed Gerbicz and LL; 11.0 .. 12.5 passed.
  expectLong("1:512:8:512:101", 44999971u, true);    // 10.73 bpw, the failing plan
  expectLong("1:512:8:512:101", 46137319u, true);    // 11.00 bpw, inside the unsafe margin
  expectLong("1:512:8:512:101", 48234495u, true);    // 11.4999 bpw
  expectLong("1:512:8:512:101", 48234497u, false);   // just above 11.5 bpw
  expectLong("1:512:8:512:101", 48649967u, false);   // 11.60 bpw
  expectLong("1:512:8:512:101", 52428767u, false);   // 12.50 bpw
  expectLong("1:512:8:512:101", 30000001u, true);    // 7.15 bpw, below the old cutoff too
  expectLong("1:256:8:512:101", 44999971u, false);   // 2M words, 21.5 bpw: normal plans keep carryFused

  // 16M words: 10 .. 12.5 bpw is unsafe.
  expectLong("1:1K:8:1K:101", 201326592u, true);     // 12.0 bpw
  expectLong("1:1K:8:1K:101", 218103808u, false);    // 13.0 bpw

  // Small transforms keep the old cutoff of 10 bpw.
  expectLong("1:256:2:256:101", 2595000u, true);     // 9.90 bpw
  expectLong("1:256:2:256:101", 2648000u, false);    // 10.10 bpw

  // Other FFT types and PFA plans: N includes the PFA radix.
  expectLong("4:512:8:512:202", 44999971u, true);
  expectLong("0:512:8:512:101", 44999971u, true);
  {
    FFTConfig pfa{"pfa3:1:512:3:512:101"};
    require(std::fabs(pfa.shape.longCarryBPW() - (0.5 * std::log2(double(pfa.size())) + 0.5)) < 1e-9, "PFA cutoff uses the full PFA size");
    expectLong("pfa3:1:512:3:512:101", u64(10.5 * pfa.size()), true);
    expectLong("pfa3:1:512:3:512:101", u64(11.0 * pfa.size()), false);
  }

  // Gpu.cpp must take its decision from this predicate, not from a separate constant.
  {
    std::ifstream in("src/Gpu.cpp");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();
    require(!src.empty(), "src/Gpu.cpp readable (run from the repository root)");
    require(src.find("useLongCarry = useLongCarry || fft.shape.needsLongCarry(E);") != std::string::npos,
            "Gpu.cpp selects the long carry through FFTShape::needsLongCarry");
    require(src.find("bitsPerWord < 10.0") == std::string::npos, "Gpu.cpp no longer uses the fixed 10 bpw cutoff");
  }

  if (failures) return 1;
  std::cout << "ok\n";
  return 0;
}
