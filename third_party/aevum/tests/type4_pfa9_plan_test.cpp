#include "FFTConfig.h"
#include "Args.h"
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Minimal host-only stubs needed by FFTConfig.cpp.  Unused policy functions are
// removed by --gc-sections in the test target.
void log(const char*, ...) {}
std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, delimiter)) result.push_back(part);
  return result;
}

int main() {
  Args args(true);
  FFTConfig plan("pfa9:4:512:9:512:202");
  if (plan.shape.fft_type != FFT323161 || plan.pfa_radix != 9 ||
      plan.shape.width != 512 || plan.shape.middle != 9 ||
      plan.shape.height != 512 || plan.variant != 202 ||
      plan.size() != 4718592 || plan.spec() != "pfa9:4:512:9:512:202") {
    throw std::runtime_error("FFT323161 PFA9 plan resolution mismatch");
  }
#if defined(__APPLE__)
  bool appleType4Rejected = false;
  try { (void) FFTConfig::bestFit(args, 175000039, "4:512:8:512:202"); }
  catch (const std::runtime_error&) { appleType4Rejected = true; }
  if (!appleType4Rejected)
    throw std::runtime_error("Apple explicit Type4 plan must be rejected before kernel creation");
#else
  FFTConfig pow2 = FFTConfig::bestFit(args, 175000039, "4:512:8:512:202");
  if (pow2.shape.fft_type != FFT323161 || pow2.isPfa() ||
      pow2.shape.width != 512 || pow2.shape.middle != 8 ||
      pow2.shape.height != 512 || pow2.variant != 202 ||
      pow2.size() != 4194304 || pow2.spec() != "4:512:8:512:202") {
    throw std::runtime_error("power-of-two FFT323161 plan resolution mismatch");
  }
#endif

  bool rejected = false;
  try { FFTConfig invalid("pfa3:4:512:3:512:202"); (void) invalid; }
  catch (...) { rejected = true; }
  if (!rejected) throw std::runtime_error("FFT323161 radix-3 must be rejected");
#if defined(__APPLE__)
  bool applePfaRejected = false;
  try { (void) FFTConfig::bestFit(args, 175000039, "pfa:9"); }
  catch (const std::runtime_error&) { applePfaRejected = true; }
  if (!applePfaRejected)
    throw std::runtime_error("Apple native PFA plan must be rejected before kernel creation");
#else
  FFTConfig adaptive = FFTConfig::bestFit(args, 175000039, "pfa9:4:512:9:512:202");
  if (adaptive.shape.fft_type != FFT3161 || adaptive.pfa_radix != 9 ||
      adaptive.variant != 202 || adaptive.size() != 4718592 || !adaptive.adaptive_type4_elided)
    throw std::runtime_error("adaptive type-4 PFA9 did not select exact paired-NTT fast path");

  FFTConfig full = FFTConfig::bestFit(args, 175000039, "pfa9full:4:512:9:512:202");
  if (full.shape.fft_type != FFT323161 || full.adaptive_type4_elided)
    throw std::runtime_error("full type-4 PFA9 plan was unexpectedly elided");

  FFTConfig auto9 = FFTConfig::bestFit(args, 175000039, "pfa:9");
  if (auto9.pfa_radix != 9 || auto9.variant != 202)
    throw std::runtime_error("forced radix-9 did not select optimized variant 202");
#endif

  FFTConfig throughput = FFTConfig::bestFit(args, 175000039, "throughput:auto");
#if defined(__APPLE__)
  // Apple excludes both Type4 and native-PFA runtime paths.  throughput:auto
  // must therefore retain the staged stock FFT3161 plan.
  FFTConfig appleStock = FFTConfig::bestFit(args, 175000039, "");
  if (throughput.isPfa() ||
      throughput.spec() != appleStock.spec() ||
      throughput.size() != appleStock.size()) {
    std::cerr << "macOS throughput selected " << throughput.spec()
              << " size=" << throughput.size()
              << ", expected stock " << appleStock.spec()
              << " size=" << appleStock.size() << std::endl;
    throw std::runtime_error(
        "macOS throughput selector did not retain the stock FFT3161 plan");
  }
#else
  if (throughput.spec() != "4:512:8:512:202" ||
      throughput.size() != 4194304) {
    std::cerr << "throughput selected " << throughput.spec()
              << " size=" << throughput.size() << std::endl;
    throw std::runtime_error(
        "throughput selector did not choose the measured 4M FFT323161 lead-cache plan");
  }
#endif

  FFTConfig pow2auto = FFTConfig::bestFit(args, 175000039, "pow2:auto");
#if defined(__APPLE__)
  // pow2:auto falls back to the normal stock plan when Apple excludes the
  // experimental FFT323161 lead-cache family.
  if (pow2auto.isPfa() ||
      pow2auto.spec() != appleStock.spec() ||
      pow2auto.size() != appleStock.size()) {
    std::cerr << "macOS pow2:auto selected " << pow2auto.spec()
              << " size=" << pow2auto.size()
              << ", expected stock " << appleStock.spec()
              << " size=" << appleStock.size() << std::endl;
    throw std::runtime_error(
        "macOS power-of-two selector did not retain the stock plan");
  }
#else
  if (pow2auto.spec() != "4:512:8:512:202")
    throw std::runtime_error(
        "power-of-two selector did not choose the measured FFT323161 plan");
#endif

  std::cout << "Aevum platform-aware throughput-auto, Apple stock safety and PFA9 plan test passed" << std::endl;
  return 0;
}
