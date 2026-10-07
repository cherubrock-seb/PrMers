#include "Args.h"
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal host-only stubs needed by Args.cpp (see type4_pfa9_plan_test.cpp).
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
  // The tail trig tables must be built for the TAIL_TRIGS* value the kernels are compiled with, which is
  // resolved (see clDefines) as: per-Gpu extra -use list, then -use, then the "! <fft> ..." config.txt line.
  const std::string fft = "1:256:2:256";
  const std::vector<KeyVal> none;
  const std::vector<KeyVal> extra{{"TAIL_TRIGS61", "1"}};

  Args args(true);
  require(args.valueFor("TAIL_TRIGS61", 0, fft, none) == 0, "default is returned when nothing sets the key");
  require(args.valueFor("TAIL_TRIGS61", 0, fft, extra) == 1, "extraConf (a tune.txt -use list) is seen");
  require(args.value("TAIL_TRIGS61", 0) == 0, "Args::value() alone does not see extraConf");

  args.perFftConfig[fft] = Args::splitUses("TAIL_TRIGS61=2,TAIL_TRIGS31=1");
  require(args.valueFor("TAIL_TRIGS61", 0, fft, none) == 2, "per-FFT config line is seen");
  require(args.valueFor("TAIL_TRIGS31", 0, fft, none) == 1, "per-FFT config line, second key");
  require(args.valueFor("TAIL_TRIGS61", 0, "1:512:8:512", none) == 0, "per-FFT config of another FFT is ignored");

  args.flags["TAIL_TRIGS61"] = "3";
  require(args.valueFor("TAIL_TRIGS61", 0, fft, none) == 3, "-use outranks the per-FFT config line");
  require(args.valueFor("TAIL_TRIGS61", 0, fft, extra) == 1, "extraConf outranks -use");
  std::cout << "ok\n";
  return 0;
}
