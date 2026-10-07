#include "FFTConfig.h"
#include "Args.h"
#include <cstdarg>
#include <cstdlib>
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

static bool refused(const Args& args, u64 E, const std::string& spec) {
  try {
    (void) FFTConfig::bestFit(args, E, spec);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

int main() {
  Args args(true);
  unsetenv("AEVUM_ALLOW_OVERCAPACITY_PLAN");

  // An explicit plan within its capacity is honored as before.
  const std::string small = "1:256:2:256:101";    // 256K words
  const u64 cap = FFTConfig{small}.maxExp();
  require(!refused(args, cap - 1000, small), "plan below its capacity must be accepted");
  require(FFTConfig::bestFit(args, cap - 1000, small).spec() == small, "explicit plan must be returned unchanged");

  // Beyond its capacity it used to run with only a warning; the result would be silently wrong.
  require(refused(args, 100000007u, small), "over-capacity explicit plan must be refused");
  require(refused(args, cap + cap / 10, small), "slightly over-capacity explicit plan must be refused");

  // Deliberate overrides: -od (the overdrive factor) and the explicit environment switch.
  Args overdrive(true);
  overdrive.fftOverdrive = 1.5;
  require(!refused(overdrive, cap + cap / 10, small), "-od must still raise the allowed exponent");
  require(refused(overdrive, cap * 2, small), "-od only raises the limit by its factor");

  setenv("AEVUM_ALLOW_OVERCAPACITY_PLAN", "0", 1);
  require(refused(args, 100000007u, small), "AEVUM_ALLOW_OVERCAPACITY_PLAN=0 does not override");
  setenv("AEVUM_ALLOW_OVERCAPACITY_PLAN", "1", 1);
  require(!refused(args, 100000007u, small), "AEVUM_ALLOW_OVERCAPACITY_PLAN=1 allows an over-capacity plan");
  unsetenv("AEVUM_ALLOW_OVERCAPACITY_PLAN");

#if !defined(__APPLE__)
  // The PRP boundary bridge plan must keep resolving across the range it is selected for.
  require(!refused(args, 197000003u, "4:512:8:512:202"), "type-4 bridge plan at 197M must stay within capacity");
#endif
  std::cout << "ok\n";
  return 0;
}
