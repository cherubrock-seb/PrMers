// Host test: AEVUM_RADIX1K=8 selects the 1K radix-8 path everywhere except where the platform's staged FFT pipelines only support radix 4.
//
// The Apple staged pipelines iterate generic radix passes (stage *= nH up to the group size, then a final radix); for a 1K side with radix 8
// that is four radix-8 passes (4096 points) instead of 8 * 8 * 16.  aevumRadix8For1K() therefore ignores the override on Apple.  The same rule
// is selected on other platforms by building with -DAEVUM_FORCE_RADIX4_1K=1; this test is built both ways.

#include "FFTConfig.h"

#include <cstdio>
#include <cstdlib>

int main() {
#if defined(__APPLE__) || defined(AEVUM_FORCE_RADIX4_1K)
  const bool expectRadix8WhenAsked = false;
#else
  const bool expectRadix8WhenAsked = true;
#endif
  int bad = 0;
  unsetenv("AEVUM_RADIX1K");
  if (aevumRadix8For1K()) { std::printf("FAIL: radix 8 without AEVUM_RADIX1K\n"); ++bad; }
  setenv("AEVUM_RADIX1K", "4", 1);
  if (aevumRadix8For1K()) { std::printf("FAIL: radix 8 with AEVUM_RADIX1K=4\n"); ++bad; }
  setenv("AEVUM_RADIX1K", "8", 1);
  if (aevumRadix8For1K() != expectRadix8WhenAsked) {
    std::printf("FAIL: AEVUM_RADIX1K=8 gave radix %d, expected %d\n", aevumRadix8For1K() ? 8 : 4, expectRadix8WhenAsked ? 8 : 4);
    ++bad;
  }
  std::printf("AEVUM_RADIX1K=8 -> radix %d for a 1K side (%s)\n", aevumRadix8For1K() ? 8 : 4,
              expectRadix8WhenAsked ? "override honoured" : "override ignored: staged pipelines are radix-4 only");
  std::printf("%s\n", bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
