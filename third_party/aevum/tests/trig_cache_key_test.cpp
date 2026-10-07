#include "TrigBufCache.h"
#include <iostream>
#include <set>
#include <stdexcept>

// Every distinct (number type in use, TAIL_TRIGS) configuration must map to its own trig cache key; otherwise two
// Gpus sharing a TrigBufCache can be handed each other's twiddle tables.
int main() {
  std::set<u32> keys;
  u32 configs = 0;
  for (u32 flags = 0; flags < 16; ++flags) {
    for (u32 tt = 0; tt < 4 * 4 * 4 * 4; ++tt) {
      for (u32 tk = 0; tk < 2; ++tk) {
        ++configs;
        keys.insert(trigKeyPart(flags & 1, tt & 3, flags & 2, (tt >> 2) & 3, flags & 4, (tt >> 4) & 3, flags & 8, (tt >> 6) & 3, tk));
      }
    }
  }
  std::cout << configs << " configurations, " << keys.size() << " distinct keys\n";
  if (keys.size() != configs) throw std::runtime_error("trig cache key aliases distinct configurations");

  // The case the old additive key got wrong: FP64 in use with TAIL_TRIGS=1 vs FP64 unused with TAIL_TRIGS=2.
  if (trigKeyPart(true, 1, false, 0, false, 0, false, 0, false) == trigKeyPart(false, 2, false, 0, false, 0, false, 0, false))
    throw std::runtime_error("in-use flag and TAIL_TRIGS are not separate");
  std::cout << "ok\n";
  return 0;
}
