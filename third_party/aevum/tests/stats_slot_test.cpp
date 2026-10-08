// Host test: the carry-stats slot allocator never hands out an index outside the buffer.
#include "StatsSlot.h"

#include <cstdio>

int main() {
  const u32 size = 4;
  u32 pos = 0;
  int bad = 0;
  // A disabled buffer keeps its position and never advances.
  for (int i = 0; i < 3; ++i) if (nextStatsSlot(pos, size, false) != 0 || pos != 0) ++bad;
  // Filling: slots 0..size-1 in order, then every later result goes to the last slot.
  for (u32 i = 0; i < size; ++i) if (nextStatsSlot(pos, size, true) != i) ++bad;
  for (int i = 0; i < 5; ++i) if (nextStatsSlot(pos, size, true) != size - 1 || pos != size) ++bad;
  std::printf("%s\n", bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
