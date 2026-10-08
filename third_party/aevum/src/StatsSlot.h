#pragma once

#include "common.h"

// Position in a stats buffer of `size` entries for the next kernel result.  A buffer that is not collecting (`enabled` false) keeps
// pointing at the current position.  Once the buffer is full, further results are folded into its last slot: handing out `size` itself
// would make the kernel's atomic_max write one element past the end.
inline u32 nextStatsSlot(u32& pos, u32 size, bool enabled) {
  if (!enabled) { return pos; }
  return pos < size ? pos++ : size - 1;
}
