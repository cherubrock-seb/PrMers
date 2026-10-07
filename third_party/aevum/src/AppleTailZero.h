#pragma once

#include "common.h"

// Element offset, in the tail output plane, of the line with index H/2 (H = WIDTH * MIDDLE) written by the Apple staged tailSquareZero path.
// The tail kernels store line k at memory line transPos(k, MIDDLE, WIDTH) = k / WIDTH + k % WIDTH * MIDDLE (fftheight.cl, INPLACE=0, which Apple
// forces), and a memory line holds SMALL_HEIGHT elements.  For an even MIDDLE this is (MIDDLE / 2) * SMALL_HEIGHT, but MIDDLE == 1 puts line
// H/2 at memory line WIDTH / 2 -- not at line 0, where (MIDDLE / 2) * SMALL_HEIGHT would overwrite the result for line 0.
inline u32 appleTailZeroHalfLineOffset(u32 width, u32 middle, u32 smallHeight) {
  const u32 halfLine = width * middle / 2;
  return (halfLine / width + halfLine % width * middle) * smallHeight;
}
