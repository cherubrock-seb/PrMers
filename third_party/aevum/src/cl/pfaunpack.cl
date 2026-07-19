// Native Aevum Good-Thomas/PFA output repacking.
// Gathers the inverse odd rows and scatters them into the exact stock
// transposed transform order consumed by carry.cl.

#include "base.cl"

#if FFT_TYPE == FFT3161 && PFA_RADIX

inline u32 pfaSourcePairIndex(u32 logical) {
  const u32 row = logical % PFA_RADIX;
  const u32 b = logical % PFA_BINARY_LENGTH;
  const u32 binary_pair = b >> 1;
  const u32 x = binary_pair / SMALL_HEIGHT;
  const u32 y = binary_pair - x * SMALL_HEIGHT;
  const u32 line = row * SMALL_HEIGHT + y;
  return line * WIDTH + x;
}

inline u32 pfaCanonicalTransformIndex(u32 pair) {
  const u32 x = pair / BIG_HEIGHT;
  const u32 g = pair - x * BIG_HEIGHT;
  return g * WIDTH + x;
}

KERNEL(256) pfaUnpack(P(T2) out, CP(T2) in) {
  const u32 pair = get_global_id(0);
  const u32 n0 = pair << 1;
  const u32 n1 = n0 + 1u;
  const u32 src0 = pfaSourcePairIndex(n0);
  const u32 src1 = pfaSourcePairIndex(n1);
  const u32 dst = pfaCanonicalTransformIndex(pair);

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);

  const GF31 a31 = in31[src0];
  const GF31 b31 = in31[src1];
  const GF61 a61 = in61[src0];
  const GF61 b61 = in61[src1];

  // tailSquare/tailMul deliberately leave inverse results component-swapped.
  // Stock carry.cl immediately applies SWAP_XY, so its canonical pre-carry
  // layout is (odd, even), not (even, odd).  A Good-Thomas gather may take
  // the adjacent logical digits from different rows; recover each logical
  // scalar from the swapped source pair, then rebuild the same stock layout.
  const Z31 even31 = (n0 & 1u) ? a31.x : a31.y;
  const Z31 odd31  = (n1 & 1u) ? b31.x : b31.y;
  const Z61 even61 = (n0 & 1u) ? a61.x : a61.y;
  const Z61 odd61  = (n1 & 1u) ? b61.x : b61.y;
  out31[dst] = U2(odd31, even31);
  out61[dst] = U2(odd61, even61);
}

#endif
