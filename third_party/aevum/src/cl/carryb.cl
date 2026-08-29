// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "carryutil.cl"
#include "weight.cl"

KERNEL(G_W) carryB(P(Word2) io, CP(CarryABM) carryIn) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;

  // Derive the big vs. little flags from the fractional number of bits in each FFT word rather read the flags from memory.
  // Calculate the most significant 32-bits of FRAC_BPW * the index of the FFT word.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 line = gy * CARRY_LEN;
  u32 word_index = (gx * G_W * H + me * H + line) * 2;
  u32 frac_bits = fracBits(word_index) + FRAC_BPW_HI;

#if PFA_RESIDENT && PFA_RADIX
  P(Word) ioResident = (P(Word)) io;
#else
  io += G_W * gx + WIDTH * CARRY_LEN * gy;
#endif

  u32 HB = BIG_HEIGHT / CARRY_LEN;

  u32 prev = (gy + HB * G_W * gx + HB * me + (HB * WIDTH - 1)) % (HB * WIDTH);
  u32 prevLine = prev % HB;
  u32 prevCol  = prev / HB;

  CarryABM carry = carryIn[WIDTH * prevLine + prevCol];

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = i * WIDTH + me;
    bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;   // Same as frac_bits + (2*i) * FRAC_BPW_HI + FRAC_BPW_HI <= FRAC_BPW_HI;
#if PFA_RESIDENT && PFA_RADIX
    const u32 logical0 = word_index + 2u * (u32)i;
    const u32 r0 = pfaResidentScalarIndex(logical0);
    const u32 r1 = pfaResidentScalarIndex(logical0 + 1u);
    Word2 value = U2(ioResident[r0], ioResident[r1]);
    value = carryWord(value, &carry, biglit0, biglit1);
    ioResident[r0] = value.x;
    ioResident[r1] = value.y;
#else
    io[p] = carryWord(io[p], &carry, biglit0, biglit1);
#endif
    if (!carry) { return; }
  }
}


#if PFA_RESIDENT && PFA_RADIX && FFT_TYPE == FFT323161
// EXP5: direct-address resident carryB for pfa9full:4:512:9:512:202.
// Semantics are identical to stock carryB.  The only change is replacing the
// generic resident inverse map (% and / per scalar) with the closed-form map
// for WIDTH=512, SMALL_HEIGHT=512, radix=9, CARRY_LEN=8.
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 9 || CARRY_LEN != 8 || NW != 8
#error carryBResidentFast is specialized for WIDTH=512, SMALL_HEIGHT=512, MIDDLE=9, CARRY_LEN=8, NW=8
#endif
KERNEL(G_W) carryBResidentFast(P(Word2) io, CP(CarryABM) carryIn) {
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  const u32 gx = g % NW;
  const u32 gy = g / NW;
  const u32 A = gx * G_W + me;          // canonical width coordinate 0..511
  const u32 H = BIG_HEIGHT;             // 4608
  const u32 line = gy * CARRY_LEN;       // 0,8,...,4600
  P(Word) ioResident = (P(Word)) io;

  const u32 word_index = (A * H + line) * 2u;
  const u32 frac_bits = fracBits(word_index) + FRAC_BPW_HI;

  // Exact simplification of stock carryB's predecessor permutation:
  // within one width column, segment gy consumes carry from gy-1; segment 0
  // wraps to the last segment of the previous width column.
  const u32 HB = H / CARRY_LEN;          // 576
  const u32 prevGy = gy ? gy - 1u : HB - 1u;
  const u32 prevA = gy ? A : ((A - 1u) & (WIDTH - 1u));
  CarryABM carry = carryIn[WIDTH * prevGy + prevA];

  // Closed-form inverse Good-Thomas resident coordinates for canonical pair
  // pair=A*4608+(line+i).  line cannot cross a 512 boundary inside CARRY_LEN.
  const u32 section = gy >> 6;           // line / 512
  const u32 y0 = (gy & 63u) << 3;        // line % 512
  const u32 q = (9u * A + section) & 511u;
  u32 rowX = (7u * (gy % 9u)) % 9u;      // (2*line) mod 9 = 16*gy mod 9
  u32 rowY = rowX + 1u;
  if (rowY >= 9u) rowY -= 9u;

#pragma unroll
  for (i32 i = 0; i < CARRY_LEN; ++i) {
    const u32 y = y0 + (u32)i;
    const u32 pairX = (rowX * SMALL_HEIGHT + y) * WIDTH + q;
    const u32 pairY = (rowY * SMALL_HEIGHT + y) * WIDTH + q;
    const u32 r0 = 2u * pairX;
    const u32 r1 = 2u * pairY + 1u;

    const bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    const bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;

    Word2 value = U2(ioResident[r0], ioResident[r1]);
    value = carryWord(value, &carry, biglit0, biglit1);
    ioResident[r0] = value.x;
    ioResident[r1] = value.y;
    if (!carry) return;

    rowX += 2u; if (rowX >= 9u) rowX -= 9u;
    rowY += 2u; if (rowY >= 9u) rowY -= 9u;
  }
}
#endif


#if PFA_RESIDENT && PFA_RADIX && FFT_TYPE == FFT323161
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 9 || CARRY_LEN != 8 || NW != 8
#error carryBResidentBlock4 is specialized for WIDTH=512, SMALL_HEIGHT=512, MIDDLE=9, CARRY_LEN=8, NW=8
#endif
#define PFA_CARRY_BLOCK 4u
KERNEL(G_W) carryBResidentBlock4(P(Word2) io, CP(CarryABM) carryIn) {
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  const u32 gx = g % NW;
  const u32 blockGy = g / NW;            // 0..143
  const u32 A = gx * G_W + me;
  const u32 H = BIG_HEIGHT;
  const u32 blockPairs = PFA_CARRY_BLOCK * CARRY_LEN; // 32
  const u32 line = blockGy * blockPairs;

  // Previous block's carry; block 0 wraps to the final block of the previous
  // canonical width column, exactly like stock carryB's cyclic predecessor.
  const u32 blocksPerColumn = H / blockPairs; // 144
  const u32 prevBlock = blockGy ? blockGy - 1u : blocksPerColumn - 1u;
  const u32 prevA = blockGy ? A : ((A - 1u) & (WIDTH - 1u));
  CarryABM carry = carryIn[WIDTH * prevBlock + prevA];

  // No incoming carry means this block is already final.  This is equivalent
  // to stock carryB's first carryWord followed by its immediate return.
  if (!carry) return;

  P(Word) ioResident = (P(Word)) io;
  const u32 word_index = (A * H + line) * 2u;
  const u32 frac_bits = fracBits(word_index) + FRAC_BPW_HI;

  const u32 section = blockGy >> 4;
  const u32 y0 = (blockGy & 15u) << 5;
  const u32 q = (9u * A + section) & 511u;
  u32 rowX = blockGy % 9u;
  u32 rowY = rowX + 1u;
  if (rowY >= 9u) rowY -= 9u;

  // By the same CARRY_LEN invariant used by stock carryB, a boundary carry
  // must die inside the first eight pairs.  Internal block boundaries were
  // already consumed directly by carryResidentBlock4.
#pragma unroll
  for (i32 i = 0; i < CARRY_LEN; ++i) {
    const u32 y = y0 + (u32)i;
    const u32 pairX = (rowX * SMALL_HEIGHT + y) * WIDTH + q;
    const u32 pairY = (rowY * SMALL_HEIGHT + y) * WIDTH + q;
    const u32 r0 = 2u * pairX;
    const u32 r1 = 2u * pairY + 1u;

    const bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    const bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;

    Word2 value = U2(ioResident[r0], ioResident[r1]);
    value = carryWord(value, &carry, biglit0, biglit1);
    ioResident[r0] = value.x;
    ioResident[r1] = value.y;
    if (!carry) return;

    rowX += 2u; if (rowX >= 9u) rowX -= 9u;
    rowY += 2u; if (rowY >= 9u) rowY -= 9u;
  }
}
#undef PFA_CARRY_BLOCK
#endif


#if PFA_RESIDENT && PFA_RADIX && FFT_TYPE == FFT323161
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 9 || CARRY_LEN != 8 || NW != 8
#error carryBResidentQBlock4 is specialized for WIDTH=512, SMALL_HEIGHT=512, MIDDLE=9, CARRY_LEN=8, NW=8
#endif
#define PFA_CARRY_BLOCK 4u
#define PFA_Q_INV9 57u
KERNEL(G_W) carryBResidentQBlock4(P(Word2) io, CP(CarryABM) carryIn) {
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  const u32 qx = g % NW;
  const u32 blockGy = g / NW;
  const u32 q = qx * G_W + me;
  const u32 section = blockGy >> 4;
  const u32 A = (PFA_Q_INV9 * ((q - section) & (WIDTH - 1u))) & (WIDTH - 1u);
  const u32 H = BIG_HEIGHT;
  const u32 blockPairs = PFA_CARRY_BLOCK * CARRY_LEN;
  const u32 line = blockGy * blockPairs;

  const u32 blocksPerColumn = H / blockPairs;
  const u32 prevBlock = blockGy ? blockGy - 1u : blocksPerColumn - 1u;
  const u32 prevA = blockGy ? A : ((A - 1u) & (WIDTH - 1u));
  CarryABM carry = carryIn[WIDTH * prevBlock + prevA];
  if (!carry) return;

  P(Word) ioResident = (P(Word)) io;
  const u32 word_index = (A * H + line) * 2u;
  const u32 frac_bits = fracBits(word_index) + FRAC_BPW_HI;
  const u32 y0 = (blockGy & 15u) << 5;
  u32 rowX = blockGy % 9u;
  u32 rowY = rowX + 1u; if (rowY >= 9u) rowY -= 9u;

#pragma unroll
  for (i32 i = 0; i < CARRY_LEN; ++i) {
    const u32 y = y0 + (u32)i;
    const u32 pairX = (rowX * SMALL_HEIGHT + y) * WIDTH + q;
    const u32 pairY = (rowY * SMALL_HEIGHT + y) * WIDTH + q;
    const u32 r0 = 2u * pairX;
    const u32 r1 = 2u * pairY + 1u;
    const bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    const bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;
    Word2 value = U2(ioResident[r0], ioResident[r1]);
    value = carryWord(value, &carry, biglit0, biglit1);
    ioResident[r0] = value.x;
    ioResident[r1] = value.y;
    if (!carry) return;
    rowX += 2u; if (rowX >= 9u) rowX -= 9u;
    rowY += 2u; if (rowY >= 9u) rowY -= 9u;
  }
}
#undef PFA_Q_INV9
#undef PFA_CARRY_BLOCK
#endif
