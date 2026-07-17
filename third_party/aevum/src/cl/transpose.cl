// Copyright (C) Mihai Preda

#include "base.cl"

#if WordSize <= 4

void transposeWords(u32 W, u32 H, local Word2 *lds, global const Word2 *restrict in, global Word2 *restrict out) {
  u32 GPW = W / 64, GPH = H / 64;

  u32 g = get_group_id(0);
  u32 gy = g % GPH;
  u32 gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += 64 * W * gy + 64 * gx;
  out  += 64 * gy + 64 * H * gx;
  u32 me = get_local_id(0);
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    lds[i * 64 + me] = in[i * W + me];
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[i * H + me] = lds[me * 64 + i];
  }
}

// from transposed to sequential.
KERNEL(64) transposeOut(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(64) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(BIG_HEIGHT, WIDTH, lds, in, out);
}

#else

void transposeWords(u32 W, u32 H, local Word *lds, global const Word2 *restrict in, global Word2 *restrict out) {
  u32 GPW = W / 64, GPH = H / 64;

  u32 g = get_group_id(0);
  u32 gy = g % GPH;
  u32 gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += 64 * W * gy + 64 * gx;
  out  += 64 * gy + 64 * H * gx;
  u32 me = get_local_id(0);
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    lds[i * 64 + me] = in[i * W + me].x;
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[i * H + me].x = lds[me * 64 + i];
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    lds[i * 64 + me] = in[i * W + me].y;
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[i * H + me].y = lds[me * 64 + i];
  }
}

// from transposed to sequential.
KERNEL(64) transposeOut(P(Word2) out, CP(Word2) in) {
  local Word lds[4096];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(64) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word lds[4096];
  transposeWords(BIG_HEIGHT, WIDTH, lds, in, out);
}

#endif


// Apple OpenCL-to-Metal compatibility path.
//
// The stock transpose uses a full 64x64 local-memory tile (32 KiB) plus
// three barriers.  On Apple OpenCL 1.2 that kernel can compile and execute
// without reporting an error yet return corrupted data, including for an
// all-zero register.  Use a direct global transpose instead.  Each work-item
// copies one Word2 and there is no cross-work-item dependency.
#if defined(AEVUM_APPLE_OPENCL12)

KERNEL(64) transposeOutAppleGlobal(P(Word2) out, CP(Word2) in) {
  const u32 gid = get_global_id(0);
  const u32 row = gid / WIDTH;
  const u32 col = gid - row * WIDTH;
  out[col * BIG_HEIGHT + row] = in[gid];
}

KERNEL(64) transposeInAppleGlobal(P(Word2) out, CP(Word2) in) {
  const u32 gid = get_global_id(0);
  const u32 row = gid / BIG_HEIGHT;
  const u32 col = gid - row * BIG_HEIGHT;
  out[col * WIDTH + row] = in[gid];
}

#endif
