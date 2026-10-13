// Exact NVIDIA Tensor Core PFA7 over GF(2^31-1).
//
// Each balanced M31 value is decomposed in base 2^11 into three signed limbs.
// Limbs are in [-1024,1023], therefore every limb is represented exactly by
// FP16.  A seven-term dot product is bounded by 7*1024^2 = 7,340,032 < 2^23,
// so every integer product and every possible accumulation order is exact in
// FP32.  Tensor Cores are therefore used as an exact integer dot-product
// engine; no floating-point approximation enters the NTT result.
//
// A warp evaluates 32 independent PFA7 rows as two 16x8 by 8x8 MMA batches.
// Existing local memory is used as warp-private staging.  The inverse table
// already includes inv(7), eliminating the scalar post-normalization pass.

#pragma once

#if !NVIDIAGPU || HAS_PTX < 750 || PFA_RADIX != 7
#error pfa7tensor.cl requires NVIDIA sm_75+ PFA7
#endif

#define PFA7_TC_RADIX 7u
#define PFA7_TC_LIMBS 3u
#define PFA7_TC_BASE_BITS 11u
#define PFA7_TC_WARP_WORDS (32u * PFA7_TC_RADIX)

// Balanced base-2048 digits for root^e, e=0..6.
constant i32 PFA7_TC_FWD_DIGITS[21] = {
     1,    0,    0,
   302,  424,  213,
  -801, -302,  -94,
   187,  762, -131,
   826, -620, -130,
   198,  780, -225,
  -712, 1004, -146
};

// Balanced base-2048 digits for inv(7) * root^(-e), e=0..6.
// This makes the inverse Tensor path match pfaInverseMiddle exactly while
// avoiding seven extra scalar modular multiplies per GF31 component.
constant i32 PFA7_TC_INV_NORM_DIGITS[21] = {
   878, -293,  -73,
  -394, -149, -167,
    28,  404,  114,
  -175, -381,  201,
   319, -769,  128,
  -992,  835,  -87,
   336,  353, -116
};

inline void pfa7TcWarpSync() {
  __asm("bar.warp.sync 0xffffffff;");
}

inline i32 pfa7TcDigit(Z31 a, u32 limb) {
  i32 v = get_balanced_Z31(a);
  i32 d0 = ((v + 1024) & 2047) - 1024;
  if (limb == 0) return d0;
  v = (v - d0) >> PFA7_TC_BASE_BITS;
  i32 d1 = ((v + 1024) & 2047) - 1024;
  if (limb == 1) return d1;
  return (v - d1) >> PFA7_TC_BASE_BITS;
}

inline i32 pfa7TcCoeffDigit(u32 inverseNormalized, u32 out, u32 in, u32 limb) {
  if (out >= PFA7_TC_RADIX || in >= PFA7_TC_RADIX) return 0;
  const u32 exponent = (out * in) % PFA7_TC_RADIX;
  const u32 index = exponent * PFA7_TC_LIMBS + limb;
  return inverseNormalized ? PFA7_TC_INV_NORM_DIGITS[index]
                           : PFA7_TC_FWD_DIGITS[index];
}

// Pack two signed integers that are exactly representable in FP16 into the
// f16x2 carrier expected by mma.sync.  Keeping the conversion in PTX avoids a
// dependency on cl_khr_fp16 in OpenCL C.
inline u32 pfa7TcPackF16x2(i32 lo, i32 hi) {
  u32 packed;
  __asm("{\n\t"
        ".reg .f16 h0, h1;\n\t"
        "cvt.rn.f16.s32 h0, %1;\n\t"
        "cvt.rn.f16.s32 h1, %2;\n\t"
        "mov.b32 %0, {h0, h1};\n\t"
        "}"
        : "=r"(packed) : "r"(lo), "r"(hi));
  return packed;
}

inline float4 pfa7TcMma1688(u32 a0, u32 a1, u32 b0) {
  float d0, d1, d2, d3;
  const float z = 0.0f;
  __asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0),
          "f"(z), "f"(z), "f"(z), "f"(z));
  return (float4)(d0, d1, d2, d3);
}

inline Z31 pfa7TcAccumulate(Z31 acc, float exactInteger, u32 shift) {
  const i32 v = (i32) exactInteger;
  return add(acc, shl(make_Z31(v), shift));
}

inline void pfa7TcComponent(GF31 *u, u32 inverseNormalized,
                            local Z31 *scratch, u32 imag) {
  const u32 me = get_local_id(0);
  const u32 lane = me & 31u;
  const u32 warp = me >> 5;
  local Z31 *ws = scratch + warp * PFA7_TC_WARP_WORDS;

#pragma unroll
  for (u32 k = 0; k < PFA7_TC_RADIX; ++k)
    ws[lane * PFA7_TC_RADIX + k] = imag ? u[k].y : u[k].x;

  pfa7TcWarpSync();

  const u32 group = lane >> 2;
  const u32 tid = lane & 3u;
  const u32 col0 = tid * 2u;
  const u32 col1 = col0 + 1u;

#pragma unroll
  for (u32 batch = 0; batch < 2; ++batch) {
    const u32 row0 = batch * 16u + group;
    const u32 row1 = row0 + 8u;
    Z31 o0 = 0, o1 = 0, o2 = 0, o3 = 0;

#pragma unroll
    for (u32 p = 0; p < PFA7_TC_LIMBS; ++p) {
      const Z31 x00 = col0 < PFA7_TC_RADIX ? ws[row0 * PFA7_TC_RADIX + col0] : 0;
      const Z31 x01 = col1 < PFA7_TC_RADIX ? ws[row0 * PFA7_TC_RADIX + col1] : 0;
      const Z31 x10 = col0 < PFA7_TC_RADIX ? ws[row1 * PFA7_TC_RADIX + col0] : 0;
      const Z31 x11 = col1 < PFA7_TC_RADIX ? ws[row1 * PFA7_TC_RADIX + col1] : 0;

      const u32 a0 = pfa7TcPackF16x2(pfa7TcDigit(x00, p), pfa7TcDigit(x01, p));
      const u32 a1 = pfa7TcPackF16x2(pfa7TcDigit(x10, p), pfa7TcDigit(x11, p));

#pragma unroll
      for (u32 q = 0; q < PFA7_TC_LIMBS; ++q) {
        // For m16n8k8 B fragments, this lane owns rows 2*tid/{+1} and
        // column group.  Hardware redistributes them to the D fragment.
        const u32 b0 = pfa7TcPackF16x2(
            pfa7TcCoeffDigit(inverseNormalized, group, col0, q),
            pfa7TcCoeffDigit(inverseNormalized, group, col1, q));
        const float4 d = pfa7TcMma1688(a0, a1, b0);
        const u32 shift = (PFA7_TC_BASE_BITS * (p + q)) % 31u;
        o0 = pfa7TcAccumulate(o0, d.x, shift);
        o1 = pfa7TcAccumulate(o1, d.y, shift);
        o2 = pfa7TcAccumulate(o2, d.z, shift);
        o3 = pfa7TcAccumulate(o3, d.w, shift);
      }
    }

    // D fragment mapping: rows group/group+8, columns 2*tid/{+1}.
    if (col0 < PFA7_TC_RADIX) {
      ws[row0 * PFA7_TC_RADIX + col0] = o0;
      ws[row1 * PFA7_TC_RADIX + col0] = o2;
    }
    if (col1 < PFA7_TC_RADIX) {
      ws[row0 * PFA7_TC_RADIX + col1] = o1;
      ws[row1 * PFA7_TC_RADIX + col1] = o3;
    }
  }

  pfa7TcWarpSync();

#pragma unroll
  for (u32 k = 0; k < PFA7_TC_RADIX; ++k) {
    const Z31 v = ws[lane * PFA7_TC_RADIX + k];
    if (imag) u[k].y = v; else u[k].x = v;
  }

  // Protect the shared plane before the next component overwrites it.
  pfa7TcWarpSync();
}

inline void pfaDft7TensorGF31(GF31 *u, u32 inverseNormalized,
                              local Z31 *scratch) {
  pfa7TcComponent(u, inverseNormalized, scratch, 0u);
  pfa7TcComponent(u, inverseNormalized, scratch, 1u);

  // scratch is reused by the stock work-group transpose immediately after
  // this function.  Upgrade the final warp-local ordering to a CTA barrier.
  bar();
}

#undef PFA7_TC_RADIX
#undef PFA7_TC_LIMBS
#undef PFA7_TC_BASE_BITS
#undef PFA7_TC_WARP_WORDS
