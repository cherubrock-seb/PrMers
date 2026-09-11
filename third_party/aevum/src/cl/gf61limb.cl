// Exact implementation of the existing GF(2^61-1) products. No 128-bit temporary.
// Shared verbatim with the C++ arithmetic regression test.
inline u64 aevumCanonical61(u64 x) {
  const u64 mask = 0x1fffffffffffffffULL;
  x = (x & mask) + (x >> 61); // <= mask+7 for every unsigned 64-bit input
  return x >= mask ? x-mask : x;
}

// Inputs canonical: low limbs <2^31, high limbs <2^30. Their sums fit u32.
inline u64 aevumMul61Canonical(u64 a, u64 b) {
  const u32 mask31 = 0x7fffffffU;
  const u64 mask61 = 0x1fffffffffffffffULL;
  const u32 a0 = (u32)a & mask31, a1 = (u32)(a >> 31);
  const u32 b0 = (u32)b & mask31, b1 = (u32)(b >> 31);
  const u64 lo = (u64)a0 * b0;
  const u64 hi = (u64)a1 * b1;
  const u64 cross = (u64)(a0+a1) * (b0+b1) - lo - hi;
  // Fold 2^61 to 1 and 2^62 to 2 before summing. Total <3*2^61+2^32.
  const u64 folded = (lo & mask61) + (lo >> 61) +
      ((cross & 0x3fffffffULL) << 31) + (cross >> 30) + (hi << 1);
  return aevumCanonical61(folded);
}
inline u64 aevumMul61(u64 a, u64 b) {
  return aevumMul61Canonical(aevumCanonical61(a), aevumCanonical61(b));
}
// Complex multiplication uses three of the same exact scalar products.
inline u64 aevumCmul61Real(u64 p, u64 q) {
  return aevumCanonical61(p + 0x1fffffffffffffffULL - q);
}
inline u64 aevumCmul61Imag(u64 p, u64 q, u64 r) {
  return aevumCanonical61(r + 0x3ffffffffffffffeULL - p - q);
}
