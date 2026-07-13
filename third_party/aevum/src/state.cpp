// Copyright 2017 Mihai Preda.

#include "state.h"
#include "shared.h"
#include "log.h"
#include "timeutil.h"

#include <algorithm>
#include <cassert>
#include <cmath>

static i64 lowBits(i64 u, int bits) { return (u << (64 - bits)) >> (64 - bits); }

std::vector<u32> compactBits(const vector<Word> &dataVect, u64 E) {
  if (dataVect.empty()) { return {}; } // Indicating all zero

  u32 N = dataVect.size();
  const Word *data = dataVect.data();

  std::vector<u32> out;
  out.reserve((E - 1) / 32 + 1);

  int carry = 0;
  u32 outWord = 0;
  int haveBits = 0;

  // Convert to compact form
  for (u32 p = 0; p < N; ++p) {
    int nBits = bitlen(N, E, p);
    assert(nBits > 0);

    //   Be careful adding in the carry -- it could overflow a 32-bit word.  Convert value into desired unsigned range.
    i64 tmp = (i64) data[p] + carry;
    carry = (int) (tmp >> nBits);
    u64 w = (u64) (tmp - ((i64) carry << nBits));
    assert(w < (1ULL << nBits));

    assert(haveBits < 32);
    while (nBits) {
      int needBits = 32 - haveBits;
      outWord |= w << haveBits;
      if (nBits >= needBits) {
        w >>= needBits;
        nBits -= needBits;
        out.push_back(outWord);
        outWord = 0;
        haveBits = 0;
      } else {
        haveBits += nBits;
        w >>= nBits;
        break;
      }
    }
  }

  assert(haveBits);
  out.push_back(outWord);

  const u32 topBits = u32(E & 31u);
  const u32 topMask = topBits ? ((u32(1) << topBits) - 1u) : 0xffffffffu;
  if (topBits) out.back() &= topMask;

  auto is_zero = [&]() {
    return std::all_of(out.begin(), out.end(), [](u32 v) { return v == 0u; });
  };

  auto is_modulus = [&]() {
    for (size_t i = 0; i + 1 < out.size(); ++i) {
      if (out[i] != 0xffffffffu) return false;
    }
    return out.back() == topMask;
  };

  auto set_modulus_minus_one = [&]() {
    for (size_t i = 0; i + 1 < out.size(); ++i) out[i] = 0xffffffffu;
    out.back() = topMask;
    out[0] -= 1u;
  };

  auto increment_mod_mersenne = [&]() {
    if (is_modulus()) std::fill(out.begin(), out.end(), 0u);
    u64 c = 1;
    for (size_t i = 0; i < out.size() && c; ++i) {
      const u64 v = u64(out[i]) + c;
      out[i] = u32(v);
      c = v >> 32;
    }
    if (topBits) out.back() &= topMask;
    if (c || is_modulus()) std::fill(out.begin(), out.end(), 0u);
  };

  auto decrement_mod_mersenne = [&]() {
    if (is_zero()) {
      set_modulus_minus_one();
      return;
    }
    u64 borrow = 1;
    for (size_t i = 0; i < out.size() && borrow; ++i) {
      const u32 before = out[i];
      out[i] = before - 1u;
      borrow = before == 0u;
    }
    if (topBits) out.back() &= topMask;
  };

  if (is_modulus()) std::fill(out.begin(), out.end(), 0u);
  if (carry > 0) {
    for (u32 n = u32(carry); n != 0; --n) increment_mod_mersenne();
  } else if (carry < 0) {
    for (u32 n = u32(-i64(carry)); n != 0; --n) decrement_mod_mersenne();
  }

  assert(out.size() == (E - 1) / 32 + 1);
  return out;
}

struct BitBucket {
  u128 bits;
  u32 size;

  BitBucket() : bits(0), size(0) {}

  void put32(u32 b) {
    assert(size <= 96);
    bits += (u128(b) << size);
    size += 32;
  }
  
  i64 popSigned(u32 n) {
    assert(size >= n);
    i64 b = lowBits((i64) bits, n);
    size -= n;
    bits >>= n;
    bits += (b < 0); // carry fixup.
    return b;
  }
};

vector<Word> expandBits(const vector<u32> &compactBits, u32 N, u64 E) {
  assert(E % 32 != 0);

  std::vector<Word> out(N);
  Word *data = out.data();
  BitBucket bucket;
  
  auto it = compactBits.cbegin();
  [[maybe_unused]] auto itEnd = compactBits.cend();
  for (u32 p = 0; p < N; ++p) {
    u32 len = bitlen(N, E, p);
    while (bucket.size < len) {
      assert(it != itEnd);
      bucket.put32(*it++);
    }
    data[p] = (Word) bucket.popSigned(len);
  }
  assert(it == itEnd);
  assert(bucket.size == 32 - E % 32);
  assert(bucket.bits == 0 || bucket.bits == 1);
  data[0] += u32(bucket.bits); // carry wrap-around.
  return out;
}
