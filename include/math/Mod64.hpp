#pragma once

#include <cstdint>

namespace math {

static constexpr uint64_t MOD_P = (((1ULL << 32) - 1ULL) << 32) + 1ULL;

uint64_t mulModP(uint64_t a, uint64_t b);
uint64_t powModP(uint64_t base, uint64_t exp);
uint64_t invModP(uint64_t x);

} // namespace math
