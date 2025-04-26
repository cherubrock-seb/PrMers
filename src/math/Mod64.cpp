#include "math/Mod64.hpp"

namespace math {

uint64_t mulModP(uint64_t a, uint64_t b) {
#ifdef _MSC_VER
    uint64_t hi, lo = _umul128(a, b, &hi);
#else
    __uint128_t prod = ( __uint128_t )a * b;
    uint64_t hi = uint64_t(prod >> 64), lo = uint64_t(prod);
#endif
    uint32_t A = uint32_t(hi >> 32), B = uint32_t(hi & 0xFFFFFFFFULL);
    uint64_t r = lo, old = r;
    r += (uint64_t(B) << 32);
    if (r < old) r += ((1ULL << 32) - 1ULL);
    uint64_t sub = uint64_t(A) + uint64_t(B);
    r = (r >= sub ? r - sub : r + MOD_P - sub);
    return (r >= MOD_P ? r - MOD_P : r);
}

uint64_t powModP(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp) {
        if (exp & 1) result = mulModP(result, base);
        base = mulModP(base, base);
        exp >>= 1;
    }
    return result;
}

uint64_t invModP(uint64_t x) {
    return powModP(x, MOD_P - 2);
}

} // namespace math
