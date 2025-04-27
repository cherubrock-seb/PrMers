#include "math/Mod64.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace math {

uint64_t mulModP(uint64_t a, uint64_t b) {
    uint64_t hi, lo;

#ifdef _MSC_VER
    lo = _umul128(a, b, &hi);
#else
    __uint128_t prod = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    hi = static_cast<uint64_t>(prod >> 64);
    lo = static_cast<uint64_t>(prod);
#endif

    uint32_t A = static_cast<uint32_t>(hi >> 32);
    uint32_t B = static_cast<uint32_t>(hi & 0xFFFFFFFFULL);

    uint64_t r = lo;
    uint64_t old = r;
    r += (static_cast<uint64_t>(B) << 32);
    if (r < old) r += ((1ULL << 32) - 1ULL);

    uint64_t sub = static_cast<uint64_t>(A) + static_cast<uint64_t>(B);
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
