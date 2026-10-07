// Host test: Marin transform sizes must keep every convolution coefficient below the NTT prime.
//
// Digits are < 2^(w + 1), the IBDWT weight contributes a factor 1 or 2 to each product and n products
// are summed, so the largest coefficient is n * 2 * (2^(w + 1) - 1)^2 and must be < MOD_P = 2^64 - 2^32 + 1.
// This test evaluates that bound exactly (128-bit) for every exponent in a sweep, checks that
// ibdwt::transform_size() returns the smallest valid transform, and pins the exponents where radix-5
// sizes used to wrap.
#include <cstdint>
#include <cstdio>

#include "marin/ibdwt.h"

namespace {

constexpr unsigned __int128 MODP = (unsigned __int128)0xFFFFFFFF00000001ull;

// worst-case coefficient of an n-point transform with digit width up to floor(q / n) + 1 (the code's w + 1)
bool valid(const size_t n, const uint32_t q)
{
	const unsigned __int128 w1 = q / n + 1;
	if (w1 > 40) return false;
	const unsigned __int128 d = ((unsigned __int128)1 << w1) - 1;
	return (unsigned __int128)n * 2 * d * d < MODP;
}

// smallest valid size among 2^k (k in [2, 26]) and 5 * 2^k (k in [3, 26])
size_t expected_size(const uint32_t q)
{
	size_t best = 0;
	for (unsigned k = 2; k <= 26; ++k) { const size_t n = size_t(1) << k; if (valid(n, q)) { best = n; break; } }
	for (unsigned k = 3; k <= 26; ++k) { const size_t n = size_t(5) << k; if (valid(n, q)) { if (best == 0 || n < best) best = n; break; } }
	return best;
}

int fails = 0;

// ibdwt::max_small_multiplier must be the largest a (up to 2^32 - 1) with  Lmax * a <= (2^64 - 1) * (2^w - 1),
// Lmax = 2 * n * (2^(w + 1) - 1)^2, w = floor(q / n): the carry of the multiply-by-a kernels then stays below 2^64.
// (For w = 0 the carry does not decay over the n digits, so Lmax is replaced by n * Lmax.)
void check_multiplier_bound(const uint32_t q)
{
	const size_t n = ibdwt::transform_size(q);
	const uint32_t cap = ibdwt::max_small_multiplier(n, q);
	const unsigned __int128 w = q / n, d = ((unsigned __int128)2 << w) - 1;
	unsigned __int128 lmax = (unsigned __int128)n * 2 * d * d;
	if (w == 0) lmax *= n;
	const unsigned __int128 decay = (w == 0) ? 1 : (((unsigned __int128)1 << w) - 1);
	const unsigned __int128 u64max = ~(uint64_t)0;
	const bool sound = lmax * cap <= u64max * decay;
	const bool tight = cap == 0xffffffffu || lmax * ((unsigned __int128)cap + 1) > u64max * decay;
	if (!sound || !tight || cap < 3) { ++fails; std::printf("q=%u n=%zu: max_small_multiplier=%u (sound=%d tight=%d)\n", q, n, cap, sound, tight); }
}

void check(const uint32_t q)
{
	const size_t got = ibdwt::transform_size(q), want = expected_size(q);
	if (got != want || !valid(got, q)) { ++fails; std::printf("q=%u: transform_size=%zu, expected %zu\n", q, got, want); }
}

void expect(const uint32_t q, const size_t n)
{
	const size_t got = ibdwt::transform_size(q);
	if (got != n) { ++fails; std::printf("q=%u: transform_size=%zu, expected %zu\n", q, got, n); }
}

}	// namespace

int main()
{
	// exponents where the old radix-5 bound let a coefficient reach MOD_P: they now take the next size
	struct { uint32_t q; size_t n_old, n_new; } wrap[] = {
		{ 1153u, 40, 64 }, { 4451u, 160, 256 }, { 17153u, 640, 1024 }, { 66049u, 2560, 4096 }, { 253953u, 10240, 16384 },
		{ 974849u, 40960, 65536 }, { 3735553u, 163840, 262144 }, { 14286849u, 655360, 1048576 },
		{ 54525957u, 2621440, 4194304 }, { 207618067u, 10485760, 16777216 } };
	for (const auto & e : wrap)
	{
		expect(e.q, e.n_new);
		expect(e.q - 1, e.n_new);
		if (valid(e.n_old, e.q)) { ++fails; std::printf("q=%u: size %zu must be invalid\n", e.q, e.n_old); }
	}

	// the largest exponent that still fits each radix-5 size is accepted at the smaller size, the next one is not
	struct { uint32_t q; size_t n; } edge[] = { { 1119u, 40 }, { 4319u, 160 }, { 16639u, 640 }, { 63999u, 2560 }, { 245759u, 10240 }, { 199229439u, 10485760 } };
	for (const auto & e : edge)
	{
		expect(e.q, e.n);
		if (ibdwt::transform_size(e.q + 1) == e.n) { ++fails; std::printf("q=%u: size %zu must not be selected\n", e.q + 1, e.n); }
	}

	// exact agreement with the 128-bit bound on a dense sweep and a coarse sweep up to the 2^26 limit
	for (uint32_t q = 3; q < 2000000u; ++q) check(q);
	for (uint64_t q = 2000000u; q < 900000000u; q += 99991u) check(uint32_t(q));
	for (const auto & e : wrap) for (int d = -3; d <= 3; ++d) check(e.q + d);

	// the multiplier bound used by the carry kernels (default Gaussian-Mersenne base is 3)
	for (uint32_t q = 3; q < 2000000u; q += 7) check_multiplier_bound(q);
	for (uint64_t q = 2000000u; q < 900000000u; q += 99991u) check_multiplier_bound(uint32_t(q));
	for (const auto & e : wrap) for (int d = -3; d <= 3; ++d) check_multiplier_bound(e.q + d);
	// known values: n = 4096, q = 86243 (w = 21) and n = 32768, q = 756839 (w = 23)
	if (ibdwt::max_small_multiplier(4096, 86243) != 268435455u) { ++fails; std::printf("q=86243: unexpected bound %u\n", ibdwt::max_small_multiplier(4096, 86243)); }
	if (ibdwt::max_small_multiplier(32768, 756839) != 8388607u) { ++fails; std::printf("q=756839: unexpected bound %u\n", ibdwt::max_small_multiplier(32768, 756839)); }

	std::printf("Marin IBDWT size bound test: %s\n", fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
