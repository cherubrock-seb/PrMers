// Host test: the helpers in include/marin/arith.h (and ibdwt::weights_widths) against exact 128-bit
// references, and their range checks. Each setup-time helper rejects an input outside its domain instead
// of silently returning a wrong value (division by zero, undefined shifts, a root of the wrong order).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "marin/ibdwt.h"

namespace {

int fails = 0;

void check(const bool ok, const char * const what)
{
	if (!ok) { std::printf("FAIL: %s\n", what); ++fails; }
}

template <typename F> bool throws(F f)
{
	try { f(); } catch (const std::invalid_argument &) { return true; } catch (const std::logic_error &) { return true; }
	return false;
}

typedef unsigned __int128 u128;

uint64 ref_mul(const uint64 a, const uint64 b) { return uint64((u128(a) * b) % MOD_P); }

uint64 ref_pow(uint64 a, uint64 e)
{
	uint64 r = 1;
	for (; e != 0; e >>= 1) { if (e & 1) r = ref_mul(r, a); a = ref_mul(a, a); }
	return r;
}

}

int main()
{
	std::mt19937_64 rng(12345);

	// --- field arithmetic against 128-bit references
	{
		const uint64 edge[] = { 0, 1, 2, MOD_P - 1, MOD_P - 2, (1ull << 32), (1ull << 32) - 1, (1ull << 32) + 1, 1ull << 63 };
		for (const uint64 a : edge)
			for (const uint64 b : edge)
			{
				check(mod_mul(a, b) == ref_mul(a, b), "mod_mul edge values");
				check(mod_add(a, b) == uint64((u128(a) + b) % MOD_P), "mod_add edge values");
				check(mod_sub(a, b) == uint64((u128(a) + MOD_P - b) % MOD_P), "mod_sub edge values");
			}
		for (int i = 0; i < 20000; ++i)
		{
			const uint64 a = rng() % MOD_P, b = rng() % MOD_P;
			check(mod_mul(a, b) == ref_mul(a, b), "mod_mul random");
			check(mod_sqr(a) == ref_mul(a, a), "mod_sqr random");
			check(mod_pow(a, b) == ref_pow(a, b), "mod_pow random");
			check(mod_muli(a) == ref_mul(a, 1ull << 48), "mod_muli is multiplication by 2^48");
			check(mod_add(mod_half(a), mod_half(a)) == a, "mod_half doubles back");
		}
		check(mod_pow(0, 0) == 1 && mod_pow(5, 0) == 1 && mod_pow(0, 5) == 0, "mod_pow small exponents");
	}

	// --- mod_invert: [1, p) only
	{
		check(throws([] { mod_invert(0); }), "mod_invert(0) is rejected");
		check(throws([] { mod_invert(MOD_P); }), "mod_invert(p) is rejected");
		check(throws([] { mod_invert(MOD_P + 1); }), "mod_invert(p + 1) is rejected");
		check(throws([] { mod_invert(~uint64(0)); }), "mod_invert(2^64 - 1) is rejected");
		check(mod_invert(1) == 1 && mod_invert(MOD_P - 1) == MOD_P - 1, "mod_invert(1), mod_invert(p - 1)");
		for (int i = 0; i < 2000; ++i)
		{
			const uint64 a = rng() % (MOD_P - 1) + 1;
			check(mod_mul(a, mod_invert(a)) == 1, "a * a^-1 == 1");
		}
	}

	// --- mod_root_nth: n | p - 1 only; the root is primitive
	{
		check(throws([] { mod_root_nth(0); }), "mod_root_nth(0) is rejected");
		check(throws([] { mod_root_nth(7); }), "mod_root_nth(7) is rejected (7 does not divide p - 1)");
		check(throws([] { mod_root_nth(25); }), "mod_root_nth(25) is rejected");
		check(throws([] { mod_root_nth(1ull << 33); }), "mod_root_nth(2^33) is rejected");
		check(throws([] { mod_root_nth(MOD_P); }), "mod_root_nth(p) is rejected");
		check(throws([] { mod_root_nth(~uint64(0)); }), "mod_root_nth(2^64 - 1) is rejected");
		check(mod_root_nth(1) == 1, "mod_root_nth(1) == 1");
		std::vector<uint64> ns = { 2, 3, 4, 5, 10, 15, 17, 257, 65537, 5ull << 26, 1ull << 26, 1ull << 32, 3ull << 32 };
		for (const uint64 n : ns)
		{
			const uint64 r = mod_root_nth(n);
			check(ref_pow(r, n) == 1, "root^n == 1");
			uint64 m = n;
			for (uint64 q = 2; q <= 65537 && m > 1; ++q)
				if (m % q == 0)
				{
					check(ref_pow(r, n / q) != 1, "root is primitive");
					while (m % q == 0) m /= q;
				}
		}
	}

	// --- adc: width in [1, 31], exact against a 128-bit reference
	{
		for (const unsigned w : { 0u, 32u, 33u, 63u, 64u, 255u })
			check(throws([w] { uint64 c = 0; adc(1, uint8(w), c); }), "adc rejects an unsupported width");
		const uint64 edge[] = { 0, 1, 2, (1ull << 32) - 1, 1ull << 32, ~uint64(0), ~uint64(0) - 1, 1ull << 63 };
		for (unsigned w = 1; w <= 31; ++w)
		{
			const auto one = [&](const uint64 lhs, const uint64 cin) {
				uint64 c = cin;
				const uint32 r = adc(lhs, uint8(w), c);
				const u128 t = u128(lhs) + cin;
				check(r == uint32(t & ((u128(1) << w) - 1)), "adc digit");
				check(c == uint64(t >> w), "adc carry out");
			};
			for (const uint64 a : edge) for (const uint64 b : edge) one(a, b);
			for (int i = 0; i < 300; ++i) one(rng(), rng() >> (rng() % 64));
		}
	}

	// --- adc_mul: lhs * a + carry_in, within the documented domain (the carry stays below 2^64)
	{
		for (unsigned w = 1; w <= 31; ++w)
			for (int i = 0; i < 300; ++i)
			{
				const uint64 lhs = rng() >> (rng() % 40 + 24);	// < 2^40
				const uint32 a = uint32(rng());
				const uint64 cin = rng() >> (rng() % 40 + 24);
				const u128 t = u128(lhs) * a + cin;
				if ((t >> w) > ~uint64(0)) continue;
				uint64 c = cin;
				const uint32 r = adc_mul(lhs, a, uint8(w), c);
				check(r == uint32(t & ((u128(1) << w) - 1)), "adc_mul digit");
				check(c == uint64(t >> w), "adc_mul carry out");
			}
		check(throws([] { uint64 c = 0; adc_mul(1, 3, 0, c); }), "adc_mul rejects width 0");
		check(throws([] { uint64 c = 0; adc_mul(1, 3, 32, c); }), "adc_mul rejects width 32");
	}

	// --- ibdwt::weights_widths: supported sizes only
	{
		std::vector<uint64> w(2 * 1024, 0);
		std::vector<uint8> width(1024, 0);
		const uint32_t q = 100003;
		ibdwt::weights_widths(1024, q, w.data(), width.data());
		uint64 sum = 0;
		for (const uint8 x : width) sum += x;
		check(sum == q, "digit widths add up to the exponent");
		for (const size_t n : { size_t(0), size_t(3), size_t(25), size_t(1) << 27, size_t(1) << 33 })
			check(throws([&] { ibdwt::weights_widths(n, q, w.data(), width.data()); }), "weights_widths rejects a size not dividing (p-1)/192");
	}

	if (fails == 0) std::printf("marin arith host test passed\n");
	return fails == 0 ? 0 : 1;
}
