// Device test: Marin square_mul / mul with a large multiplier (the -gm-base value).
//
// The carry kernels multiply every convolution coefficient by `a` and keep the carry in 64 bits, so
// a multiplier above ibdwt::max_small_multiplier(n, q) can overflow the carry and give a wrong
// residue. This test checks, against GMP modulo 2^q - 1, that
//   * every multiplier up to the bound is exact, for random residues and for the worst-case digits
//     (every digit at 2^(w+1) - 1), and
//   * a multiplier above the bound is rejected (an exception) rather than silently wrong.
//
// Build/run: bash tests/test_marin_adc_mul_large_base_device.sh [device-index] [q ...]
#define CL_TARGET_OPENCL_VERSION 120
#include "marin/engine_gpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

void digits_to_mpz(mpz_t r, const std::vector<uint64> & d, const std::vector<uint8> & wd)
{
	mpz_set_ui(r, 0);
	mpz_t t; mpz_init(t);
	size_t pos = 0;
	for (size_t k = 0; k < d.size(); ++k)
	{
		mpz_set_ui(t, (unsigned long)d[k]);
		mpz_mul_2exp(t, t, pos);
		mpz_add(r, r, t);
		pos += wd[k];
	}
	mpz_clear(t);
}

struct Tester
{
	uint32_t q;
	size_t n;
	uint32_t cap;
	engine_gpu_flat * eng;
	std::vector<uint8> wd;
	mpz_t M;
	int fails = 0, checks = 0, rejected = 0;

	Tester(const uint32_t q_, const size_t device) : q(q_), n(ibdwt::transform_size(q_)), cap(ibdwt::max_small_multiplier(n, q_))
	{
		eng = new engine_gpu_flat(q, 3, device, false);
		std::vector<uint64> w(2 * n);
		wd.resize(n);
		ibdwt::weights_widths(n, q, w.data(), wd.data());
		mpz_init(M); mpz_ui_pow_ui(M, 2, q); mpz_sub_ui(M, M, 1);
	}
	~Tester() { mpz_clear(M); delete eng; }

	// Run `op` for multiplier a on the register holding X (digits given or random mpz) and compare with GMP.
	// Returns 0 = exact, 1 = wrong result, 2 = rejected by the engine.
	int run(const char * const what, mpz_srcptr Xs, std::vector<uint64> * const digits, const uint32_t a, const bool use_mul, mpz_srcptr Ys)
	{
		mpz_t e, r, X, Y; mpz_inits(e, r, nullptr); mpz_init_set(X, Xs); mpz_init_set(Y, Ys);
		int outcome = 0;
		try
		{
			if (digits) eng->set(0, digits->data()); else eng->set_mpz(0, X);
			if (use_mul)
			{
				eng->set_mpz(1, Y);
				eng->set_multiplicand(1, 1);
				eng->mul(0, 1, a);
				mpz_mul(e, X, Y); mpz_mul_ui(e, e, a);
			}
			else
			{
				eng->square_mul(0, a);
				mpz_mul(e, X, X); mpz_mul_ui(e, e, a);
			}
			mpz_mod(e, e, M);
			eng->get_mpz(r, 0); mpz_mod(r, r, M);
			outcome = (mpz_cmp(r, e) == 0) ? 0 : 1;
		}
		catch (const std::exception & ex)
		{
			outcome = 2;
			if (a <= cap) std::printf("  q=%u %s a=%u: unexpected exception: %s\n", q, what, a, ex.what());
		}
		mpz_clears(e, r, X, Y, nullptr);

		++checks;
		if (outcome == 2) ++rejected;
		const bool ok = (a <= cap) ? (outcome == 0) : (outcome != 1);	// above the bound: rejected is fine, a wrong residue is not
		if (!ok)
		{
			++fails;
			std::printf("  q=%u n=%zu %-26s a=%-10u bound=%-10u %s\n", q, n, what, a, cap,
				outcome == 1 ? "WRONG RESULT (no error)" : "FAILED");
		}
		return outcome;
	}

	void run()
	{
		std::mt19937_64 rng(q);
		const size_t w = q / n;

		std::set<uint32_t> mults = { 2, 3, 5, 7, 1000, 65537, 1000003 };
		for (int s = 0; s < 4; ++s) mults.insert(cap - (uint32_t)(rng() % 3));	// at and just below the bound
		mults.insert(cap / 2 + 1);
		if (cap < 0xffffffffu) { mults.insert(cap + 1); mults.insert(cap + 2); mults.insert(cap + (0xffffffffu - cap) / 2); }
		mults.insert(0x80000000u); mults.insert(0xffffffffu);

		mpz_t X, Y; mpz_inits(X, Y, nullptr);
		gmp_randstate_t st; gmp_randinit_default(st); gmp_randseed_ui(st, q);

		// worst case: every digit at 2^(w+1) - 1 (the bound the transform size is chosen for)
		std::vector<uint64> worst(n, (uint64(1) << (w + 1)) - 1);
		mpz_t Xw; mpz_init(Xw);
		digits_to_mpz(Xw, worst, wd); mpz_mod(Xw, Xw, M);

		for (const uint32_t a : mults)
		{
			if (a < 2) continue;
			mpz_urandomm(X, st, M); mpz_urandomm(Y, st, M);
			run("square_mul random", X, nullptr, a, false, Y);
			run("square_mul worst-case digits", Xw, &worst, a, false, Y);
			run("mul random", X, nullptr, a, true, Y);
		}
		mpz_clears(X, Y, Xw, nullptr);
		gmp_randclear(st);
		std::printf("q=%u n=%zu w=%zu bound=%u: %d/%d checks failed (%d rejected above the bound)\n", q, n, w, cap, fails, checks, rejected);
	}
};

}	// namespace

int main(int argc, char ** argv)
{
	const size_t device = (argc > 1) ? (size_t)std::strtoul(argv[1], nullptr, 10) : 0;
	std::vector<uint32_t> qs = { 127u, 1279u, 4423u, 86243u, 132049u, 756839u, 1257787u };
	if (argc > 2) { qs.clear(); for (int i = 2; i < argc; ++i) qs.push_back((uint32_t)std::strtoul(argv[i], nullptr, 10)); }

	int fails = 0;
	for (const uint32_t q : qs)
	{
		Tester t(q, device);
		t.run();
		fails += t.fails;
	}
	std::printf(fails == 0 ? "Marin large multiplier device test passed\n" : "Marin large multiplier device test FAILED\n");
	return fails == 0 ? 0 : 1;
}
