// Device test: Marin exact register subtraction (sub_reg / addsub) against GMP.
//
// The engine's digits are not always normalised: the carry kernels leave lane s3 of each
// block as d + c (adc4), so digits equal to their base 2^w appear after every square_mul.
// This test injects such digits (and the stress bound 2^(w+1) - 1), borrow chains that cross
// the 4*CWM_WG_SZ digit groups of subtract_reg_group_*, borrows wrapping through group 0, and
// natural square_mul-produced registers, and compares every result with GMP modulo 2^q - 1.
//
// Build/run: bash tests/test_marin_exact_subtraction_device.sh [device-index] [q ...]
#define CL_TARGET_OPENCL_VERSION 120
#include "marin/engine_gpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
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
	uint32_t q = 0;
	size_t n = 0, G = 0;
	engine_gpu_flat * eng = nullptr;
	std::vector<uint8> wd;
	mpz_t M;
	std::mt19937_64 rng;
	int fails = 0, total = 0;

	explicit Tester(const uint32_t q_, const size_t device) : q(q_), n(ibdwt::transform_size(q_)), rng(q_)
	{
		eng = new engine_gpu_flat(q, 6, device, false);
		const size_t n5 = (n % 5 == 0) ? n / 5 : n;
		size_t cwm = 1;
		while (cwm * 2 <= std::min<size_t>(n5 / 4, 256)) cwm *= 2;
		G = 4 * cwm;	// digits per group (upper bound; the device may use fewer)
		std::vector<uint64> w(2 * n);
		wd.resize(n);
		ibdwt::weights_widths(n, q, w.data(), wd.data());
		mpz_init(M); mpz_ui_pow_ui(M, 2, q); mpz_sub_ui(M, M, 1);
	}
	~Tester() { mpz_clear(M); delete eng; }

	uint64 B(const size_t k) const { return uint64(1) << wd[k]; }
	void random_digits(std::vector<uint64> & v) { for (size_t k = 0; k < n; ++k) v[k] = rng() & (B(k) - 1); }

	void report(const char * const name, const bool ok)
	{
		++total;
		if (!ok) ++fails;
		if (!ok) std::printf("  q=%u n=%zu %-40s MISMATCH\n", q, n, name);
	}

	// y -= x on the device (digits injected as given), compared with (Y - X) mod M
	bool sub(const char * const name, std::vector<uint64> y, std::vector<uint64> x)
	{
		mpz_t a, b, r, e; mpz_inits(a, b, r, e, nullptr);
		digits_to_mpz(a, y, wd); digits_to_mpz(b, x, wd);
		eng->set(0, y.data()); eng->set(1, x.data());
		eng->sub_reg(0, 1);
		eng->get_mpz(r, 0); mpz_mod(r, r, M);
		mpz_sub(e, a, b); mpz_mod(e, e, M);
		const bool ok = (mpz_cmp(r, e) == 0);
		report(name, ok);
		mpz_clears(a, b, r, e, nullptr);
		return ok;
	}

	// (sum, diff) = addsub(a, b) on the device, compared with GMP
	bool addsub(const char * const name, std::vector<uint64> y, std::vector<uint64> x)
	{
		mpz_t a, b, r, e; mpz_inits(a, b, r, e, nullptr);
		digits_to_mpz(a, y, wd); digits_to_mpz(b, x, wd);
		eng->set(0, y.data()); eng->set(1, x.data());
		eng->addsub(2, 3, 0, 1);
		bool ok = true;
		eng->get_mpz(r, 2); mpz_mod(r, r, M); mpz_add(e, a, b); mpz_mod(e, e, M); ok &= (mpz_cmp(r, e) == 0);
		eng->get_mpz(r, 3); mpz_mod(r, r, M); mpz_sub(e, a, b); mpz_mod(e, e, M); ok &= (mpz_cmp(r, e) == 0);
		report(name, ok);
		mpz_clears(a, b, r, e, nullptr);
		return ok;
	}

	// x -= a (small constant) on the device, compared with (X - a) mod M
	void sub_const(const char * const name, const mpz_t & X, const uint32 a)
	{
		mpz_t r, e; mpz_inits(r, e, nullptr);
		eng->set_mpz(0, X);
		eng->sub(0, a);
		eng->get_mpz(r, 0); mpz_mod(r, r, M);
		mpz_sub_ui(e, X, a); mpz_mod(e, e, M);
		const bool ok = (mpz_cmp(r, e) == 0);
		report(name, ok);
		mpz_clears(r, e, nullptr);
	}

	void run()
	{
		std::vector<uint64> x(n), y(n);

		// single-register subtraction of a small constant (the serial `subtract` kernel)
		{
			mpz_t X; mpz_init(X);
			mpz_set_ui(X, 10); sub_const("10 - 1", X, 1);
			mpz_set_ui(X, 0); sub_const("0 - 1", X, 1);
			mpz_set_ui(X, 3); sub_const("3 - 7", X, 7);
			mpz_sub_ui(X, M, 1); sub_const("(2^q-2) - 1", X, 1);
			gmp_randstate_t st; gmp_randinit_default(st); gmp_randseed_ui(st, q + 1);
			for (int it = 0; it < 20; ++it) { mpz_urandomm(X, st, M); sub_const("random - small", X, uint32(rng() % 3)); }
			gmp_randclear(st);
			mpz_clear(X);
		}

		auto zero = [&](std::vector<uint64> & v) { std::fill(v.begin(), v.end(), 0); };

		random_digits(x); random_digits(y); sub("normalised random", y, x);

		// borrow entering a digit whose subtrahend equals its base: every lane, including the top digit
		for (size_t k = 1; k < std::min<size_t>(n, 64); ++k)
		{
			random_digits(x); random_digits(y);
			x[k] = B(k); y[k] = 0; x[k - 1] = 5; y[k - 1] = 1;
			sub("x[k]=2^w with borrow in", y, x);
		}
		{ random_digits(x); random_digits(y); const size_t k = n - 1; x[k] = B(k); y[k] = 0; x[k - 1] = 5; y[k - 1] = 1; sub("top digit 2^w with borrow in", y, x); }

		// groups equal in value but not in representation must pass a borrow along
		if (n / G >= 3)
		{
			random_digits(x); y = x;
			for (size_t g = 1; g + 1 < n / G; ++g) { const size_t k = g * G + 3; x[k] = B(k); x[k + 1] = 0; y[k] = 0; y[k + 1] = 1; }
			x[0] = 5; y[0] = 1;
			sub("borrow through equal-value groups", y, x);

			// wrapped borrow: top group generates, group 0 is equal in value only
			random_digits(x); y = x;
			x[3] = B(3); x[4] = 0; y[3] = 0; y[4] = 1;
			x[n - 1] = 5; y[n - 1] = 1;
			sub("wrapped borrow through group 0", y, x);
		}

		// every lane s3 of x at its base (X may exceed 2^q), y small or zero
		random_digits(x); for (size_t k = 3; k < n; k += 4) x[k] = B(k);
		zero(y); sub("all lanes s3 of x = 2^w, y = 0", y, x);
		y[0] = 1; sub("all lanes s3 of x = 2^w, y = 1", y, x);
		random_digits(y); sub("all lanes s3 of x = 2^w, y random", y, x);
		for (size_t k = 3; k < n; k += 4) y[k] = B(k);
		sub("all lanes s3 of x and y = 2^w", y, x);

		// stress bound: every digit at 2^(w+1) - 1
		for (size_t k = 0; k < n; ++k) x[k] = 2 * B(k) - 1;
		zero(y); sub("x = 2^(w+1)-1 everywhere, y = 0", y, x);
		random_digits(y); sub("x = 2^(w+1)-1 everywhere, y random", y, x);
		sub("y = 2^(w+1)-1 everywhere, x random", x, y);

		// small values
		zero(x); zero(y); sub("0 - 0", y, x);
		x[0] = 1; sub("0 - 1", y, x);
		y[0] = 1; x[0] = 7; sub("1 - 7", y, x);
		random_digits(y); sub("y - y", y, y);
		for (size_t k = 0; k < n; ++k) x[k] = B(k) - 1;
		zero(y); sub("0 - (2^q-1)", y, x);

		// addsub (fast add + exact subtraction)
		random_digits(x); random_digits(y); addsub("addsub normalised", y, x);
		for (size_t k = 3; k < n; k += 4) { if (rng() & 1) x[k] = B(k); if (rng() & 1) y[k] = B(k); }
		addsub("addsub lanes s3 = 2^w", y, x);
		zero(y); addsub("addsub 0, x", y, x);

		// random sweep with producer-style digits on both operands
		for (int it = 0; it < 300; ++it)
		{
			random_digits(x); random_digits(y);
			for (size_t k = 3; k < n; k += 4) { if (rng() % 4 == 0) x[k] = B(k); if (rng() % 4 == 0) y[k] = B(k); }
			if (rng() % 8 == 0) for (size_t k = 0; k < n; ++k) if (rng() % 4 == 0) y[k] = 0;
			sub("random sweep", y, x);
		}

		// natural producers: square_mul(.., 3) leaves digits equal to 2^w; then subtract
		{
			mpz_t a, b, r, e; mpz_inits(a, b, r, e, nullptr);
			gmp_randstate_t st; gmp_randinit_default(st); gmp_randseed_ui(st, q);
			for (int it = 0; it < 100; ++it)
			{
				mpz_urandomm(a, st, M); mpz_urandomm(b, st, M);
				eng->set_mpz(0, a); eng->set_mpz(1, b);
				eng->square_mul(0, 3); eng->square_mul(1, 3);
				mpz_mul(a, a, a); mpz_mul_ui(a, a, 3); mpz_mod(a, a, M);
				mpz_mul(b, b, b); mpz_mul_ui(b, b, 3); mpz_mod(b, b, M);
				eng->sub_reg(0, 1);
				eng->get_mpz(r, 0); mpz_mod(r, r, M);
				mpz_sub(e, a, b); mpz_mod(e, e, M);
				report("square_mul then sub_reg", mpz_cmp(r, e) == 0);
			}
			gmp_randclear(st);
			mpz_clears(a, b, r, e, nullptr);
		}

		std::printf("q=%u n=%zu: %d/%d mismatches\n", q, n, fails, total);
	}
};

}	// namespace

int main(int argc, char ** argv)
{
	const size_t device = (argc > 1) ? (size_t)std::strtoul(argv[1], nullptr, 10) : 0;
	std::vector<uint32_t> qs = {13u, 31u, 61u, 119u, 127u, 521u, 1159u, 4423u, 86243u, 132049u};
	if (argc > 2) { qs.clear(); for (int i = 2; i < argc; ++i) qs.push_back((uint32_t)std::strtoul(argv[i], nullptr, 10)); }

	int fails = 0, total = 0;
	for (const uint32_t q : qs)
	{
		Tester t(q, device);
		t.run();
		fails += t.fails; total += t.total;
	}
	std::printf("Marin exact subtraction device test: %d/%d mismatches -> %s\n", fails, total, fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
