// Device test: Marin squaring at the exponents where radix-5 transform sizes used to wrap.
//
// Squares (-1) = 2^q - 2 (near-maximal digits) and random residues on the OpenCL device with the transform
// size chosen by ibdwt::transform_size(), and compares with GMP modulo 2^q - 1.
//
// Build/run: bash tests/test_marin_ibdwt_wrap_device.sh [device-index] [q ...]
#define CL_TARGET_OPENCL_VERSION 120
#include "marin/engine_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv)
{
	const size_t device = (argc > 1) ? (size_t)std::strtoul(argv[1], nullptr, 10) : 0;
	std::vector<uint32_t> qs = {1153u, 1151u, 4451u, 17153u};
	if (argc > 2) { qs.clear(); for (int i = 2; i < argc; ++i) qs.push_back((uint32_t)std::strtoul(argv[i], nullptr, 10)); }

	int fails = 0, total = 0;
	for (const uint32_t q : qs)
	{
		engine_gpu_flat eng(q, 6, device, false);
		mpz_t M, a, r, e; mpz_inits(M, a, r, e, nullptr);
		mpz_ui_pow_ui(M, 2, q); mpz_sub_ui(M, M, 1);

		// (-1)^2 = 1
		mpz_sub_ui(a, M, 1);
		eng.set_mpz(0, a);
		eng.square_mul(0, 1);
		eng.get_mpz(r, 0); mpz_mod(r, r, M);
		bool ok = (mpz_cmp_ui(r, 1) == 0);
		++total; if (!ok) ++fails;
		std::printf("q=%u n=%zu (-1)^2 mod 2^q-1: %s\n", q, ibdwt::transform_size(q), ok ? "ok" : "MISMATCH");

		// (-1)^2 * 3 = 3, then (3 * -1)^2 = 9
		eng.set_mpz(0, a);
		eng.square_mul(0, 3);
		eng.get_mpz(r, 0); mpz_mod(r, r, M);
		ok = (mpz_cmp_ui(r, 3) == 0);
		++total; if (!ok) ++fails;
		std::printf("q=%u n=%zu 3 * (-1)^2: %s\n", q, ibdwt::transform_size(q), ok ? "ok" : "MISMATCH");

		// random residues
		gmp_randstate_t st; gmp_randinit_default(st); gmp_randseed_ui(st, q);
		for (int it = 0; it < 20; ++it)
		{
			mpz_urandomm(a, st, M);
			eng.set_mpz(0, a);
			eng.square_mul(0, 1);
			eng.get_mpz(r, 0); mpz_mod(r, r, M);
			mpz_mul(e, a, a); mpz_mod(e, e, M);
			ok = (mpz_cmp(r, e) == 0);
			++total; if (!ok) ++fails;
			if (!ok) std::printf("q=%u random square MISMATCH\n", q);
		}
		gmp_randclear(st);
		mpz_clears(M, a, r, e, nullptr);
	}
	std::printf("Marin IBDWT wrap device test: %d/%d mismatches -> %s\n", fails, total, fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
