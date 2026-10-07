// Device test: Marin engine with the split root/weight kernel ABI forced
// (PRMERS_MARIN_SPLIT_AUX_FORCE), compared with GMP.
//
// Forcing split-aux used to abort with CL_INVALID_KERNEL_NAME (mul512_xbuf) for every
// transform size below 32768 words, because the host asked for a kernel that is only
// compiled together with mul512 (N_SZ >= 32768). This test builds the engine with the
// split ABI for small and large transform sizes, runs the PRP recurrence x <- 3 * x^2
// from x = 3 (so x = 3^(2^k - 1) after k steps) and checks it against GMP modulo 2^q - 1.
// For Mersenne prime exponents the full run must end on 3^(2^q - 1) = 3 (mod 2^q - 1);
// for a composite 2^q - 1 it must not.
//
// Build/run: bash tests/test_marin_split_aux_prp_device.sh [device-index] [q ...]
#define CL_TARGET_OPENCL_VERSION 120
#include "marin/engine_gpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Run `iters` steps on the device and with GMP; return true if they agree.
bool check_prefix(const uint32_t q, const size_t device, const uint32_t iters)
{
	engine_gpu_flat eng(q, 3, device, false);
	mpz_t M, x, r;
	mpz_init(M); mpz_init(x); mpz_init(r);
	mpz_ui_pow_ui(M, 2, q); mpz_sub_ui(M, M, 1);
	mpz_set_ui(x, 3);
	eng.set(0, 3);
	for (uint32_t i = 0; i < iters; ++i)
	{
		eng.square_mul(0, 3);
		mpz_mul(x, x, x); mpz_mul_ui(x, x, 3); mpz_mod(x, x, M);
	}
	eng.get_mpz(r, 0); mpz_mod(r, r, M);
	const bool ok = (mpz_cmp(r, x) == 0);
	mpz_clear(M); mpz_clear(x); mpz_clear(r);
	return ok;
}

// Full PRP of 2^q - 1 on the device: 3^(2^q - 1) = 3 (mod 2^q - 1) <=> probable prime.
bool device_prp(const uint32_t q, const size_t device)
{
	engine_gpu_flat eng(q, 3, device, false);
	mpz_t r; mpz_init(r);
	eng.set(0, 3);
	for (uint32_t i = 1; i < q; ++i) eng.square_mul(0, 3);
	eng.get_mpz(r, 0);
	mpz_t M; mpz_init(M); mpz_ui_pow_ui(M, 2, q); mpz_sub_ui(M, M, 1);
	mpz_mod(r, r, M);
	const bool prp = (mpz_cmp_ui(r, 3) == 0);
	mpz_clear(M); mpz_clear(r);
	return prp;
}

} // namespace

int main(int argc, char ** argv)
{
	setenv("PRMERS_MARIN_SPLIT_AUX_FORCE", "1", 1);

	const size_t device = (argc > 1) ? (size_t)std::strtoul(argv[1], nullptr, 10) : 0;
	// Mersenne prime exponents (full PRP) ...
	std::vector<uint32_t> primes = { 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 86243 };
	// ... composite 2^q - 1 (q prime and not a Mersenne exponent), and larger sizes
	// (transform >= 32768 words, where mul512_xbuf does exist) checked on a prefix only.
	std::vector<uint32_t> composites = { 11, 23, 29, 67, 101, 4421 };
	std::vector<uint32_t> prefixes = { 13, 4423, 86243, 756839, 1257787 };
	if (argc > 2)
	{
		primes.clear(); composites.clear(); prefixes.clear();
		for (int i = 2; i < argc; ++i) prefixes.push_back((uint32_t)std::strtoul(argv[i], nullptr, 10));
	}

	int fails = 0;
	for (const uint32_t q : primes)
	{
		const bool ok = device_prp(q, device);
		std::printf("split-aux PRP M%u: %s\n", q, ok ? "prime (ok)" : "NOT prime (FAIL)");
		if (!ok) ++fails;
	}
	for (const uint32_t q : composites)
	{
		const bool prp = device_prp(q, device);
		std::printf("split-aux PRP M%u: %s\n", q, prp ? "prime (FAIL)" : "composite (ok)");
		if (prp) ++fails;
	}
	for (const uint32_t q : prefixes)
	{
		const uint32_t iters = std::min<uint32_t>(q - 1, 200);
		const bool ok = check_prefix(q, device, iters);
		std::printf("split-aux prefix q=%u n=%zu iters=%u: %s\n", q, ibdwt::transform_size(q), iters, ok ? "ok" : "MISMATCH (FAIL)");
		if (!ok) ++fails;
	}
	std::printf("%s\n", fails ? "FAILED" : "all passed");
	return fails ? 1 : 0;
}
