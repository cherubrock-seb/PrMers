// Device test for the legacy kernel_carry_mul_3 (math::Carry::carryGPU3): feed it digit vectors that are NOT
// normalised (as a raw inverse-NTT output is) and compare 3 * value mod 2^p - 1 with GMP.
// usage: legacy-carry-mul3-device-test [device-index] [p ...]
#define CL_TARGET_OPENCL_VERSION 120
#include "math/Precompute.hpp"
#include "math/Carry.hpp"
#include "opencl/Context.hpp"
#include "opencl/Program.hpp"
#include "opencl/Buffers.hpp"

#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#ifndef PRMERS_KERNEL_DIR
#define PRMERS_KERNEL_DIR "kernels"
#endif

static void to_digits(const mpz_t z, const std::vector<int> & w, std::vector<uint64_t> & x)
{
	x.assign(w.size(), 0);
	size_t bit = 0;
	for (size_t k = 0; k < w.size(); ++k)
	{
		uint64_t v = 0;
		for (int b = 0; b < w[k]; ++b) if (mpz_tstbit(z, bit + b)) v |= uint64_t(1) << b;
		x[k] = v; bit += w[k];
	}
}

// value of an arbitrary (not normalised) digit vector mod M
static void from_digits(mpz_t z, const std::vector<int> & w, const std::vector<uint64_t> & x, const mpz_t M)
{
	mpz_set_ui(z, 0);
	mpz_t t; mpz_init(t);
	size_t bit = 0;
	for (size_t k = 0; k < w.size(); ++k)
	{
		mpz_set_ui(t, (unsigned long)x[k]); mpz_mul_2exp(t, t, bit); mpz_add(z, z, t);
		bit += w[k];
	}
	mpz_clear(t);
	mpz_mod(z, z, M);
}

int main(int argc, char ** argv)
{
	size_t device = 0;
	int first = 1;
	if (argc > 1) { device = std::strtoull(argv[1], nullptr, 10); first = 2; }
	std::vector<uint64_t> ps = {127u, 1279u, 4423u, 9941u, 44497u};
	if (argc > first) { ps.clear(); for (int i = first; i < argc; ++i) ps.push_back(std::strtoull(argv[i], nullptr, 10)); }

	int fails = 0, total = 0;
	for (const uint64_t p : ps)
	{
		math::Precompute pre(p);
		const size_t n = pre.getN();
		const std::vector<int> & widths = pre.getDigitWidth();

		prmers::ocl::Context ctx(device, 0, false, false);
		ctx.computeOptimalSizes(n, widths, p, false, 0, 0);
		prmers::ocl::Program program(ctx, ctx.getDevice(), std::string(PRMERS_KERNEL_DIR) + "/prmers.cl", pre, "", false);
		opencl::Buffers buffers(ctx, pre);
		math::Carry carry(ctx, ctx.getQueue(), program.getProgram(), n, widths, buffers.digitWidthMaskBuf);

		cl_int err;
		cl_mem buf = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, n * sizeof(uint64_t), nullptr, &err);
		if (err != CL_SUCCESS) { std::fprintf(stderr, "clCreateBuffer failed\n"); return 2; }

		mpz_t M, a, r, e; mpz_inits(M, a, r, e, nullptr);
		mpz_ui_pow_ui(M, 2, (unsigned long)p); mpz_sub_ui(M, M, 1);

		std::vector<uint64_t> x;
		std::mt19937_64 rng(p);
		auto run = [&](const std::vector<uint64_t> & in, const char * what) {
			from_digits(a, widths, in, M);
			mpz_mul_ui(e, a, 3); mpz_mod(e, e, M);
			clEnqueueWriteBuffer(ctx.getQueue(), buf, CL_TRUE, 0, n * sizeof(uint64_t), in.data(), 0, nullptr, nullptr);
			carry.carryGPU3(buf, buffers.blockCarryBuf, n * sizeof(uint64_t));
			clFinish(ctx.getQueue());
			x.assign(n, 0);
			clEnqueueReadBuffer(ctx.getQueue(), buf, CL_TRUE, 0, n * sizeof(uint64_t), x.data(), 0, nullptr, nullptr);
			from_digits(r, widths, x, M);
			const bool ok = mpz_cmp(r, e) == 0;
			++total; if (!ok) { ++fails; std::printf("p=%llu n=%zu %s: MISMATCH\n", (unsigned long long)p, n, what); }
		};

		// normalised residues (what every caller passes)
		for (int it = 0; it < 10; ++it)
		{
			mpz_set_ui(a, 0);
			for (size_t k = 0; k < 4; ++k) { mpz_mul_2exp(a, a, 64); mpz_add_ui(a, a, (unsigned long)rng()); }
			mpz_mul_2exp(a, a, (unsigned long)p); mpz_mod(a, a, M);
			to_digits(a, widths, x);
			run(x, "normalised random");
		}
		// raw digits: every digit above its base by up to 6 bits (interior block carries are non-zero)
		for (int it = 0; it < 10; ++it)
		{
			x.assign(n, 0);
			for (size_t k = 0; k < n; ++k) x[k] = rng() & ((uint64_t(1) << (widths[k] + 6)) - 1);
			run(x, "raw digits (not normalised)");
		}
		// all digits exactly one above the largest legal value: every block boundary carries
		x.assign(n, 0);
		for (size_t k = 0; k < n; ++k) x[k] = uint64_t(1) << widths[k];
		run(x, "every digit = base");
		// every digit at its largest raw value (6 bits above its width)
		x.assign(n, 0);
		for (size_t k = 0; k < n; ++k) x[k] = (uint64_t(1) << (widths[k] + 6)) - 1;
		run(x, "every digit maximal raw");
		// a single carry chain: 1 + B + B^2 + ... in unit steps
		x.assign(n, 0);
		for (size_t k = 0; k < n; ++k) x[k] = (uint64_t(1) << widths[k]) - 1;
		x[0] += 1;
		run(x, "ripple carry through every digit");
		x.assign(n, 0);
		run(x, "zero");

		clReleaseMemObject(buf);
		mpz_clears(M, a, r, e, nullptr);
	}
	std::printf("Legacy kernel_carry_mul_3 device test: %d/%d mismatches -> %s\n", fails, total, fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
