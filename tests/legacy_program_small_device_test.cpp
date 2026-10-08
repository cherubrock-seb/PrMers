// Device test: the legacy Program builds for the smallest transform sizes (n = 4), where the radix-4 twiddle
// tables hold only 12 words but the build options used to read indices up to 17. The test is built with
// -D_GLIBCXX_ASSERTIONS, which turns an out-of-range std::vector::operator[] into an abort.
// usage: legacy-program-small-device-test [device-index] [p ...]
#define CL_TARGET_OPENCL_VERSION 120
#include "math/Precompute.hpp"
#include "opencl/Context.hpp"
#include "opencl/Program.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef PRMERS_KERNEL_DIR
#define PRMERS_KERNEL_DIR "kernels"
#endif

int main(int argc, char ** argv)
{
	size_t device = 0;
	int first = 1;
	if (argc > 1) { device = std::strtoull(argv[1], nullptr, 10); first = 2; }
	std::vector<uint64_t> ps = {5u, 7u, 11u, 13u, 17u, 19u, 31u, 61u, 89u, 107u, 127u};
	if (argc > first) { ps.clear(); for (int i = first; i < argc; ++i) ps.push_back(std::strtoull(argv[i], nullptr, 10)); }

	int fails = 0;
	for (const uint64_t p : ps)
	{
		math::Precompute pre(p);
		const size_t n = pre.getN();
		if (pre.twiddlesRadix4().size() != pre.invTwiddlesRadix4().size()) { ++fails; std::printf("p=%llu: twiddle table sizes differ\n", (unsigned long long)p); }
		prmers::ocl::Context ctx(device, 0, false, false);
		ctx.computeOptimalSizes(n, pre.getDigitWidth(), p, false, 0, 0);
		prmers::ocl::Program program(ctx, ctx.getDevice(), std::string(PRMERS_KERNEL_DIR) + "/prmers.cl", pre, "", false);
		std::printf("p=%llu n=%zu twiddles=%zu: Program built\n", (unsigned long long)p, n, pre.twiddlesRadix4().size());
	}
	std::printf("Legacy Program small transform test: %s\n", fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
