// Host test: legacy sizing helpers (no OpenCL device needed).
//
//  - Context::carryPropagationDepth must terminate for 1-bit digits (exponents below about 5) and keep its
//    previous value for wider digits.
//  - Buffers::maskPackedWords must leave room for the carry kernels' maskPacked[base + 2] prefetch, where
//    base reaches (n - 1) / 64; with n a multiple of 256 the old size was one word short.
//
// Build/run: bash tests/test_legacy_small_items.sh
#include "opencl/Buffers.hpp"
#include "opencl/Context.hpp"

#include <cmath>
#include <cstdio>

static int fails = 0;

static void check(const bool ok, const char * const what, const size_t a, const size_t b)
{
	if (!ok) { ++fails; std::printf("FAIL %s (%zu, %zu)\n", what, a, b); }
}

int main()
{
	using prmers::ocl::Context;

	// Reference doubling loop (terminates for maxdw >= 2 only).
	for (int maxdw = 2; maxdw <= 40; ++maxdw)
	{
		for (size_t n = 4; n <= (size_t(1) << 24); n *= 2)
		{
			size_t ref = 1;
			while (std::pow(maxdw, ref) < std::pow(maxdw, 2) * n) ref *= 2;
			check(Context::carryPropagationDepth(maxdw, n) == ref, "depth matches reference", size_t(maxdw), n);
		}
	}

	// 1-bit (or empty) digits: must return, and cover the whole transform.
	for (size_t n : {size_t(4), size_t(8), size_t(20), size_t(1024)})
	{
		check(Context::carryPropagationDepth(1, n) == n, "1-bit digits depth", 1, n);
		check(Context::carryPropagationDepth(0, n) == n, "0-bit digits depth", 0, n);
	}

	// Packed digit-width mask: the last prefetch index must be inside the buffer.
	for (size_t n = 4; n <= 4096 + 64; ++n)
	{
		const size_t last_base = (n - 1) >> 6;
		check(opencl::Buffers::maskPackedWords(n) >= last_base + 3, "mask guard words", n, opencl::Buffers::maskPackedWords(n));
	}
	for (size_t n : {size_t(256), size_t(512), size_t(1280), size_t(1 << 20), size_t(5) << 20})
	{
		check(opencl::Buffers::maskPackedWords(n) >= ((n - 1) >> 6) + 3, "mask guard words (large)", n, opencl::Buffers::maskPackedWords(n));
	}

	std::printf("legacy small items host test: %s\n", fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
