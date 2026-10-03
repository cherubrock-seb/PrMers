/*
 * Marin register-offset wrap regression test (opt-in, needs a big OpenCL device).
 *
 * Marin kernels index the register slab with 32-bit word offsets.  A register
 * set of R registers of n words needs R * n words; once that reaches 2^32 the
 * offset of the last registers wraps and a kernel silently operates on a low
 * register instead.  Host reads/writes use 64-bit byte offsets, so the damage
 * is invisible until a result is wrong.
 *
 * With q = 127 the transform size is n = 8, so R = 2^29 + 1 registers puts the
 * last register at word offset 2^32.  The test stores 3 in reg 0 and 5 in reg
 * R-1, squares reg R-1 on the device, and checks that reg 0 is still 3 and
 * reg R-1 is 25.
 *
 * Memory: the register slab is 32 GiB of address space, but the test touches
 * only a few registers, so on PoCL (CPU) the resident footprint stays small.
 * The device must accept a 32 GiB single allocation (PoCL on a 64 GiB host
 * does).  Run it with:
 *
 *   OCL_ICD_VENDORS=/etc/OpenCL/vendors/pocl.icd make test-marin-reg-offset-wrap
 *
 * Optional arguments: <device index> <register count>.
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <gmp.h>

#include "marin/engine_gpu.h"

static unsigned long reg_value(const engine & eng, const size_t r)
{
	mpz_t z;
	mpz_init(z);
	eng.get_mpz(z, r);
	const unsigned long v = mpz_fits_ulong_p(z) ? mpz_get_ui(z) : ~0ul;
	mpz_clear(z);
	return v;
}

static void set_value(const engine & eng, const size_t r, const unsigned long v)
{
	mpz_t z;
	mpz_init_set_ui(z, v);
	eng.set_mpz(r, z);
	mpz_clear(z);
}

int main(int argc, char ** argv)
{
	const uint32_t q = 127;
	const size_t device = (argc > 1) ? size_t(std::strtoull(argv[1], nullptr, 10)) : 0;
	const size_t reg_count = (argc > 2) ? size_t(std::strtoull(argv[2], nullptr, 10)) : (size_t(1) << 29) + 1;

	// Skip the full-slab zero fill: only the registers used below are touched,
	// which keeps the resident memory small on CPU OpenCL devices.
	setenv("PRMERS_MARIN_REG_NOCLEAR", "1", 0);

	const size_t n = ibdwt::transform_size(q);
	const size_t last = reg_count - 1;
	std::cout << "q=" << q << " n=" << n << " regs=" << reg_count
	          << " last-reg word offset=" << (last * n) << " (2^32=" << (uint64_t(1) << 32) << ")" << std::endl;

	std::unique_ptr<engine> eng(new engine_gpu(q, reg_count, device, false));

	set_value(*eng, 0, 3);
	set_value(*eng, last, 5);
	eng->square_mul(last);
	eng->sync();

	const unsigned long r0 = reg_value(*eng, 0);
	const unsigned long rl = reg_value(*eng, last);
	std::cout << "reg0=" << r0 << " (expect 3), reg" << last << "=" << rl << " (expect 25)" << std::endl;

	if (r0 != 3 || rl != 25)
	{
		std::cerr << "FAIL: register offsets aliased" << std::endl;
		return 1;
	}
	std::cout << "PASS" << std::endl;
	return 0;
}
