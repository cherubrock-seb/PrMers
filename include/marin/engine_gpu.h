/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <limits>
#include <set>

#include "engine.h"
#include "ibdwt.h"
#include "ocl.h"

#include "ocl/kernel.h"

#define CREATE_KERNEL_TRANSFORM(name) _##name = create_kernel_transform(#name);
#define CREATE_KERNEL_CARRY(name) _##name = create_kernel_carry(#name);

#define DEFINE_FORWARD(u) void forward##u(const size_t src, const uint32 s, const uint32 lm) { ek_fb(_forward##u, src, s, lm, (u / 4) * _chunk##u); }
#define DEFINE_BACKWARD(u) void backward##u(const size_t src, const uint32 s, const uint32 lm) { ek_fb(_backward##u, src, s, lm, (u / 4) * _chunk##u); }

#define DEFINE_FORWARD_0(u) void forward##u##_0(const size_t src) { ek_fb_0(_forward##u##_0, 8, src, (u / 4) * _chunk##u); }
#define DEFINE_BACKWARD_0(u) void backward##u##_0(const size_t src) { ek_fb_0(_backward##u##_0, 8, src, (u / 4) * _chunk##u); }

#define DEFINE_FORWARD_MUL(u) void forward_mul##u(const size_t src) { ek_fms(_forward_mul##u, 8, src, (u / 4) * _blk##u); }
#define DEFINE_SQR(u) void sqr##u(const size_t src) { ek_fms(_sqr##u, 8, src, (u / 4) * _blk##u); }
#define DEFINE_MUL(u) void mul##u(const size_t dst, const size_t src) { ek_mul(_mul##u, 8, dst, src, (u / 4) * _blk##u); }

class gpu : public ocl::device
{
private:
	const size_t _n, _n5;
	const size_t _reg_count;
	uint32 _q = 0;
	const int _lcwm_wg_size;
	const size_t _blk16, _blk32, _blk64, _blk128, _blk256, _blk512;
	const size_t _chunk16, _chunk20, _chunk64, _chunk80, _chunk256, _chunk320;
	static const size_t _blk4 = 0, _blk8 = 0, _blk1024 = 1, _blk2048 = 1, _chunk4 = 0, _chunk5 = 0, _chunk1024 = 1, _chunk1280 = 1;

	// reg is the weighted representation of registers R0, R1, ...
	cl_mem _reg = nullptr, _carry = nullptr, _root = nullptr, _weight = nullptr, _digit_width = nullptr;

	// v91: optional split auxiliary buffers for very large low-register modes
	// (MM31 on RTX 3080).  This is isolated: normal flat and v83-v90 segmented
	// register modes keep the original single root/weight buffers.
	bool _aux_split = false;
	// v93: compact GPU weight table for MM31 low-register true-delta on 10 GB NVIDIA.
	// Instead of keeping 2*n uint64 weights resident (~2.5 GiB at MM31), keep
	// one base uint64_2 per carry workgroup plus a 4*CWM_WG_SZ relative table.
	// This is isolated to low-register/split-aux programs unless explicitly forced.
	bool _weight_compact = false;
	cl_mem _root1 = nullptr, _root2 = nullptr;       // root[n..2n), root[2n..3n)
	cl_mem _weight1 = nullptr;                      // split second half OR compact relative table
	cl_kernel _mul512_xbuf = nullptr;               // dst/source in different cl_mem buffers

	// v83: GPU-only segmented register slab.  When the logical register slab
	// would exceed CL_DEVICE_MAX_MEM_ALLOC_SIZE, split it into several OpenCL
	// buffers in the same context instead of using host-backed paging.  Kernels
	// still operate on one segment at a time; cross-segment operands are copied
	// device-to-device into reserved per-segment scratch registers.  This is
	// activated only for reg slab > max single allocation, so the original flat
	// path remains unchanged for normal cases.
	bool _reg_segmented = false;
	size_t _seg_usable_regs = 0;
	size_t _seg_scratch_regs = 0;
	size_t _seg_total_regs = 0;
	size_t _seg_count = 0;
	mutable std::vector<cl_mem> _reg_segments;
	mutable std::vector<size_t> _seg_logical_regs;
	mutable std::vector<size_t> _seg_alloc_regs;
	mutable size_t _active_reg_segment = std::numeric_limits<size_t>::max();

	// cl_kernel _forward4 = nullptr, _backward4 = nullptr, _forward16 = nullptr, _backward16 = nullptr;
	cl_kernel _forward64 = nullptr, _backward64 = nullptr, _forward256 = nullptr, _backward256 = nullptr;
	cl_kernel _forward1024 = nullptr, _backward1024 = nullptr;

	cl_kernel _forward4_0 = nullptr, _backward4_0 = nullptr, _forward5_0 = nullptr, _backward5_0 = nullptr;
	cl_kernel _forward16_0 = nullptr, _backward16_0 = nullptr, _forward20_0 = nullptr, _backward20_0 = nullptr;
	cl_kernel _forward64_0 = nullptr, _backward64_0 = nullptr, _forward80_0 = nullptr, _backward80_0 = nullptr;
	cl_kernel _forward256_0 = nullptr, _backward256_0 = nullptr, _forward320_0 = nullptr, _backward320_0 = nullptr;
	cl_kernel _forward1024_0 = nullptr, _backward1024_0 = nullptr;	// _forward1280_0 = nullptr, _backward1280_0 = nullptr;

	cl_kernel _forward_mul4x1 = nullptr, _sqr4x1 = nullptr, _mul4x1 = nullptr;
	cl_kernel _forward_mul4 = nullptr, _sqr4 = nullptr, _mul4 = nullptr;
	cl_kernel _forward_mul8 = nullptr, _sqr8 = nullptr, _mul8 = nullptr;
	cl_kernel _forward_mul16 = nullptr, _sqr16 = nullptr, _mul16 = nullptr;
	cl_kernel _forward_mul32 = nullptr, _sqr32 = nullptr, _mul32 = nullptr;
	cl_kernel _forward_mul64 = nullptr, _sqr64 = nullptr, _mul64 = nullptr;
	cl_kernel _forward_mul128 = nullptr, _sqr128 = nullptr, _mul128 = nullptr;
	cl_kernel _forward_mul256 = nullptr, _sqr256 = nullptr, _mul256 = nullptr;
	cl_kernel _forward_mul512 = nullptr, _sqr512 = nullptr, _mul512 = nullptr;
	cl_kernel _forward_mul1024 = nullptr, _sqr1024 = nullptr, _mul1024 = nullptr;
	// cl_kernel _forward_mul2048 = nullptr, _sqr2048 = nullptr, _mul2048 = nullptr;
	cl_kernel _carry_weight_mul_p1 = nullptr, _carry_weight_add_p1 = nullptr, _carry_weight_add_neg_p1 = nullptr, _carry_weight_p2 = nullptr, _carry_weight_sub_p2 = nullptr, _carry_weight_sub_p2_phase = nullptr, _carry_weight_addsub_p1 = nullptr, _carry_weight_addsub_p2 = nullptr, _carry_weight_p2x2 = nullptr, _carry_weight_mul_p1_copy = nullptr, _carry_weight_p2_copy = nullptr, _carry_weight_addsub_p1_copy = nullptr, _carry_weight_addsub_p2_copy = nullptr, _carry_weight_mul2_unit_p1 = nullptr;
	cl_kernel _copy = nullptr, _subtract = nullptr, _subtract_reg = nullptr;
	cl_kernel _carry_weight_muladd_p1 = nullptr, _carry_weight_muladd_p2 = nullptr;

	std::vector<cl_kernel> _kernels;

public:
	gpu(const ocl::platform & platform, const size_t d, const size_t n, const size_t reg_count, const bool verbose)
		: device(platform, d, verbose), _n(n), _n5((n % 5 == 0) ? n / 5 : n), _reg_count(reg_count),
		_lcwm_wg_size(ilog2(std::min(_n5 / 4, std::min(get_max_local_worksize(sizeof(uint64)), size_t(256))))),

		// We must have (u / 4) * BLKu <= n / 8
		_blk16((_n5 >= 512) ? 16 : 1),		// 16 * BLK16 uint64_2 <= 4KB, workgroup size = (16 / 4) * BLK16 <= 64
		_blk32((_n5 >= 512) ? 8 : 1),		// 32 * BLK32 uint64_2 <= 4KB, workgroup size = (32 / 4) * BLK32 <= 64
		_blk64((_n5 >= 512) ? 4 : 1),		// 64 * BLK64 uint64_2 <= 4KB, workgroup size = (64 / 4) * BLK64 <= 64
		_blk128((_n5 >= 512) ? 2 : 1),		// 128 * BLK128 uint64_2 <= 4KB, workgroup size = (128 / 4) * BLK128 <= 64
		_blk256((_n5 >= 512) ? 1 : 1),		// 256 * BLK256 uint64_2 <= 4KB, workgroup size = (256 / 4) * BLK256 <= 64
		_blk512((_n5 >= 512) ? 1 : 1),		// 512 * BLK512 uint64_2 <= 8KB, workgroup size = (512 / 4) * BLK512 <= 128
		// 1024 uint64_2 = 16KB, workgroup size <= 1024 / 4 = 256, 2048 uint64_2 = 32KB, workgroup size <= 2048 / 4 = 512

		// We must have (u / 4) * CHUNKu <= n / 8 and CHUNKu < m
		_chunk16(std::min(std::max(n / 8 * 4 / 16, size_t(1)), size_t(16))),	// 16 * CHUNK16 uint64_2 <= 4KB, workgroup size = (16 / 4) * CHUNK16 <= 64
		_chunk20(std::min(std::max(n / 8 * 4 / 20, size_t(1)), size_t(16))),	// 20 * CHUNK20 uint64_2 <= 5KB, workgroup size = (20 / 4) * CHUNK20 <= 80
		_chunk64(std::min(std::max(n / 8 * 4 / 64, size_t(1)), size_t(8))),		// 64 * CHUNK64 uint64_2 <= 8KB, workgroup size = (64 / 4) * CHUNK64 <= 128
		_chunk80(std::min(std::max(n / 8 * 4 / 80, size_t(1)), size_t(8))),		// 80 * CHUNK80 uint64_2 <= 10KB, workgroup size = (80 / 4) * CHUNK80 <= 160
		_chunk256(std::min(std::max(n / 8 * 4 / 256, size_t(1)), size_t(4))),	// 256 * CHUNK256 uint64_2 <= 16KB, workgroup size = (256 / 4) * CHUNK256 <= 256
		_chunk320(std::min(std::max(n / 8 * 4 / 320, size_t(1)), size_t(2)))	// 320 * CHUNK320 uint64_2 <= 10KB, workgroup size = (320 / 4) * CHUNK320 <= 160 = 5 * 32
		// 1024: 1024 uint64_2 = 16KB, workgroup size = 1024 / 4 = 256, 1280: 1280 uint64_2 = 20KB, workgroup size = 1280 / 4 = 320
 	{}

	virtual ~gpu()
	{
		// Constructors may throw after some large OpenCL objects have already
		// been created. Release what exists so a PM1 ultralowmem 2-reg
		// allocation failure does not poison the following 1-reg fallback.
		try { release_kernels(); } catch (...) {}
		try { free_memory(); } catch (...) {}
	}

	int get_lcwm_wg_size() const { return _lcwm_wg_size; }
	size_t get_blk16() const { return _blk16; }
	size_t get_blk32() const { return _blk32; }
	size_t get_blk64() const { return _blk64; }
	size_t get_blk128() const { return _blk128; }
	size_t get_blk256() const { return _blk256; }
	size_t get_blk512() const { return _blk512; }
	size_t get_chunk16() const { return _chunk16; }
	size_t get_chunk20() const { return _chunk20; }
	size_t get_chunk64() const { return _chunk64; }
	size_t get_chunk80() const { return _chunk80; }
	size_t get_chunk256() const { return _chunk256; }
	size_t get_chunk320() const { return _chunk320; }

	bool should_split_aux() const
	{
		const cl_ulong max_alloc = get_max_mem_alloc_size();
		if (max_alloc == 0) return false;
		const size_t n = _n;
		const size_t root_bytes = 3 * n * sizeof(uint64);
		const size_t weight_bytes = 2 * n * sizeof(uint64);
		return (static_cast<long double>(root_bytes) > static_cast<long double>(max_alloc)) ||
		       (static_cast<long double>(weight_bytes) > static_cast<long double>(max_alloc));
	}

	void set_aux_split(const bool v) { _aux_split = v; }
	void set_weight_compact(const bool v) { _weight_compact = v; }
	void set_weight_exponent_q(const uint32 q) { _q = q; }
	bool uses_compact_weight() const { return _weight_compact; }

private:
	struct segloc { size_t seg; size_t local; };

	static size_t env_size_local(const char* name, const size_t def)
	{
		const char* v = std::getenv(name);
		if (v == nullptr || *v == '\0') return def;
		const unsigned long long x = std::strtoull(v, nullptr, 10);
		return x == 0 ? def : size_t(x);
	}

	uint32 xform_arg_base() const { return _aux_split ? 4u : 2u; }
	uint32 carry_arg_base() const { return (_aux_split || _weight_compact) ? 5u : 4u; }
	uint32 subtract_arg_base() const { return _aux_split ? 4u : 3u; }

	segloc seg_loc(const size_t logical) const
	{
		if (!_reg_segmented) return {0, logical};
		return { logical / _seg_usable_regs, logical % _seg_usable_regs };
	}

	size_t seg_scratch_local(const size_t seg, const size_t scratch_index) const
	{
		// v84: the last segment may contain fewer logical registers than
		// _seg_usable_regs.  Scratch space starts immediately after the
		// logical registers actually allocated in that segment.  v83 used
		// _seg_usable_regs + scratch_index for every segment, which made
		// copies into the last segment address beyond the cl_mem allocation
		// and produced CL_INVALID_VALUE during large baby-window precompute.
		if (!_reg_segmented) return _seg_usable_regs + scratch_index;
		return _seg_logical_regs.at(seg) + scratch_index;
	}

	bool same_segment(const size_t a, const size_t b) const
	{
		return (!_reg_segmented) || (seg_loc(a).seg == seg_loc(b).seg);
	}

	bool same_segment4(const size_t a, const size_t b, const size_t c, const size_t d) const
	{
		if (!_reg_segmented) return true;
		const size_t s = seg_loc(a).seg;
		return seg_loc(b).seg == s && seg_loc(c).seg == s && seg_loc(d).seg == s;
	}

	void check_segment_local_range(const size_t seg, const size_t local, const char* what) const
	{
		if (!_reg_segmented) return;
		if (seg >= _seg_alloc_regs.size() || local >= _seg_alloc_regs.at(seg))
		{
			std::ostringstream ss;
			ss << "segmented-regspace: " << what << " local=" << local
			   << " outside segment " << seg << " alloc_regs="
			   << (seg < _seg_alloc_regs.size() ? _seg_alloc_regs.at(seg) : 0);
			throw std::runtime_error(ss.str());
		}
	}

	void bind_reg_segment(cl_kernel kernel, const size_t seg) const
	{
		if (!_reg_segmented) return;
		cl_mem mem = _reg_segments.at(seg);
		_set_kernel_arg(kernel, 0, sizeof(cl_mem), &mem);
	}

	void bind_all_reg_kernels_to_segment(const size_t seg) const
	{
		if (!_reg_segmented) return;
		cl_mem mem = _reg_segments.at(seg);
		for (cl_kernel k : _kernels) _set_kernel_arg(k, 0, sizeof(cl_mem), &mem);
		_active_reg_segment = seg;
	}

	void copy_reg_device_to_device(const size_t dst, const size_t src)
	{
		if (!_reg_segmented) { copy(dst, src); return; }
		const segloc d = seg_loc(dst), s = seg_loc(src);
		check_segment_local_range(d.seg, d.local, "copy dst");
		check_segment_local_range(s.seg, s.local, "copy src");
		_copy_buffer(_reg_segments[d.seg], _reg_segments[s.seg], _n * sizeof(uint64), d.local * _n * sizeof(uint64), s.local * _n * sizeof(uint64));
	}

	size_t materialize_src_in_segment(const size_t src, const size_t dstSeg, const size_t scratchIndex)
	{
		if (!_reg_segmented) return src;
		const segloc s = seg_loc(src);
		if (s.seg == dstSeg) return s.local;
		if (scratchIndex >= _seg_scratch_regs) throw std::runtime_error("segmented-regspace: not enough segment scratch registers");
		const size_t scratchLocal = seg_scratch_local(dstSeg, scratchIndex);
		check_segment_local_range(dstSeg, scratchLocal, "scratch dst");
		check_segment_local_range(s.seg, s.local, "scratch src");
		_copy_buffer(_reg_segments[dstSeg], _reg_segments[s.seg], _n * sizeof(uint64), scratchLocal * _n * sizeof(uint64), s.local * _n * sizeof(uint64));
		return scratchLocal;
	}

public:

///////////////////////////////

	void alloc_memory()
	{
#if defined(ocl_debug)
		std::cout << "Alloc gpu memory." << std::endl;
#endif
		const size_t n = _n;
		if (n != 0)
		{
			const size_t reg_bytes = _reg_count * n * sizeof(uint64);
			const size_t carry_bytes = n / 4 * sizeof(uint64);
			const size_t root_bytes = 3 * n * sizeof(uint64);
			const size_t full_weight_bytes = 2 * n * sizeof(uint64);
			const size_t width_bytes = n * sizeof(uint8);
			const bool lowmem_host_staging = (_reg_count <= 3);
			const bool delta3reg_no_giant_clear = (_reg_count == 3);
			const bool force_reg_no_clear = (std::getenv("PRMERS_MARIN_REG_NOCLEAR") != nullptr) || (std::getenv("PRMERS_GPU_ALLOC_DIAG") != nullptr);
			auto gib = [](const size_t b) { return double(b) / 1073741824.0; };
			const cl_ulong device_mem_bytes = get_global_mem_size();
			const cl_ulong max_alloc_bytes = get_max_mem_alloc_size();

			const bool disableSegmented = (std::getenv("PRMERS_MARIN_SEGMENTED_DISABLE") != nullptr);
			const bool forceAuxSplit = (std::getenv("PRMERS_MARIN_SPLIT_AUX_FORCE") != nullptr);
			const bool disableAuxSplit = (std::getenv("PRMERS_MARIN_SPLIT_AUX_DISABLE") != nullptr);
			_aux_split = (!disableAuxSplit && max_alloc_bytes != 0 &&
				(static_cast<long double>(root_bytes) > static_cast<long double>(max_alloc_bytes) ||
				 static_cast<long double>(full_weight_bytes) > static_cast<long double>(max_alloc_bytes))) || forceAuxSplit;

			const bool disableCompactWeight = (std::getenv("PRMERS_MARIN_COMPACT_WEIGHT_DISABLE") != nullptr);
			const bool forceCompactWeight = (std::getenv("PRMERS_MARIN_COMPACT_WEIGHT_FORCE") != nullptr);
			// v93: only low-register/split-aux programs need this by default.
			// Normal PRP/LL/ECM and larger-register PM1 engines keep the old full GPU weight table.
			_weight_compact = !disableCompactWeight && (forceCompactWeight || (lowmem_host_staging && _aux_split));

			const size_t cwm = size_t(1) << _lcwm_wg_size;
			const size_t compact_digits_per_group = 4 * cwm;
			const size_t compact_base_pairs = (n + compact_digits_per_group - 1) / compact_digits_per_group;
			const size_t compact_rel_pairs = compact_digits_per_group;
			const size_t compact_exp_bytes = (compact_base_pairs + compact_rel_pairs) * sizeof(uint32);
			const size_t compact_weight_bytes = (compact_base_pairs + compact_rel_pairs) * 2 * sizeof(uint64) + compact_exp_bytes;
			const size_t weight_bytes = _weight_compact ? compact_weight_bytes : full_weight_bytes;
			const size_t total_bytes = reg_bytes + carry_bytes + root_bytes + weight_bytes + width_bytes;

			std::cout << "[GPU memory plan] regs=" << _reg_count
			          << " reg=" << gib(reg_bytes) << " GiB"
			          << " root=" << gib(root_bytes) << " GiB"
			          << " weight=" << gib(weight_bytes) << " GiB" << (_weight_compact ? " compact" : "")
			          << " carry=" << gib(carry_bytes) << " GiB"
			          << " width=" << gib(width_bytes) << " GiB"
			          << " total=" << gib(total_bytes) << " GiB before driver overhead";
			if (device_mem_bytes != 0) std::cout << " | device=" << gib(static_cast<size_t>(device_mem_bytes)) << " GiB";
			if (max_alloc_bytes != 0) std::cout << " | max-alloc=" << gib(static_cast<size_t>(max_alloc_bytes)) << " GiB";
			std::cout << std::endl;

			if (_aux_split)
			{
				std::cout << "[MARIN-SPLIT-AUX] root/weight exceed max single allocation or split forced; "
				          << (_weight_compact ? "using split root[3] + compact weight-base/relative kernel ABI. " : "using split root[3] + split weight[2] kernel ABI. ")
				          << "Disable with PRMERS_MARIN_SPLIT_AUX_DISABLE=1." << std::endl;
			}
			if (_weight_compact)
			{
				std::cout << "[MARIN-COMPACT-WEIGHT] enabled: base-pairs=" << compact_base_pairs
				          << ", relative-pairs=" << compact_rel_pairs
				          << ", GPU weight table=" << gib(compact_weight_bytes) << " GiB (v94 carry-corrected compact weights). "
				          << "Disable with PRMERS_MARIN_COMPACT_WEIGHT_DISABLE=1." << std::endl;
			}
			// v90: preserve the original flat Marin allocation path whenever the
			// complete register slab fits under the hard OpenCL per-object limit.
			// v83-v88 used a 0.90*maxAlloc trigger, which was good for large
			// many-register RTX 3080 Stage-2 plans but regressed MM31 ultralowmem
			// on P100: 3 regs = 3.75 GiB fits below the ~4 GiB hard maxAlloc,
			// but segmentation with one scratch reg left only 2 usable regs.
			// The safety fraction still applies below when segmenting is actually
			// required; it must not forbid a flat slab that the driver allows.
			const bool needSegmented = (!disableSegmented && max_alloc_bytes != 0 &&
				static_cast<long double>(reg_bytes) > static_cast<long double>(max_alloc_bytes));

			if (needSegmented)
			{
				long double frac = 0.94L;
				if (const char* envFrac = std::getenv("PRMERS_MARIN_SEGMENTED_MAXALLOC_FRAC"))
				{
					const long double f = std::strtold(envFrac, nullptr);
					if (f > 0.10L && f < 0.985L) frac = f;
				}
				// v88: one scratch register per segment is enough for current segmented
				// dispatch. v91 adds a special no-scratch one-register-per-buffer mode
				// for huge low-register MM31 where even one reg+scratch exceeds maxAlloc.
				const size_t per_reg_bytes = n * sizeof(uint64);
				const size_t slots_under_limit = static_cast<size_t>((static_cast<long double>(max_alloc_bytes) * frac) / static_cast<long double>(per_reg_bytes));
				const bool one_reg_split_possible = (lowmem_host_staging && _aux_split && per_reg_bytes <= static_cast<size_t>(max_alloc_bytes));
				if (one_reg_split_possible && slots_under_limit >= 1)
				{
					_seg_scratch_regs = 0;
					_seg_total_regs = 1;
					_seg_usable_regs = 1;
				}
				else
				{
					_seg_scratch_regs = env_size_local("PRMERS_MARIN_SEGMENTED_SCRATCH_REGS", 1);
					if (slots_under_limit <= _seg_scratch_regs + 1)
					{
						std::ostringstream ss;
						ss << "segmented-regspace: only " << slots_under_limit << " regs fit below max single allocation; scratch=" << _seg_scratch_regs;
						throw std::runtime_error(ss.str());
					}
					_seg_total_regs = slots_under_limit;
					_seg_usable_regs = _seg_total_regs - _seg_scratch_regs;
				}
				_reg_segmented = true;
				_seg_count = (_reg_count + _seg_usable_regs - 1) / _seg_usable_regs;
				std::cout << "[MARIN-SEGMENTED] logical regs=" << _reg_count
				          << " reg slab=" << gib(reg_bytes) << " GiB exceeds OpenCL max single allocation="
					          << gib(static_cast<size_t>(max_alloc_bytes)) << " GiB.\n"
				          << "[MARIN-SEGMENTED] using " << _seg_count << " GPU-only cl_mem segment(s), "
				          << "usable-regs/segment=" << _seg_usable_regs << ", scratch-regs/segment=" << _seg_scratch_regs
					          << ", no host backing in hot loop. Disable with PRMERS_MARIN_SEGMENTED_DISABLE=1.\n";
				_reg_segments.resize(_seg_count, nullptr);
				_seg_logical_regs.assign(_seg_count, 0);
				_seg_alloc_regs.assign(_seg_count, 0);
				for (size_t seg = 0; seg < _seg_count; ++seg)
				{
					const size_t first = seg * _seg_usable_regs;
					const size_t logical_here = std::min(_seg_usable_regs, _reg_count - first);
					const size_t alloc_regs = logical_here + _seg_scratch_regs;
					_seg_logical_regs[seg] = logical_here;
					_seg_alloc_regs[seg] = alloc_regs;
					const size_t bytes = alloc_regs * per_reg_bytes;
					std::cout << "[MARIN-SEGMENTED] allocating segment " << (seg + 1) << "/" << _seg_count
					          << ": logical=" << logical_here << " + scratch=" << _seg_scratch_regs
					          << " reg=" << gib(bytes) << " GiB..." << std::endl;
					_reg_segments[seg] = _create_buffer(CL_MEM_READ_WRITE, bytes, false);
				}
				_reg = _reg_segments[0];
			}
			else if (lowmem_host_staging || delta3reg_no_giant_clear || force_reg_no_clear)
			{
				// Low-memory PM1 paths explicitly initialize every register they use.
				std::cout << "[GPU memory alloc] allocating reg (no giant clear)..." << std::endl;
				_reg = _create_buffer(CL_MEM_READ_WRITE, reg_bytes, false);
				std::cout << "[GPU memory alloc] reg OK" << std::endl;
			}
			else
			{
				// Original/default Marin behaviour for normal PRP/LL/PM1/ECM modes.
				_reg = _create_buffer(CL_MEM_READ_WRITE, reg_bytes);
			}

			std::cout << "[GPU memory alloc] allocating carry..." << std::endl;
			_carry = _create_buffer(CL_MEM_READ_WRITE, carry_bytes);
			std::cout << "[GPU memory alloc] carry OK" << std::endl;
			if (_aux_split)
			{
				std::cout << "[GPU memory alloc] allocating split root[0..2]..." << std::endl;
				_root  = _create_buffer(CL_MEM_READ_ONLY, n * sizeof(uint64), false);
				_root1 = _create_buffer(CL_MEM_READ_ONLY, n * sizeof(uint64), false);
				_root2 = _create_buffer(CL_MEM_READ_ONLY, n * sizeof(uint64), false);
				std::cout << "[GPU memory alloc] split root OK" << std::endl;
				if (_weight_compact)
				{
					const size_t cwm = size_t(1) << _lcwm_wg_size;
					const size_t rel_pairs = 4 * cwm;
					const size_t base_pairs = (n + rel_pairs - 1) / rel_pairs;
					std::cout << "[GPU memory alloc] allocating compact weight base+relative..." << std::endl;
					_weight  = _create_buffer(CL_MEM_READ_ONLY, base_pairs * 2 * sizeof(uint64) + base_pairs * sizeof(uint32), false);
					_weight1 = _create_buffer(CL_MEM_READ_ONLY, rel_pairs  * 2 * sizeof(uint64) + rel_pairs  * sizeof(uint32), false);
					std::cout << "[GPU memory alloc] compact weight OK" << std::endl;
				}
				else
				{
					std::cout << "[GPU memory alloc] allocating split weight[0..1]..." << std::endl;
					_weight  = _create_buffer(CL_MEM_READ_ONLY, n * sizeof(uint64), false);
					_weight1 = _create_buffer(CL_MEM_READ_ONLY, n * sizeof(uint64), false);
					std::cout << "[GPU memory alloc] split weight OK" << std::endl;
				}
			}
			else
			{
				std::cout << "[GPU memory alloc] allocating root..." << std::endl;
				_root = _create_buffer(CL_MEM_READ_ONLY, root_bytes, false);
				std::cout << "[GPU memory alloc] root OK" << std::endl;
				if (_weight_compact)
				{
					const size_t cwm = size_t(1) << _lcwm_wg_size;
					const size_t rel_pairs = 4 * cwm;
					const size_t base_pairs = (n + rel_pairs - 1) / rel_pairs;
					std::cout << "[GPU memory alloc] allocating compact weight base+relative..." << std::endl;
					_weight  = _create_buffer(CL_MEM_READ_ONLY, base_pairs * 2 * sizeof(uint64) + base_pairs * sizeof(uint32), false);
					_weight1 = _create_buffer(CL_MEM_READ_ONLY, rel_pairs  * 2 * sizeof(uint64) + rel_pairs  * sizeof(uint32), false);
					std::cout << "[GPU memory alloc] compact weight OK" << std::endl;
				}
				else
				{
					std::cout << "[GPU memory alloc] allocating weight..." << std::endl;
					_weight = _create_buffer(CL_MEM_READ_ONLY, weight_bytes, false);
					std::cout << "[GPU memory alloc] weight OK" << std::endl;
				}
			}
			std::cout << "[GPU memory alloc] allocating digit_width..." << std::endl;
			_digit_width = _create_buffer(CL_MEM_READ_ONLY, width_bytes, false);
			std::cout << "[GPU memory alloc] digit_width OK" << std::endl;
		}
	}

	void free_memory()
	{
#if defined(ocl_debug)
		std::cout << "Free gpu memory." << std::endl;
#endif
		if (_n != 0)
		{
			if (_reg_segmented)
			{
				for (cl_mem & mem : _reg_segments) _release_buffer(mem);
				_reg_segments.clear();
				_seg_logical_regs.clear();
				_seg_alloc_regs.clear();
				_reg = nullptr;
				_reg_segmented = false;
				_active_reg_segment = std::numeric_limits<size_t>::max();
			}
			else _release_buffer(_reg);
			_release_buffer(_carry);
			_release_buffer(_root); _release_buffer(_root1); _release_buffer(_root2); _release_buffer(_weight); _release_buffer(_weight1); _release_buffer(_digit_width);
		}
	}

///////////////////////////////

	cl_kernel create_kernel_transform(const char * const kernel_name)
	{
		cl_kernel kernel = _create_kernel(kernel_name);
		_set_kernel_arg(kernel, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(kernel, 1, sizeof(cl_mem), &_root);
		if (_aux_split)
		{
			_set_kernel_arg(kernel, 2, sizeof(cl_mem), &_root1);
			_set_kernel_arg(kernel, 3, sizeof(cl_mem), &_root2);
		}
		_kernels.push_back(kernel);
		return kernel;
	}

	cl_kernel create_kernel_transform_xbuf(const char * const kernel_name)
	{
		cl_kernel kernel = _create_kernel(kernel_name);
		_set_kernel_arg(kernel, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(kernel, 1, sizeof(cl_mem), &_root);
		_set_kernel_arg(kernel, 2, sizeof(cl_mem), &_root1);
		_set_kernel_arg(kernel, 3, sizeof(cl_mem), &_root2);
		_kernels.push_back(kernel);
		return kernel;
	}

	cl_kernel create_kernel_carry(const char * const kernel_name)
	{
		cl_kernel kernel = _create_kernel(kernel_name);
		_set_kernel_arg(kernel, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(kernel, 1, sizeof(cl_mem), &_carry);
		_set_kernel_arg(kernel, 2, sizeof(cl_mem), &_weight);
		if (_aux_split || _weight_compact)
		{
			_set_kernel_arg(kernel, 3, sizeof(cl_mem), &_weight1);
			_set_kernel_arg(kernel, 4, sizeof(cl_mem), &_digit_width);
		}
		else
		{
			_set_kernel_arg(kernel, 3, sizeof(cl_mem), &_digit_width);
		}
		_kernels.push_back(kernel);
		return kernel;
	}

	void create_kernels()
	{
#if defined(ocl_debug)
		std::cout << "Create ocl kernels." << std::endl;
#endif
		const size_t n = _n;

		// CREATE_KERNEL_TRANSFORM(forward4);
		// CREATE_KERNEL_TRANSFORM(backward4);
		if ((n % 5 != 0) && (n <= 32))
		{
			CREATE_KERNEL_TRANSFORM(forward4_0);
			CREATE_KERNEL_TRANSFORM(backward4_0);
		}
		if (n == 40)
		{
			CREATE_KERNEL_TRANSFORM(forward5_0);
			CREATE_KERNEL_TRANSFORM(backward5_0);
		}

		// CREATE_KERNEL_TRANSFORM(forward16);
		// CREATE_KERNEL_TRANSFORM(backward16);
		if ((n % 5 != 0) && (n >= 64) && (n <= 2048))
		{
			CREATE_KERNEL_TRANSFORM(forward16_0);
			CREATE_KERNEL_TRANSFORM(backward16_0);
		}
		if ((n % 5 == 0) && (n >= 80) && (n <= 2560))
		{
			CREATE_KERNEL_TRANSFORM(forward20_0);
			CREATE_KERNEL_TRANSFORM(backward20_0);
		}

		if (n >= 655360)
		{
			CREATE_KERNEL_TRANSFORM(forward64);
			CREATE_KERNEL_TRANSFORM(backward64);
		}
		if ((n % 5 != 0) && (n >= 4096))
		{
			CREATE_KERNEL_TRANSFORM(forward64_0);
			CREATE_KERNEL_TRANSFORM(backward64_0);
		}
		if ((n % 5 == 0) && (n >= 5120))
		{
			CREATE_KERNEL_TRANSFORM(forward80_0);
			CREATE_KERNEL_TRANSFORM(backward80_0);
		}

		if (n >= 2621440)
		{
			CREATE_KERNEL_TRANSFORM(forward256);
			CREATE_KERNEL_TRANSFORM(backward256);
		}
		if ((n % 5 != 0) && (n >= 131072))
		{
			CREATE_KERNEL_TRANSFORM(forward256_0);
			CREATE_KERNEL_TRANSFORM(backward256_0);
		}
		if ((n % 5 == 0) && (n >= 81920))
		{
			CREATE_KERNEL_TRANSFORM(forward320_0);
			CREATE_KERNEL_TRANSFORM(backward320_0);
		}

		if (get_max_workgroup_size() >= 1024 / 4)
		{
			CREATE_KERNEL_TRANSFORM(forward1024);
			CREATE_KERNEL_TRANSFORM(backward1024);
		}
		if ((n % 5 != 0) && (n >= 524288) && (n <= 1048576))
		{
			CREATE_KERNEL_TRANSFORM(forward1024_0);
			CREATE_KERNEL_TRANSFORM(backward1024_0);
		}
		// if ((n % 5 == 0) && (get_max_workgroup_size() >= 1280 / 4))
		// {
		// 	CREATE_KERNEL_TRANSFORM(forward1280_0);
		// 	CREATE_KERNEL_TRANSFORM(backward1280_0);
		// }

		if (n == 4)
		{
			CREATE_KERNEL_TRANSFORM(forward_mul4x1);
			CREATE_KERNEL_TRANSFORM(sqr4x1);
			CREATE_KERNEL_TRANSFORM(mul4x1);
		}

		if ((n >= 16) && (n <= 80))
		{
			CREATE_KERNEL_TRANSFORM(forward_mul4);
			CREATE_KERNEL_TRANSFORM(sqr4);
			CREATE_KERNEL_TRANSFORM(mul4);
		}

		if ((n >= 8) && (n <= 160))
		{
			CREATE_KERNEL_TRANSFORM(forward_mul8);
			CREATE_KERNEL_TRANSFORM(sqr8);
			CREATE_KERNEL_TRANSFORM(mul8);
		}

		if ((n >= 256) && (n <= 320))
		{
			CREATE_KERNEL_TRANSFORM(forward_mul16);
			CREATE_KERNEL_TRANSFORM(sqr16);
			CREATE_KERNEL_TRANSFORM(mul16);
		}

		if ((n >= 512) && (n <= 640))
		{
			CREATE_KERNEL_TRANSFORM(forward_mul32);
			CREATE_KERNEL_TRANSFORM(sqr32);
			CREATE_KERNEL_TRANSFORM(mul32);
		}

		if ((n >= 1024) && (n <= 5120))
		{
			CREATE_KERNEL_TRANSFORM(forward_mul64);
			CREATE_KERNEL_TRANSFORM(sqr64);
			CREATE_KERNEL_TRANSFORM(mul64);
		}

		if (n >= 2048)
		{
			CREATE_KERNEL_TRANSFORM(forward_mul128);
			CREATE_KERNEL_TRANSFORM(sqr128);
			CREATE_KERNEL_TRANSFORM(mul128);
		}

		if (n >= 16384)
		{
			CREATE_KERNEL_TRANSFORM(forward_mul256);
			CREATE_KERNEL_TRANSFORM(sqr256);
			CREATE_KERNEL_TRANSFORM(mul256);
		}

		if (n >= 32768)
		{
			CREATE_KERNEL_TRANSFORM(forward_mul512);
			CREATE_KERNEL_TRANSFORM(sqr512);
			CREATE_KERNEL_TRANSFORM(mul512);
		}

		if (n >= 65536)
		{
			CREATE_KERNEL_TRANSFORM(forward_mul1024);
			CREATE_KERNEL_TRANSFORM(sqr1024);
			CREATE_KERNEL_TRANSFORM(mul1024);
		}

		// if (get_max_workgroup_size() >= 2048 / 4)
		// {
		// 	CREATE_KERNEL_TRANSFORM(forward_mul2048);
		// 	CREATE_KERNEL_TRANSFORM(sqr2048);
		// 	CREATE_KERNEL_TRANSFORM(mul2048);
		// }
		CREATE_KERNEL_CARRY(carry_weight_mul_p1);
		CREATE_KERNEL_CARRY(carry_weight_muladd_p1);
		CREATE_KERNEL_CARRY(carry_weight_muladd_p2);
		CREATE_KERNEL_CARRY(carry_weight_add_p1);
		CREATE_KERNEL_CARRY(carry_weight_add_neg_p1);
		CREATE_KERNEL_CARRY(carry_weight_p2);
		CREATE_KERNEL_CARRY(carry_weight_sub_p2);
			CREATE_KERNEL_CARRY(carry_weight_sub_p2_phase);
		CREATE_KERNEL_CARRY(carry_weight_addsub_p1);
		CREATE_KERNEL_CARRY(carry_weight_addsub_p2);
		CREATE_KERNEL_CARRY(carry_weight_p2x2);
		CREATE_KERNEL_CARRY(carry_weight_mul_p1_copy);
		CREATE_KERNEL_CARRY(carry_weight_p2_copy);
		CREATE_KERNEL_CARRY(carry_weight_addsub_p1_copy);
		CREATE_KERNEL_CARRY(carry_weight_addsub_p2_copy);
		CREATE_KERNEL_CARRY(carry_weight_mul2_unit_p1);

		if (_aux_split) _mul512_xbuf = create_kernel_transform_xbuf("mul512_xbuf");

		_copy = _create_kernel("copy");
		_set_kernel_arg(_copy, 0, sizeof(cl_mem), &_reg);
		_kernels.push_back(_copy);
		_subtract = _create_kernel("subtract");
		_set_kernel_arg(_subtract, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(_subtract, 1, sizeof(cl_mem), &_weight);
		if (_aux_split)
		{
			_set_kernel_arg(_subtract, 2, sizeof(cl_mem), &_weight1);
			_set_kernel_arg(_subtract, 3, sizeof(cl_mem), &_digit_width);
		}
		else _set_kernel_arg(_subtract, 2, sizeof(cl_mem), &_digit_width);
		_kernels.push_back(_subtract);

		_subtract_reg = _create_kernel("subtract_reg");
		_set_kernel_arg(_subtract_reg, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(_subtract_reg, 1, sizeof(cl_mem), &_weight);
		if (_aux_split)
		{
			_set_kernel_arg(_subtract_reg, 2, sizeof(cl_mem), &_weight1);
			_set_kernel_arg(_subtract_reg, 3, sizeof(cl_mem), &_digit_width);
		}
		else _set_kernel_arg(_subtract_reg, 2, sizeof(cl_mem), &_digit_width);
		_kernels.push_back(_subtract_reg);


	}

	void release_kernels()
	{
#if defined(ocl_debug)
		std::cout << "Release ocl kernels." << std::endl;
#endif
		for (cl_kernel & kernel : _kernels) _release_kernel(kernel);
		_kernels.clear();
	}

///////////////////////////////

	void read_regs(uint64 * const ptr)
	{
		if (!_reg_segmented) { _read_buffer(_reg, ptr, _reg_count * _n * sizeof(uint64)); return; }
		for (size_t r = 0; r < _reg_count; ++r) read_reg(ptr + r * _n, r);
	}
	void write_regs(const uint64 * const ptr)
	{
		if (!_reg_segmented) { _write_buffer(_reg, ptr, _reg_count * _n * sizeof(uint64)); return; }
		for (size_t r = 0; r < _reg_count; ++r) write_reg(ptr + r * _n, r);
	}
	void read_reg(uint64 * const ptr, const size_t index)
	{
		if (!_reg_segmented) { _read_buffer(_reg, ptr, _n * sizeof(uint64), index * _n * sizeof(uint64)); return; }
		const segloc l = seg_loc(index);
		_read_buffer(_reg_segments[l.seg], ptr, _n * sizeof(uint64), l.local * _n * sizeof(uint64));
	}
	void write_reg(const uint64 * const ptr, const size_t index)
	{
		if (!_reg_segmented) { _write_buffer(_reg, ptr, _n * sizeof(uint64), index * _n * sizeof(uint64)); return; }
		const segloc l = seg_loc(index);
		_write_buffer(_reg_segments[l.seg], ptr, _n * sizeof(uint64), l.local * _n * sizeof(uint64));
	}
	void write_reg_part(const uint64 * const ptr, const size_t elems, const size_t index, const size_t offset_elems)
	{
		if (!_reg_segmented) { _write_buffer(_reg, ptr, elems * sizeof(uint64), (index * _n + offset_elems) * sizeof(uint64)); return; }
		const segloc l = seg_loc(index);
		_write_buffer(_reg_segments[l.seg], ptr, elems * sizeof(uint64), (l.local * _n + offset_elems) * sizeof(uint64));
	}
	void fill_reg_zero(const size_t index)
	{
		const uint8_t zero = 0;
		if (!_reg_segmented)
		{
			_fill_buffer(_reg, zero, _n * sizeof(uint64), index * _n * sizeof(uint64));
			return;
		}

		// v91c: NVIDIA may accept clEnqueueFillBuffer for a ~1.25 GiB
		// segmented MM31 register and only report CL_MEM_OBJECT_ALLOCATION_FAILURE
		// at the next blocking command/finish.  Avoid that hidden temporary path
		// completely for segmented registers and clear by small host chunks, the
		// same safe upload mechanism already used by lowmem set_mpz().  This is
		// cold-path initialization only; normal kernels/hot loop are unchanged.
		const segloc l = seg_loc(index);
		check_segment_local_range(l.seg, l.local, "fill zero");
		static constexpr size_t CHUNK_ELEMS = size_t(1) << 20;
		std::vector<uint64> zeros(CHUNK_ELEMS, 0);
		for (size_t off = 0; off < _n; off += CHUNK_ELEMS)
		{
			const size_t len = std::min(CHUNK_ELEMS, _n - off);
			_write_buffer(_reg_segments[l.seg], zeros.data(), len * sizeof(uint64), (l.local * _n + off) * sizeof(uint64));
		}
	}

	void write_root(const uint64 * const ptr)
	{
		if (!_aux_split) { _write_buffer(_root, ptr, 3 * _n * sizeof(uint64)); return; }
		_write_buffer(_root,  ptr,           _n * sizeof(uint64));
		_write_buffer(_root1, ptr + _n,      _n * sizeof(uint64));
		_write_buffer(_root2, ptr + 2 * _n,  _n * sizeof(uint64));
	}
	void write_root_part(const uint64 * const ptr, const size_t elems, const size_t offset_elems)
	{
		if (!_aux_split) { _write_buffer(_root, ptr, elems * sizeof(uint64), offset_elems * sizeof(uint64)); return; }
		size_t done = 0;
		while (done < elems)
		{
			const size_t global = offset_elems + done;
			const size_t part = global / _n;
			const size_t local = global % _n;
			const size_t len = std::min(elems - done, _n - local);
			cl_mem dst = (part == 0) ? _root : ((part == 1) ? _root1 : _root2);
			_write_buffer(dst, ptr + done, len * sizeof(uint64), local * sizeof(uint64));
			done += len;
		}
	}
	void write_weight(const uint64 * const ptr)
	{
		if (_weight_compact)
		{
			const size_t cwm = size_t(1) << _lcwm_wg_size;
			const size_t rel_pairs = 4 * cwm;
			const size_t base_pairs = (_n + rel_pairs - 1) / rel_pairs;
			if (_q == 0) throw std::runtime_error("compact weight: missing exponent q");
			auto storage_pair_index = [this](const size_t digit) -> size_t {
				return digit / 4 + (digit % 4) * (_n / 4);
			};
			auto weight_exp_for_digit = [this](const size_t digit) -> uint32 {
				if (digit == 0) return 0u;
				const uint64 qj = uint64(_q) * uint64(digit);
				const uint32 r = uint32(qj % uint64(_n));
				return (r == 0u) ? 0u : uint32(_n) - r;
			};

			std::vector<uint64> base(2 * base_pairs);
			std::vector<uint64> rel(2 * rel_pairs);
			std::vector<uint32> base_exp(base_pairs);
			std::vector<uint32> rel_exp(rel_pairs);
			for (size_t g = 0; g < base_pairs; ++g)
			{
				const size_t digit = std::min(g * rel_pairs, _n - 1);
				const size_t i = storage_pair_index(digit);
				base[2 * g + 0] = ptr[2 * i + 0];
				base[2 * g + 1] = ptr[2 * i + 1];
				base_exp[g] = weight_exp_for_digit(digit);
			}
			for (size_t d = 0; d < rel_pairs; ++d)
			{
				const size_t digit = std::min(d, _n - 1);
				const size_t i = storage_pair_index(digit);
				rel[2 * d + 0] = ptr[2 * i + 0];
				rel[2 * d + 1] = ptr[2 * i + 1];
				rel_exp[d] = weight_exp_for_digit(digit);
			}
			_write_buffer(_weight,  base.data(), base.size() * sizeof(uint64));
			_write_buffer(_weight,  base_exp.data(), base_exp.size() * sizeof(uint32), base.size() * sizeof(uint64));
			_write_buffer(_weight1, rel.data(),  rel.size()  * sizeof(uint64));
			_write_buffer(_weight1, rel_exp.data(), rel_exp.size() * sizeof(uint32), rel.size() * sizeof(uint64));
			return;
		}
		if (!_aux_split) { _write_buffer(_weight, ptr, 2 * _n * sizeof(uint64)); return; }
		// Split by uint64_2 pair index: first n/2 pairs then second n/2 pairs.
		_write_buffer(_weight,  ptr,      _n * sizeof(uint64));
		_write_buffer(_weight1, ptr + _n, _n * sizeof(uint64));
	}
	void write_width(const uint8 * const ptr) { _write_buffer(_digit_width, ptr, _n * sizeof(uint8)); }

///////////////////////////////

	void ek_fb(cl_kernel & kernel, const size_t src, const uint32 s, const uint32 lm, const size_t local_size = 0)
	{
		const segloc l = seg_loc(src);
		bind_reg_segment(kernel, l.seg);
		const uint32 offset = uint32(l.local * _n);
		_set_kernel_arg(kernel, xform_arg_base(), sizeof(uint32), &offset);
		_set_kernel_arg(kernel, xform_arg_base() + 1, sizeof(uint32), &s);
		_set_kernel_arg(kernel, xform_arg_base() + 2, sizeof(uint32), &lm);
		_execute_kernel(kernel, _n / 8, local_size);
	}

	void ek_fb_0(cl_kernel & kernel, const size_t step, const size_t src, const size_t local_size = 0)
	{
		const segloc l = seg_loc(src);
		bind_reg_segment(kernel, l.seg);
		const uint32 offset = uint32(l.local * _n);
		_set_kernel_arg(kernel, xform_arg_base(), sizeof(uint32), &offset);
		_execute_kernel(kernel, _n / step, local_size);
	}

	void ek_fms(cl_kernel & kernel, const size_t step, const size_t src, const size_t local_size = 0)
	{
		const segloc l = seg_loc(src);
		bind_reg_segment(kernel, l.seg);
		const uint32 offset = uint32(l.local * _n);
		_set_kernel_arg(kernel, xform_arg_base(), sizeof(uint32), &offset);
		_execute_kernel(kernel, _n / step, local_size);
	}

	void ek_mul(cl_kernel & kernel, const size_t step, const size_t dst, const size_t src, const size_t local_size = 0)
	{
		const segloc d = seg_loc(dst);
		const size_t srcLocal = materialize_src_in_segment(src, d.seg, 0);
		bind_reg_segment(kernel, d.seg);
		const uint32 offset_y = uint32(srcLocal * _n);
		_set_kernel_arg(kernel, xform_arg_base() + 1, sizeof(uint32), &offset_y);
		const uint32 offset = uint32(d.local * _n);
		_set_kernel_arg(kernel, xform_arg_base(), sizeof(uint32), &offset);
		_execute_kernel(kernel, _n / step, local_size);
	}

	void ek_mul_xbuf512(const size_t dst, const size_t src, const size_t local_size = 0)
	{
		if (_mul512_xbuf == nullptr) throw std::runtime_error("split-reg cross-buffer mul512 kernel not available");
		const segloc d = seg_loc(dst), s = seg_loc(src);
		bind_reg_segment(_mul512_xbuf, d.seg);
		cl_mem src_mem = _reg_segmented ? _reg_segments[s.seg] : _reg;
		_set_kernel_arg(_mul512_xbuf, 4, sizeof(cl_mem), &src_mem);
		const uint32 offset = uint32(d.local * _n);
		const uint32 offset_y = uint32(s.local * _n);
		_set_kernel_arg(_mul512_xbuf, 5, sizeof(uint32), &offset);
		_set_kernel_arg(_mul512_xbuf, 6, sizeof(uint32), &offset_y);
		_execute_kernel(_mul512_xbuf, _n / 8, local_size);
	}

	// DEFINE_FORWARD(4);
	// DEFINE_BACKWARD(4);
	DEFINE_FORWARD_0(4);
	DEFINE_BACKWARD_0(4);
	DEFINE_FORWARD_0(5);
	DEFINE_BACKWARD_0(5);

	// DEFINE_FORWARD(16);
	// DEFINE_BACKWARD(16);
	DEFINE_FORWARD_0(16);
	DEFINE_BACKWARD_0(16);
	DEFINE_FORWARD_0(20);
	DEFINE_BACKWARD_0(20);

	DEFINE_FORWARD(64);
	DEFINE_BACKWARD(64);
	DEFINE_FORWARD_0(64);
	DEFINE_BACKWARD_0(64);
	DEFINE_FORWARD_0(80);
	DEFINE_BACKWARD_0(80);

	DEFINE_FORWARD(256);
	DEFINE_BACKWARD(256);
	DEFINE_FORWARD_0(256);
	DEFINE_BACKWARD_0(256);
	DEFINE_FORWARD_0(320);
	DEFINE_BACKWARD_0(320);

	DEFINE_FORWARD(1024);
	DEFINE_BACKWARD(1024);
	DEFINE_FORWARD_0(1024);
	DEFINE_BACKWARD_0(1024);
	// DEFINE_FORWARD_0(1280);
	// DEFINE_BACKWARD_0(1280);

	void forward_mul4x1(const size_t src) { ek_fms(_forward_mul4x1, 4, src); }
	void sqr4x1(const size_t src) { ek_fms(_sqr4x1, 4, src); }
	void mul4x1(const size_t dst, const size_t src) { ek_mul(_mul4x1, 4, dst, src); }

	DEFINE_FORWARD_MUL(4);
	DEFINE_SQR(4);
	DEFINE_MUL(4);

	DEFINE_FORWARD_MUL(8);
	DEFINE_SQR(8);
	DEFINE_MUL(8);

	DEFINE_FORWARD_MUL(16);
	DEFINE_SQR(16);
	DEFINE_MUL(16);

	DEFINE_FORWARD_MUL(32);
	DEFINE_SQR(32);
	DEFINE_MUL(32);

	DEFINE_FORWARD_MUL(64);
	DEFINE_SQR(64);
	DEFINE_MUL(64);

	DEFINE_FORWARD_MUL(128);
	DEFINE_SQR(128);
	DEFINE_MUL(128);

	DEFINE_FORWARD_MUL(256);
	DEFINE_SQR(256);
	DEFINE_MUL(256);

	DEFINE_FORWARD_MUL(512);
	DEFINE_SQR(512);
	void mul512(const size_t dst, const size_t src)
	{
		if (_reg_segmented && !same_segment(dst, src) && _seg_scratch_regs == 0) { ek_mul_xbuf512(dst, src, (512 / 4) * _blk512); return; }
		ek_mul(_mul512, 8, dst, src, (512 / 4) * _blk512);
	}

	DEFINE_FORWARD_MUL(1024);
	DEFINE_SQR(1024);
	DEFINE_MUL(1024);

	// DEFINE_FORWARD_MUL(2048);
	// DEFINE_SQR(2048);
	// DEFINE_MUL(2048);

	void carry_weight_mul(const size_t src, const uint32 a)
	{
		const segloc l = seg_loc(src);
		bind_reg_segment(_carry_weight_mul_p1, l.seg);
		bind_reg_segment(_carry_weight_p2, l.seg);
		const uint32 offset = uint32(l.local * _n);
		_set_kernel_arg(_carry_weight_mul_p1, carry_arg_base(), sizeof(uint32), &a);
		_set_kernel_arg(_carry_weight_mul_p1, carry_arg_base() + 1, sizeof(uint32), &offset);
		_execute_kernel(_carry_weight_mul_p1, _n / 4, 1u << _lcwm_wg_size);
		_set_kernel_arg(_carry_weight_p2, carry_arg_base(), sizeof(uint32), &offset);
		_execute_kernel(_carry_weight_p2, (_n / 4) >> _lcwm_wg_size);
	}

	void carry_weight_mul_copy(const size_t src, const size_t dst, const uint32 a)
	{
		if (_reg_segmented && !same_segment(src, dst))
		{
			carry_weight_mul(src, a);
			copy_reg_device_to_device(dst, src);
			return;
		}
		const segloc sl = seg_loc(src), dl = seg_loc(dst);
		bind_reg_segment(_carry_weight_mul_p1_copy, sl.seg);
		bind_reg_segment(_carry_weight_p2_copy, sl.seg);
		uint32 offS = (uint32)(sl.local * _n);
		uint32 offD = (uint32)(dl.local * _n);
		_set_kernel_arg(_carry_weight_mul_p1_copy, carry_arg_base(), sizeof(uint32), &a);
		_set_kernel_arg(_carry_weight_mul_p1_copy, carry_arg_base() + 1, sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_mul_p1_copy, carry_arg_base() + 2, sizeof(uint32), &offD);
		_execute_kernel(_carry_weight_mul_p1_copy, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_p2_copy, carry_arg_base(), sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_p2_copy, carry_arg_base() + 1, sizeof(uint32), &offD);
		_execute_kernel(_carry_weight_p2_copy, (_n / 4) >> _lcwm_wg_size);
	}

	void carry_weight_muladd(const size_t dst, const size_t add_src, const uint32 a)
	{
		const segloc d = seg_loc(dst);
		const size_t addLocal = materialize_src_in_segment(add_src, d.seg, 0);
		bind_reg_segment(_carry_weight_muladd_p1, d.seg);
		bind_reg_segment(_carry_weight_muladd_p2, d.seg);
		const uint32 offY = uint32(d.local * _n);
		const uint32 offX = uint32(addLocal * _n);

		_set_kernel_arg(_carry_weight_muladd_p1, carry_arg_base(), sizeof(uint32), &a);
		_set_kernel_arg(_carry_weight_muladd_p1, carry_arg_base() + 1, sizeof(uint32), &offY);
		_set_kernel_arg(_carry_weight_muladd_p1, carry_arg_base() + 2, sizeof(uint32), &offX);
		_execute_kernel(_carry_weight_muladd_p1, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_muladd_p2, carry_arg_base(), sizeof(uint32), &offY);
		_execute_kernel(_carry_weight_muladd_p2, (_n / 4) >> _lcwm_wg_size);
	}
	

	
	void carry_weight_add(const size_t dst, const size_t src)
	{
		const segloc d = seg_loc(dst);
		const size_t srcLocal = materialize_src_in_segment(src, d.seg, 0);
		bind_reg_segment(_carry_weight_add_p1, d.seg);
		bind_reg_segment(_carry_weight_p2, d.seg);
		const uint32 offset_y = uint32(d.local * _n), offset_x = uint32(srcLocal * _n);
		_set_kernel_arg(_carry_weight_add_p1, carry_arg_base(), sizeof(uint32), &offset_y);
		_set_kernel_arg(_carry_weight_add_p1, carry_arg_base() + 1, sizeof(uint32), &offset_x);
		_execute_kernel(_carry_weight_add_p1, _n / 4, 1u << _lcwm_wg_size);
		_set_kernel_arg(_carry_weight_p2, carry_arg_base(), sizeof(uint32), &offset_y);
		_execute_kernel(_carry_weight_p2, (_n / 4) >> _lcwm_wg_size);

	}
	

	
	void carry_weight_sub(const size_t dst, const size_t src)
	{
		const segloc d = seg_loc(dst);
		const size_t srcLocal = materialize_src_in_segment(src, d.seg, 0);
		bind_reg_segment(_carry_weight_add_neg_p1, d.seg);
		bind_reg_segment(_carry_weight_p2, d.seg);
		const uint32 offset_y = uint32(d.local * _n), offset_x = uint32(srcLocal * _n);
		_set_kernel_arg(_carry_weight_add_neg_p1, carry_arg_base(), sizeof(uint32), &offset_y);
		_set_kernel_arg(_carry_weight_add_neg_p1, carry_arg_base() + 1, sizeof(uint32), &offset_x);
		_execute_kernel(_carry_weight_add_neg_p1, _n / 4, 1u << _lcwm_wg_size);
		_set_kernel_arg(_carry_weight_p2, carry_arg_base(), sizeof(uint32), &offset_y);
		_execute_kernel(_carry_weight_p2, (_n / 4) >> _lcwm_wg_size);

	}



	void carry_weight_sub_safe(const size_t dst, const size_t src)
	{
		carry_weight_sub(dst, src);
	}

	
	void carry_weight_addsub(const size_t sum, const size_t diff, const size_t a, const size_t b)
	{
		if (_reg_segmented && !same_segment4(sum, diff, a, b))
		{
			copy(sum, a);
			carry_weight_add(sum, b);
			copy(diff, a);
			carry_weight_sub(diff, b);
			return;
		}
		const segloc sl = seg_loc(sum), dl = seg_loc(diff), al = seg_loc(a), bl = seg_loc(b);
		bind_reg_segment(_carry_weight_addsub_p1, sl.seg);
		bind_reg_segment(_carry_weight_addsub_p2, sl.seg);
		uint32 offS = (uint32)(sl.local * _n);
		uint32 offD = (uint32)(dl.local * _n);
		uint32 offA = (uint32)(al.local * _n);
		uint32 offB = (uint32)(bl.local * _n);

		_set_kernel_arg(_carry_weight_addsub_p1, carry_arg_base(), sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_addsub_p1, carry_arg_base() + 1, sizeof(uint32), &offD);
		_set_kernel_arg(_carry_weight_addsub_p1, carry_arg_base() + 2, sizeof(uint32), &offA);
		_set_kernel_arg(_carry_weight_addsub_p1, carry_arg_base() + 3, sizeof(uint32), &offB);
		_execute_kernel(_carry_weight_addsub_p1, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_addsub_p2, carry_arg_base(), sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_addsub_p2, carry_arg_base() + 1, sizeof(uint32), &offD);
		_execute_kernel(_carry_weight_addsub_p2, (_n / 4) >> _lcwm_wg_size);
	}


	void carry_weight_mul2_unit(const size_t dst0, const size_t dst1)
	{
		if (_reg_segmented && !same_segment(dst0, dst1))
		{
			carry_weight_mul(dst0, 1);
			carry_weight_mul(dst1, 1);
			return;
		}
		const segloc d0 = seg_loc(dst0), d1 = seg_loc(dst1);
		bind_reg_segment(_carry_weight_mul2_unit_p1, d0.seg);
		bind_reg_segment(_carry_weight_p2x2, d0.seg);
		uint32 off0 = (uint32)(d0.local * _n);
		uint32 off1 = (uint32)(d1.local * _n);

		_set_kernel_arg(_carry_weight_mul2_unit_p1, carry_arg_base(), sizeof(uint32), &off0);
		_set_kernel_arg(_carry_weight_mul2_unit_p1, carry_arg_base() + 1, sizeof(uint32), &off1);
		_execute_kernel(_carry_weight_mul2_unit_p1, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_p2x2, carry_arg_base(), sizeof(uint32), &off0);
		_set_kernel_arg(_carry_weight_p2x2, carry_arg_base() + 1, sizeof(uint32), &off1);
		_execute_kernel(_carry_weight_p2x2, (_n / 4) >> _lcwm_wg_size);
	}


	void addsub_copy(const size_t sum, const size_t diff, const size_t sum_copy, const size_t diff_copy,
					const size_t a, const size_t b)
	{
		if (_reg_segmented && !(same_segment4(sum, diff, a, b) && same_segment(sum, sum_copy) && same_segment(diff, diff_copy)))
		{
			carry_weight_addsub(sum, diff, a, b);
			copy(sum_copy, sum);
			copy(diff_copy, diff);
			return;
		}
		const segloc sl = seg_loc(sum), dl = seg_loc(diff), scl = seg_loc(sum_copy), dcl = seg_loc(diff_copy), al = seg_loc(a), bl = seg_loc(b);
		bind_reg_segment(_carry_weight_addsub_p1_copy, sl.seg);
		bind_reg_segment(_carry_weight_addsub_p2_copy, sl.seg);
		const uint32 offS = (uint32)(sl.local * _n);
		const uint32 offD = (uint32)(dl.local * _n);
		const uint32 offSc = (uint32)(scl.local * _n);
		const uint32 offDc = (uint32)(dcl.local * _n);
		const uint32 offA = (uint32)(al.local * _n);
		const uint32 offB = (uint32)(bl.local * _n);

		_set_kernel_arg(_carry_weight_addsub_p1_copy, carry_arg_base(), sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, carry_arg_base() + 1, sizeof(uint32), &offD);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, carry_arg_base() + 2, sizeof(uint32), &offSc);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, carry_arg_base() + 3, sizeof(uint32), &offDc);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, carry_arg_base() + 4, sizeof(uint32), &offA);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, carry_arg_base() + 5, sizeof(uint32), &offB);
		_execute_kernel(_carry_weight_addsub_p1_copy, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_addsub_p2_copy, carry_arg_base(), sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_addsub_p2_copy, carry_arg_base() + 1, sizeof(uint32), &offD);
		_set_kernel_arg(_carry_weight_addsub_p2_copy, carry_arg_base() + 2, sizeof(uint32), &offSc);
		_set_kernel_arg(_carry_weight_addsub_p2_copy, carry_arg_base() + 3, sizeof(uint32), &offDc);
		_execute_kernel(_carry_weight_addsub_p2_copy, (_n / 4) >> _lcwm_wg_size);
	}


	void copy(const size_t dst, const size_t src)
	{
		if (_reg_segmented && !same_segment(dst, src))
		{
			copy_reg_device_to_device(dst, src);
			return;
		}
		const segloc d = seg_loc(dst), s = seg_loc(src);
		bind_reg_segment(_copy, d.seg);
		const uint32 offset_y = uint32(d.local * _n), offset_x = uint32(s.local * _n);
		_set_kernel_arg(_copy, 1, sizeof(uint32), &offset_y);
		_set_kernel_arg(_copy, 2, sizeof(uint32), &offset_x);
		_execute_kernel(_copy, _n);
	}

	void subtract(const size_t src, const uint32 a)
	{
		const segloc l = seg_loc(src);
		bind_reg_segment(_subtract, l.seg);
		const uint32 offset = uint32(l.local * _n);
		_set_kernel_arg(_subtract, subtract_arg_base(), sizeof(uint32), &offset);
		_set_kernel_arg(_subtract, subtract_arg_base() + 1, sizeof(uint32), &a);
		_execute_kernel(_subtract, 1);
	}

	void subtract_reg_strong(const size_t dst, const size_t src)
	{
		if (_subtract_reg == nullptr || (_reg_segmented && !same_segment(dst, src)))
		{
			carry_weight_sub(dst, src);
			return;
		}
		const segloc d = seg_loc(dst), s = seg_loc(src);
		bind_reg_segment(_subtract_reg, d.seg);
		const uint32 offset_y = uint32(d.local * _n), offset_x = uint32(s.local * _n);
		_set_kernel_arg(_subtract_reg, subtract_arg_base(), sizeof(uint32), &offset_y);
		_set_kernel_arg(_subtract_reg, subtract_arg_base() + 1, sizeof(uint32), &offset_x);
		_execute_kernel(_subtract_reg, 1);
	}
};

class engine_gpu_flat : public engine
{
private:
	const size_t _reg_count;
	const size_t _n;
	gpu * _gpu;
	std::vector<uint64> _weight;
	std::vector<uint8> _digit_width;


public:
	engine_gpu_flat(const uint32_t q, const size_t reg_count, const size_t device, const bool verbose) : engine(),
		_reg_count(reg_count), _n(ibdwt::transform_size(q))
	{
		const size_t n = _n;

		const ocl::platform eng_platform = ocl::platform();
		std::unique_ptr<gpu> gpu_owner(new gpu(eng_platform, device, n, _reg_count, verbose));
		_gpu = gpu_owner.get();

		std::ostringstream src;
		src << "#define N_SZ\t" << n << "u" << std::endl;
		const size_t s5 = (n % 5 == 0) ? 5 : 4;
		src << "#define LN_SZ_S5\t" << ilog2(n / s5) << std::endl;
		src << "#define INV_N_2\t" << MOD_P - (MOD_P - 1) / (n / 2) << "ul" << std::endl;

		const uint64 K = mod_root_nth(5), K2 = mod_sqr(K), K3 = mod_mul(K, K2), K4 = mod_sqr(K2);
		const uint64 cosu = mod_half(mod_add(K, K4)), isinu = mod_half(mod_sub(K, K4));
		const uint64 cos2u = mod_half(mod_add(K2, K3)), isin2u = mod_half(mod_sub(K2, K3));
		const uint64 F1 = mod_sub(mod_half(mod_add(cosu, cos2u)), 1), F2 = mod_half(mod_sub(cosu, cos2u));
		const uint64 F3 = mod_add(isinu, isin2u), F4 = isin2u, F5 = mod_sub(isinu, isin2u);
		src << "#define W_F1\t" << F1 << "ul" << std::endl;
		src << "#define W_F2\t" << F2 << "ul" << std::endl;
		src << "#define W_F3\t" << F3 << "ul" << std::endl;
		src << "#define W_F4\t" << F4 << "ul" << std::endl;
		src << "#define W_F5\t" << F5 << "ul" << std::endl;

		src << "#define BLK16\t" << _gpu->get_blk16() << "u" << std::endl;
		src << "#define BLK32\t" << _gpu->get_blk32() << "u" << std::endl;
		src << "#define BLK64\t" << _gpu->get_blk64() << "u" << std::endl;
		src << "#define BLK128\t" << _gpu->get_blk128() << "u" << std::endl;
		src << "#define BLK256\t" << _gpu->get_blk256() << "u" << std::endl;
		src << "#define BLK512\t" << _gpu->get_blk512() << "u" << std::endl;

		src << "#define CHUNK16\t" << _gpu->get_chunk16() << "u" << std::endl;
		src << "#define CHUNK20\t" << _gpu->get_chunk20() << "u" << std::endl;
		src << "#define CHUNK64\t" << _gpu->get_chunk64() << "u" << std::endl;
		src << "#define CHUNK80\t" << _gpu->get_chunk80() << "u" << std::endl;
		src << "#define CHUNK256\t" << _gpu->get_chunk256() << "u" << std::endl;
		src << "#define CHUNK320\t" << _gpu->get_chunk320() << "u" << std::endl;

		src << "#define CWM_WG_SZ\t" << (1u << _gpu->get_lcwm_wg_size()) << "u" << std::endl;

		src << "#define MAX_WG_SZ\t" << _gpu->get_max_workgroup_size() << "u" << std::endl;

		const bool auxSplitProgram = ((std::getenv("PRMERS_MARIN_SPLIT_AUX_DISABLE") == nullptr) && _gpu->should_split_aux()) || (std::getenv("PRMERS_MARIN_SPLIT_AUX_FORCE") != nullptr);
		const bool compactWeightProgram = (std::getenv("PRMERS_MARIN_COMPACT_WEIGHT_DISABLE") == nullptr) &&
			((std::getenv("PRMERS_MARIN_COMPACT_WEIGHT_FORCE") != nullptr) || (_reg_count <= 3 && auxSplitProgram));
		_gpu->set_aux_split(auxSplitProgram);
		_gpu->set_weight_compact(compactWeightProgram);
		_gpu->set_weight_exponent_q(q);
		if (auxSplitProgram)
		{
			src << "#define MARIN_SPLIT_AUX 1" << std::endl;
			std::cout << "[MARIN-SPLIT-AUX] compiling split root kernel ABI" << std::endl;
		}
		if (compactWeightProgram)
		{
			src << "#define MARIN_COMPACT_WEIGHT 1" << std::endl;
			std::cout << "[MARIN-COMPACT-WEIGHT] compiling compact weight kernel ABI" << std::endl;
		}
		src << std::endl;

		if (!_gpu->read_OpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_gpu->load_program(src.str());
		_gpu->alloc_memory();
		_gpu->create_kernels();

		if (_reg_count <= 3)
		{
			// V41: staged root generation is used by low-memory 1/2-register paths
			// and by the canonical PM1 delta 3-register path.  This avoids a
			// 3.75 GiB host root vector on Kaggle/P100.  Normal Marin modes with
			// higher register counts keep the original full-root build below.
			std::cout << "[host memory] building/uploading roots in staged mode (v41 lowmem/delta3 3n/2 pack)" << std::endl;
			const size_t r2pack_elems = n + n / 2;
			std::vector<uint64> r2pack(r2pack_elems, 0);
			uint64 * const r2 = &r2pack[0];
			uint64 * const r2i = &r2pack[n / 2];
			for (size_t s = (n % 5 == 0) ? 5 : 1; s <= n / 4; s *= 2)
			{
				const uint64 rs = mod_root_nth(2 * s), rsi = mod_invert(rs);
				uint64 rsj = 1, rsji = 1;
				for (size_t j = 0; j < s; ++j)
				{
					const size_t jr = ibdwt::inv_reversal(j, s);
					r2[s + jr] = rsj;
					r2i[s + jr] = rsji;
					rsj = mod_mul(rsj, rs);
					rsji = mod_mul(rsji, rsi);
				}
			}
			_gpu->write_root_part(r2pack.data(), n, 0);

			for (size_t s = (n % 5 == 0) ? 5 : 1; s <= n / 4; s *= 2)
			{
				const size_t elems = 2 * s;
				std::vector<uint64> block(elems, 0);
				for (size_t j = 0; j < s; ++j)
				{
					const size_t sj = s + j;
					block[2 * j + 0] = r2[2 * sj];
					block[2 * j + 1] = mod_mul(r2[sj], r2[2 * sj]);
				}
				_gpu->write_root_part(block.data(), elems, n + 2 * s);

				for (size_t j = 0; j < s; ++j)
				{
					const size_t sj = s + j;
					block[2 * j + 0] = r2i[2 * sj];
					block[2 * j + 1] = mod_mul(r2i[sj], r2i[2 * sj]);
				}
				_gpu->write_root_part(block.data(), elems, 2 * n + 2 * s);
			}
			std::vector<uint64>().swap(r2pack);
			std::cout << "[host memory] roots uploaded; host root staging released" << std::endl;
		}
		else
		{
			// Original/default Marin root generation for non-low-memory modes.
			std::vector<uint64> root(3 * n);
			ibdwt::roots(n, root.data());
			_gpu->write_root(root.data());
		}

		if (_reg_count <= 3) std::cout << "[host memory] allocating/building weight + digit_width host tables" << std::endl;
		_weight.resize(2 * n);
		_digit_width.resize(n);
		ibdwt::weights_widths(n, q, _weight.data(), _digit_width.data());
		if (_reg_count <= 3) std::cout << "[host memory] uploading weight + digit_width" << std::endl;
		_gpu->write_weight(_weight.data());
		_gpu->write_width(_digit_width.data());
		if (_reg_count <= 3) std::cout << "[host memory] engine transform tables ready" << std::endl;

		// Only transfer ownership after every large allocation/upload completed.
		// If construction throws before this point, gpu_owner cleans the OpenCL
		// context/buffers and the caller can safely create a fallback engine.
		gpu_owner.release();
	}

	void release_gpu_resources_for_lowmem_handoff() override
	{
		if (_gpu == nullptr) return;
		// Make every queued kernel/read/write visible to the driver before releasing
		// the huge MM31 buffers. This is intentionally used by the PM1 low-memory
		// Stage1 -> Stage2 handoff, where a second large engine is allocated on the
		// same GPU immediately afterwards.
		_gpu->finish_all_queues();
		_gpu->release_kernels();
		_gpu->free_memory();
		_gpu->finish_all_queues();
		_gpu->clear_program();
		delete _gpu;
		_gpu = nullptr;
		std::vector<uint64>().swap(_weight);
		std::vector<uint8>().swap(_digit_width);
	}

	virtual ~engine_gpu_flat()
	{
		release_gpu_resources_for_lowmem_handoff();
	}

	size_t get_size() const override { return _n; }

	void sync() const override
	{
		if (_gpu != nullptr) _gpu->finish_all_queues();
	}

	void set(const Reg dst, const uint32 a) const override
	{
		const size_t n = _n;
		if (_reg_count <= 3)
		{
			std::cout << "[host memory] lowmem streamed set(reg=" << size_t(dst) << ", const=" << a << ")" << std::endl;
			_gpu->fill_reg_zero(size_t(dst));
			uint64 first = a; // weight[0] = 1
			_gpu->write_reg_part(&first, 1, size_t(dst), 0);
			return;
		}

		std::vector<uint64> x(n);

		x[0] = a;	// weight[0] = 1
		for (size_t k = 1; k < n; ++k) x[k] = 0;

		_gpu->write_reg(x.data(), size_t(dst));
	}

	void set(const Reg dst, uint64 * const d) const override
	{
		const size_t n = _n;
		const uint64 * const weight = _weight.data();

		if (_reg_count <= 3)
		{
			std::cout << "[host memory] lowmem streamed weighted register upload(reg=" << size_t(dst) << ")" << std::endl;
			static constexpr size_t CHUNK = size_t(1) << 20;
			std::vector<uint64> x(CHUNK);
			for (size_t off = 0; off < n; off += CHUNK)
			{
				const size_t len = std::min(CHUNK, n - off);
				for (size_t t = 0; t < len; ++t)
				{
					const size_t k = off + t;
					const uint64 w = weight[2 * (k / 4 + (k % 4) * (n / 4)) + 0];
					x[t] = mod_mul(uint32(d[k]), w);
				}
				_gpu->write_reg_part(x.data(), len, size_t(dst), off);
			}
			return;
		}

		// weight
		std::vector<uint64> x(n);
		for (size_t k = 0; k < n; ++k)
		{
			const uint64 w = weight[2 * (k / 4 + (k % 4) * (n / 4)) + 0];
			x[k] = mod_mul(uint32(d[k]), w);
		}

		_gpu->write_reg(x.data(), size_t(dst));
	}

	void set_mpz(const Reg dst, const mpz_t & z) const override
	{
		if (_reg_count > 3)
		{
			engine::set_mpz(dst, z);
			return;
		}

		const size_t n = _n;
		const uint64 * const weight = _weight.data();
		const uint8 * const width = _digit_width.data();
		std::cout << "[host memory] lowmem streamed set_mpz(reg=" << size_t(dst) << ") begin" << std::endl;

		const size_t zbits = (mpz_sgn(z) == 0) ? 0 : mpz_sizeinbase(z, 2);
		const size_t words32 = zbits / 32 + 3;
		std::vector<uint32> v(words32, 0);
		size_t d_size = 0;
		mpz_export(v.data(), &d_size, -1, sizeof(uint32), 0, 0, z);

		static constexpr size_t CHUNK = size_t(1) << 20;
		std::vector<uint64> x(CHUNK);
		size_t bit_index = 0;
		for (size_t off = 0; off < n; off += CHUNK)
		{
			const size_t len = std::min(CHUNK, n - off);
			for (size_t t = 0; t < len; ++t)
			{
				const size_t k = off + t;
				const uint8 wdt = width[k];
				const size_t i = bit_index / (8 * sizeof(uint32));
				const size_t s = bit_index % (8 * sizeof(uint32));
				uint32 u = 0;
				if (i < v.size())
				{
					u = v[i] >> s;
					if (s != 0 && i + 1 < v.size()) u |= v[i + 1] << (32 - s);
				}
				u &= ((uint32(1) << wdt) - 1);
				const uint64 ww = weight[2 * (k / 4 + (k % 4) * (n / 4)) + 0];
				x[t] = mod_mul(u, ww);
				bit_index += wdt;
			}
			_gpu->write_reg_part(x.data(), len, size_t(dst), off);
		}
		std::cout << "[host memory] lowmem streamed set_mpz(reg=" << size_t(dst) << ") done" << std::endl;
	}

	void get(uint64 * const d, const Reg src) const override
	{
		const size_t n = _n;
		const uint64 * const weight = _weight.data();
		const uint8 * const width = _digit_width.data();

		_gpu->read_reg(d, size_t(src));

		// unweight, carry (strong)
		uint64 c = 0;
		for (size_t k = 0; k < n; ++k)
		{
			const uint64 wi = weight[2 * (k / 4 + (k % 4) * (n / 4)) + 1];
			d[k] = adc(mod_mul(d[k], wi), width[k], c);
		} 

		while (c != 0)
		{
			for (size_t k = 0; k < n; ++k)
			{
				d[k] = adc(d[k], width[k], c);
				if (c == 0) break;
			}
		}

		// encode
		for (size_t k = 0; k < n; ++k) d[k] = uint32(d[k]) | (uint64(width[k]) << 32);
	}

	void copy(const Reg dst, const Reg src) const override
	{
		_gpu->copy(size_t(dst), size_t(src));
	}

	void square_mul(const Reg rsrc, const uint32 a = 1) const override
	{
		const size_t n = _n, src = size_t(rsrc);

		switch (n)
		{
			case 1u <<  2:	_gpu->sqr4x1(src); break;
			case 1u <<  3:	_gpu->sqr8(src); break;
			case 1u <<  4:	_gpu->forward4_0(src); _gpu->sqr4(src); _gpu->backward4_0(src); break;
			case 1u <<  5:	_gpu->forward4_0(src); _gpu->sqr8(src); _gpu->backward4_0(src); break;
			case 1u <<  6:	_gpu->forward16_0(src); _gpu->sqr4(src); _gpu->backward16_0(src); break;
			case 1u <<  7:	_gpu->forward16_0(src); _gpu->sqr8(src); _gpu->backward16_0(src); break;
			case 1u <<  8:	_gpu->forward16_0(src); _gpu->sqr16(src); _gpu->backward16_0(src); break;
			case 1u <<  9:	_gpu->forward16_0(src); _gpu->sqr32(src); _gpu->backward16_0(src); break;
			case 1u << 10:	_gpu->forward16_0(src); _gpu->sqr64(src); _gpu->backward16_0(src); break;
			case 1u << 11:	_gpu->forward16_0(src); _gpu->sqr128(src); _gpu->backward16_0(src); break;
			case 1u << 12:	_gpu->forward64_0(src); _gpu->sqr64(src); _gpu->backward64_0(src); break;
			case 1u << 13:	_gpu->forward64_0(src); _gpu->sqr128(src); _gpu->backward64_0(src); break;
			case 1u << 14:	_gpu->forward64_0(src); _gpu->sqr256(src); _gpu->backward64_0(src); break;
			case 1u << 15:	_gpu->forward64_0(src); _gpu->sqr512(src); _gpu->backward64_0(src); break;
			case 1u << 16:	_gpu->forward64_0(src); _gpu->sqr1024(src); _gpu->backward64_0(src); break;
			case 1u << 17:	_gpu->forward256_0(src); _gpu->sqr512(src); _gpu->backward256_0(src); break;
			case 1u << 18:	_gpu->forward256_0(src); _gpu->sqr1024(src); _gpu->backward256_0(src); break;
			case 1u << 19:	_gpu->forward1024_0(src); _gpu->sqr512(src); _gpu->backward1024_0(src); break;
			case 1u << 20:	_gpu->forward1024_0(src); _gpu->sqr1024(src); _gpu->backward1024_0(src); break;
			case 1u << 21:	_gpu->forward64_0(src); _gpu->forward64(src, 1024, 8); _gpu->sqr512(src); _gpu->backward64(src, 1024, 8); _gpu->backward64_0(src); break;
			case 1u << 22:	_gpu->forward64_0(src); _gpu->forward64(src, 1024, 9); _gpu->sqr1024(src); _gpu->backward64(src, 1024, 9); _gpu->backward64_0(src); break;
			case 1u << 23:	_gpu->forward64_0(src); _gpu->forward256(src, 4096, 8); _gpu->sqr512(src); _gpu->backward256(src, 4096, 8); _gpu->backward64_0(src); break;
			case 1u << 24:	_gpu->forward64_0(src); _gpu->forward256(src, 4096, 9); _gpu->sqr1024(src); _gpu->backward256(src, 4096, 9); _gpu->backward64_0(src); break;
			case 1u << 25:	_gpu->forward256_0(src); _gpu->forward256(src, 16384, 8); _gpu->sqr512(src); _gpu->backward256(src, 16384, 8); _gpu->backward256_0(src); break;
			case 1u << 26:	_gpu->forward256_0(src); _gpu->forward256(src, 16384, 9); _gpu->sqr1024(src); _gpu->backward256(src, 16384, 9); _gpu->backward256_0(src); break;

			case 5u <<  3: _gpu->forward5_0(src); _gpu->sqr8(src); _gpu->backward5_0(src); break;
			// sqr16 cannot be applied because we have 80 / 8 = 10 global items and local size = 4
			case 5u <<  4: _gpu->forward20_0(src); _gpu->sqr4(src); _gpu->backward20_0(src); break;
			case 5u <<  5: _gpu->forward20_0(src); _gpu->sqr8(src); _gpu->backward20_0(src); break;
			case 5u <<  6: _gpu->forward20_0(src); _gpu->sqr16(src); _gpu->backward20_0(src); break;
			case 5u <<  7: _gpu->forward20_0(src); _gpu->sqr32(src); _gpu->backward20_0(src); break;
			case 5u <<  8: _gpu->forward20_0(src); _gpu->sqr64(src); _gpu->backward20_0(src); break;
			case 5u <<  9: _gpu->forward20_0(src); _gpu->sqr128(src); _gpu->backward20_0(src); break;
			case 5u << 10: _gpu->forward80_0(src); _gpu->sqr64(src); _gpu->backward80_0(src); break;
			case 5u << 11: _gpu->forward80_0(src); _gpu->sqr128(src); _gpu->backward80_0(src); break;
			case 5u << 12: _gpu->forward80_0(src); _gpu->sqr256(src); _gpu->backward80_0(src); break;
			case 5u << 13: _gpu->forward80_0(src); _gpu->sqr512(src); _gpu->backward80_0(src); break;
			case 5u << 14: _gpu->forward320_0(src); _gpu->sqr256(src); _gpu->backward320_0(src); break;
			case 5u << 15: _gpu->forward320_0(src); _gpu->sqr512(src); _gpu->backward320_0(src); break;
			case 5u << 16: _gpu->forward320_0(src); _gpu->sqr1024(src); _gpu->backward320_0(src); break;
			case 5u << 17: _gpu->forward80_0(src); _gpu->forward64(src, 1280, 6); _gpu->sqr128(src); _gpu->backward64(src, 1280, 6); _gpu->backward80_0(src); break;
			case 5u << 18: _gpu->forward80_0(src); _gpu->forward64(src, 1280, 7); _gpu->sqr256(src); _gpu->backward64(src, 1280, 7); _gpu->backward80_0(src); break;
			case 5u << 19: _gpu->forward80_0(src); _gpu->forward256(src, 5120, 6); _gpu->sqr128(src); _gpu->backward256(src, 5120, 6); _gpu->backward80_0(src); break;
			case 5u << 20: _gpu->forward80_0(src); _gpu->forward256(src, 5120, 7); _gpu->sqr256(src); _gpu->backward256(src, 5120, 7); _gpu->backward80_0(src); break;
			case 5u << 21: _gpu->forward80_0(src); _gpu->forward256(src, 5120, 8); _gpu->sqr512(src); _gpu->backward256(src, 5120, 8); _gpu->backward80_0(src); break;
			case 5u << 22: _gpu->forward80_0(src); _gpu->forward256(src, 5120, 9); _gpu->sqr1024(src); _gpu->backward256(src, 5120, 9); _gpu->backward80_0(src); break;
			case 5u << 23: _gpu->forward320_0(src); _gpu->forward256(src, 20480, 8); _gpu->sqr512(src); _gpu->backward256(src, 20480, 8); _gpu->backward320_0(src); break;
			case 5u << 24: _gpu->forward320_0(src); _gpu->forward256(src, 20480, 9); _gpu->sqr1024(src); _gpu->backward256(src, 20480, 9); _gpu->backward320_0(src); break;
			case 5u << 25: _gpu->forward320_0(src); _gpu->forward1024(src, 81920, 8); _gpu->sqr512(src); _gpu->backward1024(src, 81920, 8); _gpu->backward320_0(src); break;
			case 5u << 26: _gpu->forward320_0(src); _gpu->forward1024(src, 81920, 9); _gpu->sqr1024(src); _gpu->backward1024(src, 81920, 9); _gpu->backward320_0(src); break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}

		_gpu->carry_weight_mul(src, a);
	}

	void square_mul_copy(const Reg rsrc, const Reg rdst_copy, const uint32 a = 1) const override
	{
		const size_t n = _n, src = size_t(rsrc), dcopy = size_t(rdst_copy);

		switch (n)
		{
			case 1u <<  2:	_gpu->sqr4x1(src); break;
			case 1u <<  3:	_gpu->sqr8(src); break;
			case 1u <<  4:	_gpu->forward4_0(src); _gpu->sqr4(src); _gpu->backward4_0(src); break;
			case 1u <<  5:	_gpu->forward4_0(src); _gpu->sqr8(src); _gpu->backward4_0(src); break;
			case 1u <<  6:	_gpu->forward16_0(src); _gpu->sqr4(src); _gpu->backward16_0(src); break;
			case 1u <<  7:	_gpu->forward16_0(src); _gpu->sqr8(src); _gpu->backward16_0(src); break;
			case 1u <<  8:	_gpu->forward16_0(src); _gpu->sqr16(src); _gpu->backward16_0(src); break;
			case 1u <<  9:	_gpu->forward16_0(src); _gpu->sqr32(src); _gpu->backward16_0(src); break;
			case 1u << 10:	_gpu->forward16_0(src); _gpu->sqr64(src); _gpu->backward16_0(src); break;
			case 1u << 11:	_gpu->forward16_0(src); _gpu->sqr128(src); _gpu->backward16_0(src); break;
			case 1u << 12:	_gpu->forward64_0(src); _gpu->sqr64(src); _gpu->backward64_0(src); break;
			case 1u << 13:	_gpu->forward64_0(src); _gpu->sqr128(src); _gpu->backward64_0(src); break;
			case 1u << 14:	_gpu->forward64_0(src); _gpu->sqr256(src); _gpu->backward64_0(src); break;
			case 1u << 15:	_gpu->forward64_0(src); _gpu->sqr512(src); _gpu->backward64_0(src); break;
			case 1u << 16:	_gpu->forward64_0(src); _gpu->sqr1024(src); _gpu->backward64_0(src); break;
			case 1u << 17:	_gpu->forward256_0(src); _gpu->sqr512(src); _gpu->backward256_0(src); break;
			case 1u << 18:	_gpu->forward256_0(src); _gpu->sqr1024(src); _gpu->backward256_0(src); break;
			case 1u << 19:	_gpu->forward1024_0(src); _gpu->sqr512(src); _gpu->backward1024_0(src); break;
			case 1u << 20:	_gpu->forward1024_0(src); _gpu->sqr1024(src); _gpu->backward1024_0(src); break;
			case 1u << 21:	_gpu->forward64_0(src); _gpu->forward64(src, 1024, 8); _gpu->sqr512(src); _gpu->backward64(src, 1024, 8); _gpu->backward64_0(src); break;
			case 1u << 22:	_gpu->forward64_0(src); _gpu->forward64(src, 1024, 9); _gpu->sqr1024(src); _gpu->backward64(src, 1024, 9); _gpu->backward64_0(src); break;
			case 1u << 23:	_gpu->forward64_0(src); _gpu->forward256(src, 4096, 8); _gpu->sqr512(src); _gpu->backward256(src, 4096, 8); _gpu->backward64_0(src); break;
			case 1u << 24:	_gpu->forward64_0(src); _gpu->forward256(src, 4096, 9); _gpu->sqr1024(src); _gpu->backward256(src, 4096, 9); _gpu->backward64_0(src); break;
			case 1u << 25:	_gpu->forward256_0(src); _gpu->forward256(src, 16384, 8); _gpu->sqr512(src); _gpu->backward256(src, 16384, 8); _gpu->backward256_0(src); break;
			case 1u << 26:	_gpu->forward256_0(src); _gpu->forward256(src, 16384, 9); _gpu->sqr1024(src); _gpu->backward256(src, 16384, 9); _gpu->backward256_0(src); break;

			case 5u <<  3: _gpu->forward5_0(src); _gpu->sqr8(src); _gpu->backward5_0(src); break;
			case 5u <<  4: _gpu->forward20_0(src); _gpu->sqr4(src); _gpu->backward20_0(src); break;
			case 5u <<  5: _gpu->forward20_0(src); _gpu->sqr8(src); _gpu->backward20_0(src); break;
			case 5u <<  6: _gpu->forward20_0(src); _gpu->sqr16(src); _gpu->backward20_0(src); break;
			case 5u <<  7: _gpu->forward20_0(src); _gpu->sqr32(src); _gpu->backward20_0(src); break;
			case 5u <<  8: _gpu->forward20_0(src); _gpu->sqr64(src); _gpu->backward20_0(src); break;
			case 5u <<  9: _gpu->forward20_0(src); _gpu->sqr128(src); _gpu->backward20_0(src); break;
			case 5u << 10: _gpu->forward80_0(src); _gpu->sqr64(src); _gpu->backward80_0(src); break;
			case 5u << 11: _gpu->forward80_0(src); _gpu->sqr128(src); _gpu->backward80_0(src); break;
			case 5u << 12: _gpu->forward80_0(src); _gpu->sqr256(src); _gpu->backward80_0(src); break;
			case 5u << 13: _gpu->forward80_0(src); _gpu->sqr512(src); _gpu->backward80_0(src); break;
			case 5u << 14: _gpu->forward320_0(src); _gpu->sqr256(src); _gpu->backward320_0(src); break;
			case 5u << 15: _gpu->forward320_0(src); _gpu->sqr512(src); _gpu->backward320_0(src); break;
			case 5u << 16: _gpu->forward320_0(src); _gpu->sqr1024(src); _gpu->backward320_0(src); break;
			case 5u << 17: _gpu->forward80_0(src); _gpu->forward64(src, 1280, 6); _gpu->sqr128(src); _gpu->backward64(src, 1280, 6); _gpu->backward80_0(src); break;
			case 5u << 18: _gpu->forward80_0(src); _gpu->forward64(src, 1280, 7); _gpu->sqr256(src); _gpu->backward64(src, 1280, 7); _gpu->backward80_0(src); break;
			case 5u << 19: _gpu->forward80_0(src); _gpu->forward256(src, 5120, 6); _gpu->sqr128(src); _gpu->backward256(src, 5120, 6); _gpu->backward80_0(src); break;
			case 5u << 20: _gpu->forward80_0(src); _gpu->forward256(src, 5120, 7); _gpu->sqr256(src); _gpu->backward256(src, 5120, 7); _gpu->backward80_0(src); break;
			case 5u << 21: _gpu->forward80_0(src); _gpu->forward256(src, 5120, 8); _gpu->sqr512(src); _gpu->backward256(src, 5120, 8); _gpu->backward80_0(src); break;
			case 5u << 22: _gpu->forward80_0(src); _gpu->forward256(src, 5120, 9); _gpu->sqr1024(src); _gpu->backward256(src, 5120, 9); _gpu->backward80_0(src); break;
			case 5u << 23: _gpu->forward320_0(src); _gpu->forward256(src, 20480, 8); _gpu->sqr512(src); _gpu->backward256(src, 20480, 8); _gpu->backward320_0(src); break;
			case 5u << 24: _gpu->forward320_0(src); _gpu->forward256(src, 20480, 9); _gpu->sqr1024(src); _gpu->backward256(src, 20480, 9); _gpu->backward320_0(src); break;
			case 5u << 25: _gpu->forward320_0(src); _gpu->forward1024(src, 81920, 8); _gpu->sqr512(src); _gpu->backward1024(src, 81920, 8); _gpu->backward320_0(src); break;
			case 5u << 26: _gpu->forward320_0(src); _gpu->forward1024(src, 81920, 9); _gpu->sqr1024(src); _gpu->backward1024(src, 81920, 9); _gpu->backward320_0(src); break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}

		_gpu->carry_weight_mul_copy(src, dcopy, a);
	}

	void set_multiplicand(const Reg rdst, const Reg rsrc) const override
	{
		if (rsrc != rdst) copy(rdst, rsrc);

		const size_t n = _n, dst = size_t(rdst);

		switch (n)
		{
			case 1u <<  2:	_gpu->forward_mul4x1(dst); break;
			case 1u <<  3:	_gpu->forward_mul8(dst); break;
			case 1u <<  4:	_gpu->forward4_0(dst); _gpu->forward_mul4(dst); break;
			case 1u <<  5:	_gpu->forward4_0(dst); _gpu->forward_mul8(dst); break;
			case 1u <<  6:	_gpu->forward16_0(dst); _gpu->forward_mul4(dst); break;
			case 1u <<  7:	_gpu->forward16_0(dst); _gpu->forward_mul8(dst); break;
			case 1u <<  8:	_gpu->forward16_0(dst); _gpu->forward_mul16(dst); break;
			case 1u <<  9:	_gpu->forward16_0(dst); _gpu->forward_mul32(dst); break;
			case 1u << 10:	_gpu->forward16_0(dst); _gpu->forward_mul64(dst); break;
			case 1u << 11:	_gpu->forward16_0(dst); _gpu->forward_mul128(dst); break;
			case 1u << 12:	_gpu->forward64_0(dst); _gpu->forward_mul64(dst); break;
			case 1u << 13:	_gpu->forward64_0(dst); _gpu->forward_mul128(dst); break;
			case 1u << 14:	_gpu->forward64_0(dst); _gpu->forward_mul256(dst); break;
			case 1u << 15:	_gpu->forward64_0(dst); _gpu->forward_mul512(dst); break;
			case 1u << 16:	_gpu->forward64_0(dst); _gpu->forward_mul1024(dst); break;
			case 1u << 17:	_gpu->forward256_0(dst); _gpu->forward_mul512(dst); break;
			case 1u << 18:	_gpu->forward256_0(dst); _gpu->forward_mul1024(dst); break;
			case 1u << 19:	_gpu->forward1024_0(dst); _gpu->forward_mul512(dst); break;
			case 1u << 20:	_gpu->forward1024_0(dst); _gpu->forward_mul1024(dst); break;
			case 1u << 21:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 8); _gpu->forward_mul512(dst); break;
			case 1u << 22:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 9); _gpu->forward_mul1024(dst); break;
			case 1u << 23:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 8); _gpu->forward_mul512(dst); break;
			case 1u << 24:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 9); _gpu->forward_mul1024(dst); break;
			case 1u << 25:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 8); _gpu->forward_mul512(dst); break;
			case 1u << 26:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 9); _gpu->forward_mul1024(dst); break;

			case 5u <<  3: _gpu->forward5_0(dst); _gpu->forward_mul8(dst); break;
			case 5u <<  4: _gpu->forward20_0(dst); _gpu->forward_mul4(dst); break;
			case 5u <<  5: _gpu->forward20_0(dst); _gpu->forward_mul8(dst); break;
			case 5u <<  6: _gpu->forward20_0(dst); _gpu->forward_mul16(dst); break;
			case 5u <<  7: _gpu->forward20_0(dst); _gpu->forward_mul32(dst); break;
			case 5u <<  8: _gpu->forward20_0(dst); _gpu->forward_mul64(dst); break;
			case 5u <<  9: _gpu->forward20_0(dst); _gpu->forward_mul128(dst); break;
			case 5u << 10: _gpu->forward80_0(dst); _gpu->forward_mul64(dst); break;
			case 5u << 11: _gpu->forward80_0(dst); _gpu->forward_mul128(dst); break;
			case 5u << 12: _gpu->forward80_0(dst); _gpu->forward_mul256(dst); break;
			case 5u << 13: _gpu->forward80_0(dst); _gpu->forward_mul512(dst); break;
			case 5u << 14: _gpu->forward320_0(dst); _gpu->forward_mul256(dst); break;
			case 5u << 15: _gpu->forward320_0(dst); _gpu->forward_mul512(dst); break;
			case 5u << 16: _gpu->forward320_0(dst); _gpu->forward_mul1024(dst); break;
			case 5u << 17: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 6); _gpu->forward_mul128(dst); break;
			case 5u << 18: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 7); _gpu->forward_mul256(dst); break;
			case 5u << 19: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 6); _gpu->forward_mul128(dst); break;
			case 5u << 20: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 7); _gpu->forward_mul256(dst); break;
			case 5u << 21: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 8); _gpu->forward_mul512(dst); break;
			case 5u << 22: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 9); _gpu->forward_mul1024(dst); break;
			case 5u << 23: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 8); _gpu->forward_mul512(dst); break;
			case 5u << 24: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 9); _gpu->forward_mul1024(dst); break;
			case 5u << 25: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 8); _gpu->forward_mul512(dst); break;
			case 5u << 26: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 9); _gpu->forward_mul1024(dst); break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}
	}

	void set_multiplicand2(const Reg rdst, const Reg rsrc) const override
	{
		copy(rdst, rsrc);

		const size_t n = _n, dst = size_t(rdst);

		switch (n)
		{
			//case 1u <<  2:	_gpu->mul4x1(dst, src); break;
			//case 1u <<  3:	_gpu->mul8(dst, src); break;
			case 1u <<  2: break;
			case 1u <<  3: break;
			case 1u <<  4:	_gpu->forward4_0(dst); break;
			case 1u <<  5:	_gpu->forward4_0(dst); break;
			case 1u <<  6:	_gpu->forward16_0(dst); break;
			case 1u <<  7:	_gpu->forward16_0(dst); break;
			case 1u <<  8:	_gpu->forward16_0(dst);  break;
			case 1u <<  9:	_gpu->forward16_0(dst);  break;
			case 1u << 10:	_gpu->forward16_0(dst);  break;
			case 1u << 11:	_gpu->forward16_0(dst);  break;
			case 1u << 12:	_gpu->forward64_0(dst); break;
			case 1u << 13:	_gpu->forward64_0(dst);  break;
			case 1u << 14:	_gpu->forward64_0(dst);  break;
			case 1u << 15:	_gpu->forward64_0(dst);  break;
			case 1u << 16:	_gpu->forward64_0(dst);  break;
			case 1u << 17:	_gpu->forward256_0(dst); break;
			case 1u << 18:	_gpu->forward256_0(dst); break;
			case 1u << 19:	_gpu->forward1024_0(dst);  break;
			case 1u << 20:	_gpu->forward1024_0(dst); break;
			case 1u << 21:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 8); break;
			case 1u << 22:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 9); break;
			case 1u << 23:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 8);  break;
			case 1u << 24:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 9);  break;
			case 1u << 25:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 8); break;
			case 1u << 26:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 9); break;

			case 5u <<  3: _gpu->forward5_0(dst);   break;
			case 5u <<  4: _gpu->forward20_0(dst);  break;
			case 5u <<  5: _gpu->forward20_0(dst);  break;
			case 5u <<  6: _gpu->forward20_0(dst);  break;
			case 5u <<  7: _gpu->forward20_0(dst);  break;
			case 5u <<  8: _gpu->forward20_0(dst);  break;
			case 5u <<  9: _gpu->forward20_0(dst);  break;
			case 5u << 10: _gpu->forward80_0(dst);  break;
			case 5u << 11: _gpu->forward80_0(dst);  break;
			case 5u << 12: _gpu->forward80_0(dst);  break;
			case 5u << 13: _gpu->forward80_0(dst);  break;
			case 5u << 14: _gpu->forward320_0(dst); break;
			case 5u << 15: _gpu->forward320_0(dst); break;
			case 5u << 16: _gpu->forward320_0(dst); break;
			case 5u << 17: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 6); break;
			case 5u << 18: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 7); break;
			case 5u << 19: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 6); break;
			case 5u << 20: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 7); break;
			case 5u << 21: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 8); break;
			case 5u << 22: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 9); break;
			case 5u << 23: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 8); break;
			case 5u << 24: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 9); break;
			case 5u << 25: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 8); break;
			case 5u << 26: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 9); break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}
	}

	void mul(const Reg rdst, const Reg rsrc, const uint32 a = 1) const override
	{
		const size_t n = _n, dst = size_t(rdst), src = size_t(rsrc);

		switch (n)
		{
			case 1u <<  2:	_gpu->mul4x1(dst, src); break;
			case 1u <<  3:	_gpu->mul8(dst, src); break;
			case 1u <<  4:	_gpu->forward4_0(dst); _gpu->mul4(dst, src); _gpu->backward4_0(dst); break;
			case 1u <<  5:	_gpu->forward4_0(dst); _gpu->mul8(dst, src); _gpu->backward4_0(dst); break;
			case 1u <<  6:	_gpu->forward16_0(dst); _gpu->mul4(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  7:	_gpu->forward16_0(dst); _gpu->mul8(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  8:	_gpu->forward16_0(dst); _gpu->mul16(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  9:	_gpu->forward16_0(dst); _gpu->mul32(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 10:	_gpu->forward16_0(dst); _gpu->mul64(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 11:	_gpu->forward16_0(dst); _gpu->mul128(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 12:	_gpu->forward64_0(dst); _gpu->mul64(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 13:	_gpu->forward64_0(dst); _gpu->mul128(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 14:	_gpu->forward64_0(dst); _gpu->mul256(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 15:	_gpu->forward64_0(dst); _gpu->mul512(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 16:	_gpu->forward64_0(dst); _gpu->mul1024(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 17:	_gpu->forward256_0(dst); _gpu->mul512(dst, src); _gpu->backward256_0(dst); break;
			case 1u << 18:	_gpu->forward256_0(dst); _gpu->mul1024(dst, src); _gpu->backward256_0(dst); break;
			case 1u << 19:	_gpu->forward1024_0(dst); _gpu->mul512(dst, src); _gpu->backward1024_0(dst); break;
			case 1u << 20:	_gpu->forward1024_0(dst); _gpu->mul1024(dst, src); _gpu->backward1024_0(dst); break;
			case 1u << 21:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 8); _gpu->mul512(dst, src); _gpu->backward64(dst, 1024, 8); _gpu->backward64_0(dst); break;
			case 1u << 22:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 9); _gpu->mul1024(dst, src); _gpu->backward64(dst, 1024, 9); _gpu->backward64_0(dst); break;
			case 1u << 23:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 4096, 8); _gpu->backward64_0(dst); break;
			case 1u << 24:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 4096, 9); _gpu->backward64_0(dst); break;
			case 1u << 25:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 16384, 8); _gpu->backward256_0(dst); break;
			case 1u << 26:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 16384, 9); _gpu->backward256_0(dst); break;

			case 5u <<  3: _gpu->forward5_0(dst); _gpu->mul8(dst, src); _gpu->backward5_0(dst); break;
			case 5u <<  4: _gpu->forward20_0(dst); _gpu->mul4(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  5: _gpu->forward20_0(dst); _gpu->mul8(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  6: _gpu->forward20_0(dst); _gpu->mul16(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  7: _gpu->forward20_0(dst); _gpu->mul32(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  8: _gpu->forward20_0(dst); _gpu->mul64(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  9: _gpu->forward20_0(dst); _gpu->mul128(dst, src); _gpu->backward20_0(dst); break;
			case 5u << 10: _gpu->forward80_0(dst); _gpu->mul64(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 11: _gpu->forward80_0(dst); _gpu->mul128(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 12: _gpu->forward80_0(dst); _gpu->mul256(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 13: _gpu->forward80_0(dst); _gpu->mul512(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 14: _gpu->forward320_0(dst); _gpu->mul256(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 15: _gpu->forward320_0(dst); _gpu->mul512(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 16: _gpu->forward320_0(dst); _gpu->mul1024(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 17: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 6); _gpu->mul128(dst, src); _gpu->backward64(dst, 1280, 6); _gpu->backward80_0(dst); break;
			case 5u << 18: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 7); _gpu->mul256(dst, src); _gpu->backward64(dst, 1280, 7); _gpu->backward80_0(dst); break;
			case 5u << 19: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 6); _gpu->mul128(dst, src); _gpu->backward256(dst, 5120, 6); _gpu->backward80_0(dst); break;
			case 5u << 20: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 7); _gpu->mul256(dst, src); _gpu->backward256(dst, 5120, 7); _gpu->backward80_0(dst); break;
			case 5u << 21: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 5120, 8); _gpu->backward80_0(dst); break;
			case 5u << 22: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 5120, 9); _gpu->backward80_0(dst); break;
			case 5u << 23: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 20480, 8); _gpu->backward320_0(dst); break;
			case 5u << 24: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 20480, 9); _gpu->backward320_0(dst); break;
			case 5u << 25: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 8); _gpu->mul512(dst, src); _gpu->backward1024(dst, 81920, 8); _gpu->backward320_0(dst); break;
			case 5u << 26: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 9); _gpu->mul1024(dst, src); _gpu->backward1024(dst, 81920, 9); _gpu->backward320_0(dst); break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}

		_gpu->carry_weight_mul(dst, a);
	}
	
	void mul_new_core(const Reg rdst, const Reg rsrc) const
	{
		const size_t n = _n, dst = size_t(rdst), src = size_t(rsrc);

		switch (n)
		{
			case 1u <<  2:	_gpu->mul4x1(dst, src); break;
			case 1u <<  3:	_gpu->mul8(dst, src); break;
			case 1u <<  4:	_gpu->mul4(dst, src); _gpu->backward4_0(dst); break;
			case 1u <<  5:	_gpu->mul8(dst, src); _gpu->backward4_0(dst); break;
			case 1u <<  6:	_gpu->mul4(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  7:	_gpu->mul8(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  8:	_gpu->mul16(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  9:	_gpu->mul32(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 10:	_gpu->mul64(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 11:	_gpu->mul128(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 12:	_gpu->mul64(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 13:	_gpu->mul128(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 14:	_gpu->mul256(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 15:	_gpu->mul512(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 16:	_gpu->mul1024(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 17:	_gpu->mul512(dst, src); _gpu->backward256_0(dst); break;
			case 1u << 18:	_gpu->mul1024(dst, src); _gpu->backward256_0(dst); break;
			case 1u << 19:	_gpu->mul512(dst, src); _gpu->backward1024_0(dst); break;
			case 1u << 20:	_gpu->mul1024(dst, src); _gpu->backward1024_0(dst); break;
			case 1u << 21:	_gpu->mul512(dst, src); _gpu->backward64(dst, 1024, 8); _gpu->backward64_0(dst); break;
			case 1u << 22:	_gpu->mul1024(dst, src); _gpu->backward64(dst, 1024, 9); _gpu->backward64_0(dst); break;
			case 1u << 23:	_gpu->mul512(dst, src); _gpu->backward256(dst, 4096, 8); _gpu->backward64_0(dst); break;
			case 1u << 24:	_gpu->mul1024(dst, src); _gpu->backward256(dst, 4096, 9); _gpu->backward64_0(dst); break;
			case 1u << 25:	_gpu->mul512(dst, src); _gpu->backward256(dst, 16384, 8); _gpu->backward256_0(dst); break;
			case 1u << 26:	_gpu->mul1024(dst, src); _gpu->backward256(dst, 16384, 9); _gpu->backward256_0(dst); break;

			case 5u <<  3: _gpu->mul8(dst, src); _gpu->backward5_0(dst); break;
			case 5u <<  4: _gpu->mul4(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  5: _gpu->mul8(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  6: _gpu->mul16(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  7: _gpu->mul32(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  8: _gpu->mul64(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  9: _gpu->mul128(dst, src); _gpu->backward20_0(dst); break;
			case 5u << 10: _gpu->mul64(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 11: _gpu->mul128(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 12: _gpu->mul256(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 13: _gpu->mul512(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 14: _gpu->mul256(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 15: _gpu->mul512(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 16: _gpu->mul1024(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 17: _gpu->mul128(dst, src); _gpu->backward64(dst, 1280, 6); _gpu->backward80_0(dst); break;
			case 5u << 18: _gpu->mul256(dst, src); _gpu->backward64(dst, 1280, 7); _gpu->backward80_0(dst); break;
			case 5u << 19: _gpu->mul128(dst, src); _gpu->backward256(dst, 5120, 6); _gpu->backward80_0(dst); break;
			case 5u << 20: _gpu->mul256(dst, src); _gpu->backward256(dst, 5120, 7); _gpu->backward80_0(dst); break;
			case 5u << 21: _gpu->mul512(dst, src); _gpu->backward256(dst, 5120, 8); _gpu->backward80_0(dst); break;
			case 5u << 22: _gpu->mul1024(dst, src); _gpu->backward256(dst, 5120, 9); _gpu->backward80_0(dst); break;
			case 5u << 23: _gpu->mul512(dst, src); _gpu->backward256(dst, 20480, 8); _gpu->backward320_0(dst); break;
			case 5u << 24: _gpu->mul1024(dst, src); _gpu->backward256(dst, 20480, 9); _gpu->backward320_0(dst); break;
			case 5u << 25: _gpu->mul512(dst, src); _gpu->backward1024(dst, 81920, 8); _gpu->backward320_0(dst); break;
			case 5u << 26: _gpu->mul1024(dst, src); _gpu->backward1024(dst, 81920, 9); _gpu->backward320_0(dst); break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}
	}

	void mul_new(const Reg rdst, const Reg rsrc, const uint32 a = 1) const override
	{
		mul_new_core(rdst, rsrc);
		_gpu->carry_weight_mul(size_t(rdst), a);
	}

	void mul_pair_unit(const Reg dst0, const Reg src0, const Reg dst1, const Reg src1) const override
	{
		set_multiplicand(src0, src0);
		mul(dst0, src0);
		set_multiplicand(src1, src1);
		mul(dst1, src1);
	}
	void mul_add(const Reg rdst, const Reg rsrc, const Reg radd, const uint32 a = 1) const override
	{
		const size_t n = _n, dst = size_t(rdst), src = size_t(rsrc);

		switch (n)
		{
			case 1u <<  2:	_gpu->mul4x1(dst, src); break;
			case 1u <<  3:	_gpu->mul8(dst, src); break;
			case 1u <<  4:	_gpu->forward4_0(dst); _gpu->mul4(dst, src); _gpu->backward4_0(dst); break;
			case 1u <<  5:	_gpu->forward4_0(dst); _gpu->mul8(dst, src); _gpu->backward4_0(dst); break;
			case 1u <<  6:	_gpu->forward16_0(dst); _gpu->mul4(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  7:	_gpu->forward16_0(dst); _gpu->mul8(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  8:	_gpu->forward16_0(dst); _gpu->mul16(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  9:	_gpu->forward16_0(dst); _gpu->mul32(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 10:	_gpu->forward16_0(dst); _gpu->mul64(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 11:	_gpu->forward16_0(dst); _gpu->mul128(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 12:	_gpu->forward64_0(dst); _gpu->mul64(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 13:	_gpu->forward64_0(dst); _gpu->mul128(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 14:	_gpu->forward64_0(dst); _gpu->mul256(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 15:	_gpu->forward64_0(dst); _gpu->mul512(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 16:	_gpu->forward64_0(dst); _gpu->mul1024(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 17:	_gpu->forward256_0(dst); _gpu->mul512(dst, src); _gpu->backward256_0(dst); break;
			case 1u << 18:	_gpu->forward256_0(dst); _gpu->mul1024(dst, src); _gpu->backward256_0(dst); break;
			case 1u << 19:	_gpu->forward1024_0(dst); _gpu->mul512(dst, src); _gpu->backward1024_0(dst); break;
			case 1u << 20:	_gpu->forward1024_0(dst); _gpu->mul1024(dst, src); _gpu->backward1024_0(dst); break;
			case 1u << 21:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 8); _gpu->mul512(dst, src); _gpu->backward64(dst, 1024, 8); _gpu->backward64_0(dst); break;
			case 1u << 22:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 9); _gpu->mul1024(dst, src); _gpu->backward64(dst, 1024, 9); _gpu->backward64_0(dst); break;
			case 1u << 23:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 4096, 8); _gpu->backward64_0(dst); break;
			case 1u << 24:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 4096, 9); _gpu->backward64_0(dst); break;
			case 1u << 25:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 16384, 8); _gpu->backward256_0(dst); break;
			case 1u << 26:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 16384, 9); _gpu->backward256_0(dst); break;

			case 5u <<  3: _gpu->forward5_0(dst); _gpu->mul8(dst, src); _gpu->backward5_0(dst); break;
			case 5u <<  4: _gpu->forward20_0(dst); _gpu->mul4(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  5: _gpu->forward20_0(dst); _gpu->mul8(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  6: _gpu->forward20_0(dst); _gpu->mul16(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  7: _gpu->forward20_0(dst); _gpu->mul32(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  8: _gpu->forward20_0(dst); _gpu->mul64(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  9: _gpu->forward20_0(dst); _gpu->mul128(dst, src); _gpu->backward20_0(dst); break;
			case 5u << 10: _gpu->forward80_0(dst); _gpu->mul64(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 11: _gpu->forward80_0(dst); _gpu->mul128(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 12: _gpu->forward80_0(dst); _gpu->mul256(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 13: _gpu->forward80_0(dst); _gpu->mul512(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 14: _gpu->forward320_0(dst); _gpu->mul256(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 15: _gpu->forward320_0(dst); _gpu->mul512(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 16: _gpu->forward320_0(dst); _gpu->mul1024(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 17: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 6); _gpu->mul128(dst, src); _gpu->backward64(dst, 1280, 6); _gpu->backward80_0(dst); break;
			case 5u << 18: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 7); _gpu->mul256(dst, src); _gpu->backward64(dst, 1280, 7); _gpu->backward80_0(dst); break;
			case 5u << 19: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 6); _gpu->mul128(dst, src); _gpu->backward256(dst, 5120, 6); _gpu->backward80_0(dst); break;
			case 5u << 20: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 7); _gpu->mul256(dst, src); _gpu->backward256(dst, 5120, 7); _gpu->backward80_0(dst); break;
			case 5u << 21: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 5120, 8); _gpu->backward80_0(dst); break;
			case 5u << 22: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 5120, 9); _gpu->backward80_0(dst); break;
			case 5u << 23: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 20480, 8); _gpu->backward320_0(dst); break;
			case 5u << 24: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 20480, 9); _gpu->backward320_0(dst); break;
			case 5u << 25: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 8); _gpu->mul512(dst, src); _gpu->backward1024(dst, 81920, 8); _gpu->backward320_0(dst); break;
			case 5u << 26: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 9); _gpu->mul1024(dst, src); _gpu->backward1024(dst, 81920, 9); _gpu->backward320_0(dst); break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}

		_gpu->carry_weight_muladd(dst, size_t(radd), a);
	}
	void mul_copy(const Reg rdst, const Reg rsrc, const Reg rdst_copy, const uint32 a = 1) const override
	{
		const size_t n = _n, dst = size_t(rdst), src = size_t(rsrc), dcopy = size_t(rdst_copy);

		switch (n)
		{
			case 1u <<  2:	_gpu->mul4x1(dst, src); break;
			case 1u <<  3:	_gpu->mul8(dst, src); break;
			case 1u <<  4:	_gpu->forward4_0(dst); _gpu->mul4(dst, src); _gpu->backward4_0(dst); break;
			case 1u <<  5:	_gpu->forward4_0(dst); _gpu->mul8(dst, src); _gpu->backward4_0(dst); break;
			case 1u <<  6:	_gpu->forward16_0(dst); _gpu->mul4(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  7:	_gpu->forward16_0(dst); _gpu->mul8(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  8:	_gpu->forward16_0(dst); _gpu->mul16(dst, src); _gpu->backward16_0(dst); break;
			case 1u <<  9:	_gpu->forward16_0(dst); _gpu->mul32(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 10:	_gpu->forward16_0(dst); _gpu->mul64(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 11:	_gpu->forward16_0(dst); _gpu->mul128(dst, src); _gpu->backward16_0(dst); break;
			case 1u << 12:	_gpu->forward64_0(dst); _gpu->mul64(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 13:	_gpu->forward64_0(dst); _gpu->mul128(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 14:	_gpu->forward64_0(dst); _gpu->mul256(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 15:	_gpu->forward64_0(dst); _gpu->mul512(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 16:	_gpu->forward64_0(dst); _gpu->mul1024(dst, src); _gpu->backward64_0(dst); break;
			case 1u << 17:	_gpu->forward256_0(dst); _gpu->mul512(dst, src); _gpu->backward256_0(dst); break;
			case 1u << 18:	_gpu->forward256_0(dst); _gpu->mul1024(dst, src); _gpu->backward256_0(dst); break;
			case 1u << 19:	_gpu->forward1024_0(dst); _gpu->mul512(dst, src); _gpu->backward1024_0(dst); break;
			case 1u << 20:	_gpu->forward1024_0(dst); _gpu->mul1024(dst, src); _gpu->backward1024_0(dst); break;
			case 1u << 21:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 8); _gpu->mul512(dst, src); _gpu->backward64(dst, 1024, 8); _gpu->backward64_0(dst); break;
			case 1u << 22:	_gpu->forward64_0(dst); _gpu->forward64(dst, 1024, 9); _gpu->mul1024(dst, src); _gpu->backward64(dst, 1024, 9); _gpu->backward64_0(dst); break;
			case 1u << 23:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 4096, 8); _gpu->backward64_0(dst); break;
			case 1u << 24:	_gpu->forward64_0(dst); _gpu->forward256(dst, 4096, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 4096, 9); _gpu->backward64_0(dst); break;
			case 1u << 25:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 16384, 8); _gpu->backward256_0(dst); break;
			case 1u << 26:	_gpu->forward256_0(dst); _gpu->forward256(dst, 16384, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 16384, 9); _gpu->backward256_0(dst); break;

			case 5u <<  3: _gpu->forward5_0(dst); _gpu->mul8(dst, src); _gpu->backward5_0(dst); break;
			case 5u <<  4: _gpu->forward20_0(dst); _gpu->mul4(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  5: _gpu->forward20_0(dst); _gpu->mul8(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  6: _gpu->forward20_0(dst); _gpu->mul16(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  7: _gpu->forward20_0(dst); _gpu->mul32(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  8: _gpu->forward20_0(dst); _gpu->mul64(dst, src); _gpu->backward20_0(dst); break;
			case 5u <<  9: _gpu->forward20_0(dst); _gpu->mul128(dst, src); _gpu->backward20_0(dst); break;
			case 5u << 10: _gpu->forward80_0(dst); _gpu->mul64(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 11: _gpu->forward80_0(dst); _gpu->mul128(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 12: _gpu->forward80_0(dst); _gpu->mul256(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 13: _gpu->forward80_0(dst); _gpu->mul512(dst, src); _gpu->backward80_0(dst); break;
			case 5u << 14: _gpu->forward320_0(dst); _gpu->mul256(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 15: _gpu->forward320_0(dst); _gpu->mul512(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 16: _gpu->forward320_0(dst); _gpu->mul1024(dst, src); _gpu->backward320_0(dst); break;
			case 5u << 17: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 6); _gpu->mul128(dst, src); _gpu->backward64(dst, 1280, 6); _gpu->backward80_0(dst); break;
			case 5u << 18: _gpu->forward80_0(dst); _gpu->forward64(dst, 1280, 7); _gpu->mul256(dst, src); _gpu->backward64(dst, 1280, 7); _gpu->backward80_0(dst); break;
			case 5u << 19: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 6); _gpu->mul128(dst, src); _gpu->backward256(dst, 5120, 6); _gpu->backward80_0(dst); break;
			case 5u << 20: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 7); _gpu->mul256(dst, src); _gpu->backward256(dst, 5120, 7); _gpu->backward80_0(dst); break;
			case 5u << 21: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 5120, 8); _gpu->backward80_0(dst); break;
			case 5u << 22: _gpu->forward80_0(dst); _gpu->forward256(dst, 5120, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 5120, 9); _gpu->backward80_0(dst); break;
			case 5u << 23: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 8); _gpu->mul512(dst, src); _gpu->backward256(dst, 20480, 8); _gpu->backward320_0(dst); break;
			case 5u << 24: _gpu->forward320_0(dst); _gpu->forward256(dst, 20480, 9); _gpu->mul1024(dst, src); _gpu->backward256(dst, 20480, 9); _gpu->backward320_0(dst); break;
			case 5u << 25: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 8); _gpu->mul512(dst, src); _gpu->backward1024(dst, 81920, 8); _gpu->backward320_0(dst); break;
			case 5u << 26: _gpu->forward320_0(dst); _gpu->forward1024(dst, 81920, 9); _gpu->mul1024(dst, src); _gpu->backward1024(dst, 81920, 9); _gpu->backward320_0(dst); break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}

		_gpu->carry_weight_mul_copy(dst, dcopy, a);
	}

	void sub(const Reg src, const uint32 a) const override { _gpu->subtract(size_t(src), a); }

	void add(const Reg dst, const Reg src) const override
	{
		_gpu->carry_weight_add(dst, src);
	}
	
	void sub_reg(const Reg dst, const Reg src) const override
	{
		_gpu->carry_weight_sub_safe(size_t(dst), size_t(src));
	}

	void addsub(const Reg sum_out, const Reg diff_out, const Reg a, const Reg b) const override
	{
		_gpu->carry_weight_addsub((size_t)sum_out, (size_t)diff_out, (size_t)a, (size_t)b);
	}

	void addsub_copy(const Reg sum, const Reg diff, const Reg sum_copy, const Reg diff_copy,
					const Reg a, const Reg b) const override
	{
		_gpu->addsub_copy((size_t)sum, (size_t)diff, (size_t)sum_copy, (size_t)diff_copy, (size_t)a, (size_t)b);
	}
	void mul_pair_prepared(const Reg rdst0, const Reg rsrc0,
						const Reg rdst1, const Reg rsrc1,
						const uint32 a0 = 1, const uint32 a1 = 1) const override
	{
		if ((a0 != 1) || (a1 != 1))
		{
			mul(rdst0, rsrc0, a0);
			mul(rdst1, rsrc1, a1);
			return;
		}

		mul(rdst0, rsrc0);
		mul(rdst1, rsrc1);
	}

	void xdbl_tail_uv(const Reg x_out, const Reg z_out,
					const Reg u_work, const Reg v_reg,
					const Reg a24_mul,
					const Reg tmp_e_mul, const Reg tmp_v_mul) const override
	{
		sub_reg(u_work, v_reg);
		set_multiplicand(tmp_e_mul, u_work);
		set_multiplicand(tmp_v_mul, v_reg);
		mul_pair_prepared(x_out, tmp_v_mul, u_work, a24_mul);
		add(u_work, v_reg);
		mul_copy(u_work, tmp_e_mul, z_out);
	}
	size_t get_register_data_size() const override { return _n * sizeof(uint64); }

	bool get_data(std::vector<char> & data, const Reg src) const override
	{
		if (data.size() != get_register_data_size()) return false;
		_gpu->read_reg(reinterpret_cast<uint64 *>(data.data()), size_t(src));
		return true;
	}

	bool set_data(const Reg dst, const std::vector<char> & data) const override
	{
		if (data.size() != get_register_data_size()) return false;
		_gpu->write_reg(reinterpret_cast<const uint64 *>(data.data()), size_t(dst));
		return true;
	}

	size_t get_checkpoint_size() const override { return _reg_count * _n * sizeof(uint64); }

	bool get_checkpoint(std::vector<char> & data) const override
	{
		if (data.size() != get_checkpoint_size()) return false;
		_gpu->read_regs(reinterpret_cast<uint64 *>(data.data()));
		return true;
	}

	bool set_checkpoint(const std::vector<char> & data) const override
	{
		if (data.size() != get_checkpoint_size()) return false;
		_gpu->write_regs(reinterpret_cast<const uint64 *>(data.data()));
		return true;
	}
};


// v83: engine wrapper. Default path always constructs engine_gpu_flat; the gpu
// layer itself now performs GPU-only segmented-regspace when the register slab
// exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE.  The old host-backed paged wrapper is
// kept only behind PRMERS_MARIN_HOST_PAGED_FORCE=1 for comparison/debug.
class engine_gpu : public engine
{
private:
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    const size_t _logical_reg_count;
    const size_t _n;
    const size_t _reg_bytes;
    bool _paged = false;
    size_t _physical_reg_count = 0;
    mutable std::unique_ptr<engine_gpu_flat> _flat;

    mutable std::vector<std::vector<char>> _backing;      // raw weighted register images, one logical reg each
    mutable std::vector<uint8_t> _valid;                  // backing contains a valid logical register
    mutable std::vector<size_t> _logical_to_slot;
    mutable std::vector<size_t> _slot_to_logical;
    mutable std::vector<uint8_t> _slot_dirty;
    mutable std::vector<uint64_t> _slot_clock;
    mutable uint64_t _clock = 1;

    static cl_ulong query_max_alloc(const ocl::platform& platform, const size_t device)
    {
        cl_ulong v = 0;
        (void)clGetDeviceInfo(platform.get_device(device), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(v), &v, nullptr);
        return v;
    }

    static cl_ulong query_global_mem(const ocl::platform& platform, const size_t device)
    {
        cl_ulong v = 0;
        (void)clGetDeviceInfo(platform.get_device(device), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(v), &v, nullptr);
        return v;
    }

    static bool env_enabled(const char* name)
    {
        const char* v = std::getenv(name);
        return v != nullptr && std::string(v) != "0";
    }

    static size_t env_size(const char* name, const size_t def)
    {
        const char* v = std::getenv(name);
        if (v == nullptr || *v == '\0') return def;
        const unsigned long long x = std::strtoull(v, nullptr, 10);
        return x == 0 ? def : size_t(x);
    }

    bool is_protected_slot(const size_t slot, const std::vector<size_t>& protect) const
    {
        for (size_t p : protect) if (p == slot) return true;
        return false;
    }

    void ensure_backing(const size_t logical) const
    {
        if (_backing[logical].size() != _reg_bytes) _backing[logical].assign(_reg_bytes, 0);
    }

    void save_slot(const size_t slot) const
    {
        if (!_paged) return;
        const size_t logical = _slot_to_logical[slot];
        if (logical == npos || !_slot_dirty[slot]) return;
        ensure_backing(logical);
        if (!_flat->get_data(_backing[logical], (Reg)slot))
            throw std::runtime_error("paged-regspace: cannot read physical register during eviction");
        _valid[logical] = 1;
        _slot_dirty[slot] = 0;
    }

    size_t choose_slot(const std::vector<size_t>& protect) const
    {
        for (size_t s = 0; s < _physical_reg_count; ++s)
            if (_slot_to_logical[s] == npos && !is_protected_slot(s, protect)) return s;

        size_t best = npos;
        uint64_t best_clock = std::numeric_limits<uint64_t>::max();
        for (size_t s = 0; s < _physical_reg_count; ++s)
        {
            if (is_protected_slot(s, protect)) continue;
            if (_slot_clock[s] < best_clock) { best_clock = _slot_clock[s]; best = s; }
        }
        if (best == npos)
            throw std::runtime_error("paged-regspace: not enough physical slots for this multi-register kernel");
        return best;
    }

    size_t pin(const size_t logical, const bool load, const std::vector<size_t>& protect) const
    {
        if (!_paged) return logical;
        if (logical >= _logical_reg_count) throw std::runtime_error("paged-regspace: logical register out of range");

        size_t slot = _logical_to_slot[logical];
        if (slot != npos)
        {
            _slot_clock[slot] = ++_clock;
            return slot;
        }

        slot = choose_slot(protect);
        save_slot(slot);

        const size_t oldLogical = _slot_to_logical[slot];
        if (oldLogical != npos) _logical_to_slot[oldLogical] = npos;

        _slot_to_logical[slot] = logical;
        _logical_to_slot[logical] = slot;
        _slot_dirty[slot] = 0;
        _slot_clock[slot] = ++_clock;

        if (load)
        {
            if (_valid[logical])
            {
                ensure_backing(logical);
                if (!_flat->set_data((Reg)slot, _backing[logical]))
                    throw std::runtime_error("paged-regspace: cannot upload backing register");
            }
            else
            {
                _flat->set((Reg)slot, uint32(0));
            }
        }
        return slot;
    }

    size_t pin_read(const Reg logical, std::vector<size_t>& protect) const
    {
        const size_t s = pin(size_t(logical), true, protect);
        protect.push_back(s);
        return s;
    }

    size_t pin_rw(const Reg logical, std::vector<size_t>& protect) const
    {
        const size_t s = pin(size_t(logical), true, protect);
        protect.push_back(s);
        return s;
    }

    size_t pin_overwrite(const Reg logical, std::vector<size_t>& protect) const
    {
        const size_t l = size_t(logical);
        if (_paged && _logical_to_slot[l] != npos)
        {
            const size_t s = _logical_to_slot[l];
            protect.push_back(s);
            _slot_clock[s] = ++_clock;
            return s;
        }
        const size_t s = pin(l, false, protect);
        protect.push_back(s);
        return s;
    }

    void mark_dirty(const Reg logical) const
    {
        if (!_paged) return;
        const size_t l = size_t(logical);
        const size_t s = _logical_to_slot[l];
        if (s == npos) throw std::runtime_error("paged-regspace: dirty logical register is not resident");
        _slot_dirty[s] = 1;
        _valid[l] = 1;
        _slot_clock[s] = ++_clock;
    }

    void flush_all() const
    {
        if (!_paged) return;
        _flat->sync();
        for (size_t s = 0; s < _physical_reg_count; ++s) save_slot(s);
    }

public:
    engine_gpu(const uint32_t q, const size_t reg_count, const size_t device, const bool verbose)
        : engine(), _logical_reg_count(reg_count), _n(ibdwt::transform_size(q)), _reg_bytes(_n * sizeof(uint64))
    {
        const ocl::platform platform;
        const cl_ulong maxAlloc = query_max_alloc(platform, device);
        const cl_ulong globalMem = query_global_mem(platform, device);
        const size_t logicalRegBytes = _logical_reg_count * _reg_bytes;
        const bool disablePaged = env_enabled("PRMERS_MARIN_PAGED_DISABLE");
        const long double triggerFrac = 0.90L;
        const bool exceedsSingleAlloc = (maxAlloc != 0 && (long double)logicalRegBytes > (long double)maxAlloc * triggerFrac);

        if (env_enabled("PRMERS_MARIN_HOST_PAGED_FORCE") && !disablePaged && exceedsSingleAlloc)
        {
            // Legacy v74-v82 host-backed paged regspace. Disabled by default in v83;
            // use PRMERS_MARIN_HOST_PAGED_FORCE=1 only for comparison.
            // v82: use a larger physical scratch by default.  v74-v81 used 0.70
            // of CL_DEVICE_MAX_MEM_ALLOC_SIZE, which left only 21 slots on RTX 3080
            // for a 10,485,760 transform.  That made batch+paged windows thrash.
            // 0.94 stays below the single-allocation limit while giving ~29 slots.
            size_t safeBytes = size_t((long double)maxAlloc * 0.94L);
            if (const char* envFrac = std::getenv("PRMERS_MARIN_PAGED_MAXALLOC_FRAC"))
            {
                const long double f = std::strtold(envFrac, nullptr);
                if (f > 0.10L && f < 0.985L) safeBytes = size_t((long double)maxAlloc * f);
            }
            size_t physical = safeBytes / std::max<size_t>(_reg_bytes, 1);
            const size_t minSlots = env_size("PRMERS_MARIN_PAGED_MIN_SLOTS", 8);
            const size_t maxSlots = env_size("PRMERS_MARIN_PAGED_MAX_SLOTS", 0);
            if (maxSlots != 0 && physical > maxSlots) physical = maxSlots;
            if (physical > _logical_reg_count) physical = _logical_reg_count;
            if (physical < minSlots)
            {
                std::ostringstream ss;
                ss << "paged-regspace requested but only " << physical
                   << " physical register slots fit below max single allocation; need at least " << minSlots
                   << ". Try smaller transform/exponent or set PRMERS_MARIN_PAGED_MIN_SLOTS.";
                throw std::runtime_error(ss.str());
            }

            _paged = true;
            _physical_reg_count = physical;
            if (verbose || env_enabled("PRMERS_GPU_ALLOC_DIAG"))
            {
                auto gib = [](long double b){ return b / 1073741824.0L; };
                std::cout << "[MARIN-PAGED] logical regs=" << _logical_reg_count
                          << " would allocate reg slab=" << std::fixed << std::setprecision(2) << gib((long double)logicalRegBytes)
                          << " GiB, above OpenCL max single allocation=" << gib((long double)maxAlloc) << " GiB.\n"
                          << "[MARIN-PAGED] using physical scratch regs=" << _physical_reg_count
                          << " (scratch slab=" << gib((long double)(_physical_reg_count * _reg_bytes))
                          << " GiB), kernels unchanged, logical registers backed by host RAM"
                          << " [scratch-frac default=0.94, override PRMERS_MARIN_PAGED_MAXALLOC_FRAC]";
                if (globalMem != 0) std::cout << ", device=" << gib((long double)globalMem) << " GiB";
                std::cout << ".\n";
            }
            _flat.reset(new engine_gpu_flat(q, _physical_reg_count, device, verbose));
            _backing.resize(_logical_reg_count);
            _valid.assign(_logical_reg_count, 0);
            _logical_to_slot.assign(_logical_reg_count, npos);
            _slot_to_logical.assign(_physical_reg_count, npos);
            _slot_dirty.assign(_physical_reg_count, 0);
            _slot_clock.assign(_physical_reg_count, 0);
        }
        else
        {
            _paged = false;
            _physical_reg_count = _logical_reg_count;
            _flat.reset(new engine_gpu_flat(q, _logical_reg_count, device, verbose));
        }
    }

    virtual ~engine_gpu() { release_gpu_resources_for_lowmem_handoff(); }

    void release_gpu_resources_for_lowmem_handoff() override
    {
        if (_flat) { flush_all(); _flat->release_gpu_resources_for_lowmem_handoff(); _flat.reset(); }
        std::vector<std::vector<char>>().swap(_backing);
        std::vector<uint8_t>().swap(_valid);
        std::vector<size_t>().swap(_logical_to_slot);
        std::vector<size_t>().swap(_slot_to_logical);
        std::vector<uint8_t>().swap(_slot_dirty);
        std::vector<uint64_t>().swap(_slot_clock);
    }

    size_t get_size() const override { return _n; }
    void sync() const override { if (_flat) _flat->sync(); }

    void set(const Reg dst, const uint32 a) const override
    {
        if (!_paged) { _flat->set(dst, a); return; }
        std::vector<size_t> p; const size_t pd = pin_overwrite(dst, p);
        _flat->set((Reg)pd, a); mark_dirty(dst);
    }

    void set(const Reg dst, uint64 * const d) const override
    {
        if (!_paged) { _flat->set(dst, d); return; }
        std::vector<size_t> p; const size_t pd = pin_overwrite(dst, p);
        _flat->set((Reg)pd, d); mark_dirty(dst);
    }

    void set_mpz(const Reg dst, const mpz_t & z) const override
    {
        if (!_paged) { _flat->set_mpz(dst, z); return; }
        std::vector<size_t> p; const size_t pd = pin_overwrite(dst, p);
        _flat->set_mpz((Reg)pd, z); mark_dirty(dst);
    }

    void get(uint64 * const d, const Reg src) const override
    {
        if (!_paged) { _flat->get(d, src); return; }
        std::vector<size_t> p; const size_t ps = pin_read(src, p);
        _flat->get(d, (Reg)ps);
    }

    void copy(const Reg dst, const Reg src) const override
    {
        if (!_paged) { _flat->copy(dst, src); return; }
        if (dst == src) return;
        std::vector<size_t> p; const size_t ps = pin_read(src, p); const size_t pd = pin_overwrite(dst, p);
        _flat->copy((Reg)pd, (Reg)ps); mark_dirty(dst);
    }

    void square_mul(const Reg src, const uint32 a = 1) const override
    {
        if (!_paged) { _flat->square_mul(src, a); return; }
        std::vector<size_t> p; const size_t ps = pin_rw(src, p);
        _flat->square_mul((Reg)ps, a); mark_dirty(src);
    }

    void set_multiplicand(const Reg dst, const Reg src) const override
    {
        if (!_paged) { _flat->set_multiplicand(dst, src); return; }
        std::vector<size_t> p; const size_t ps = pin_read(src, p); const size_t pd = (dst == src) ? ps : pin_overwrite(dst, p);
        _flat->set_multiplicand((Reg)pd, (Reg)ps); mark_dirty(dst);
    }

    void set_multiplicand2(const Reg dst, const Reg src) const override
    {
        set_multiplicand(dst, src);
    }

    void mul(const Reg dst, const Reg src, const uint32 a = 1) const override
    {
        if (!_paged) { _flat->mul(dst, src, a); return; }
        std::vector<size_t> p; const size_t pd = pin_rw(dst, p); const size_t ps = pin_read(src, p);
        _flat->mul((Reg)pd, (Reg)ps, a); mark_dirty(dst);
    }

    void mul_new(const Reg dst, const Reg src, const uint32 a = 1) const override
    {
        if (!_paged) { _flat->mul_new(dst, src, a); return; }
        mul(dst, src, a);
    }

    void mul_add(const Reg dst, const Reg mul_src, const Reg add_src, const uint32 a = 1) const override
    {
        if (!_paged) { _flat->mul_add(dst, mul_src, add_src, a); return; }
        std::vector<size_t> p; const size_t pd = pin_rw(dst, p); const size_t pm = pin_read(mul_src, p); const size_t pa = pin_read(add_src, p);
        _flat->mul_add((Reg)pd, (Reg)pm, (Reg)pa, a); mark_dirty(dst);
    }

    void square_mul_copy(const Reg src, const Reg dst_copy, const uint32 a = 1) const override
    {
        if (!_paged) { _flat->square_mul_copy(src, dst_copy, a); return; }
        std::vector<size_t> p; const size_t ps = pin_rw(src, p); const size_t pc = (dst_copy == src) ? ps : pin_overwrite(dst_copy, p);
        _flat->square_mul_copy((Reg)ps, (Reg)pc, a); mark_dirty(src); mark_dirty(dst_copy);
    }

    void mul_copy(const Reg dst, const Reg src, const Reg dst_copy, const uint32 a = 1) const override
    {
        if (!_paged) { _flat->mul_copy(dst, src, dst_copy, a); return; }
        std::vector<size_t> p; const size_t pd = pin_rw(dst, p); const size_t ps = pin_read(src, p); const size_t pc = (dst_copy == dst) ? pd : pin_overwrite(dst_copy, p);
        _flat->mul_copy((Reg)pd, (Reg)ps, (Reg)pc, a); mark_dirty(dst); mark_dirty(dst_copy);
    }

    void sub(const Reg src, const uint32 a) const override
    {
        if (!_paged) { _flat->sub(src, a); return; }
        std::vector<size_t> p; const size_t ps = pin_rw(src, p);
        _flat->sub((Reg)ps, a); mark_dirty(src);
    }

    void add(const Reg dst, const Reg src) const override
    {
        if (!_paged) { _flat->add(dst, src); return; }
        std::vector<size_t> p; const size_t pd = pin_rw(dst, p); const size_t ps = pin_read(src, p);
        _flat->add((Reg)pd, (Reg)ps); mark_dirty(dst);
    }

    void sub_reg(const Reg dst, const Reg src) const override
    {
        if (!_paged) { _flat->sub_reg(dst, src); return; }
        std::vector<size_t> p; const size_t pd = pin_rw(dst, p); const size_t ps = pin_read(src, p);
        _flat->sub_reg((Reg)pd, (Reg)ps); mark_dirty(dst);
    }

    void addsub(const Reg sum_out, const Reg diff_out, const Reg a, const Reg b) const override
    {
        if (!_paged) { _flat->addsub(sum_out, diff_out, a, b); return; }
        // Generic alias-safe sequence through the public virtual operations.
        engine::addsub(sum_out, diff_out, a, b);
    }

    void addsub_copy(const Reg sum, const Reg diff, const Reg sum_copy, const Reg diff_copy, const Reg a, const Reg b) const override
    {
        if (!_paged) { _flat->addsub_copy(sum, diff, sum_copy, diff_copy, a, b); return; }
        engine::addsub_copy(sum, diff, sum_copy, diff_copy, a, b);
    }

    void mul_pair_unit(const Reg dst0, const Reg src0, const Reg dst1, const Reg src1) const override
    {
        if (!_paged) { _flat->mul_pair_unit(dst0, src0, dst1, src1); return; }
        engine::mul_pair_unit(dst0, src0, dst1, src1);
    }

    void mul_pair_prepared(const Reg dst0, const Reg mul_src0, const Reg dst1, const Reg mul_src1, const uint32 a0 = 1, const uint32 a1 = 1) const override
    {
        if (!_paged) { _flat->mul_pair_prepared(dst0, mul_src0, dst1, mul_src1, a0, a1); return; }
        if ((a0 != 1) || (a1 != 1)) { mul(dst0, mul_src0, a0); mul(dst1, mul_src1, a1); return; }
        std::vector<size_t> p;
        const size_t pd0 = pin_rw(dst0, p); const size_t ps0 = pin_read(mul_src0, p);
        const size_t pd1 = pin_rw(dst1, p); const size_t ps1 = pin_read(mul_src1, p);
        _flat->mul_pair_prepared((Reg)pd0, (Reg)ps0, (Reg)pd1, (Reg)ps1, a0, a1);
        mark_dirty(dst0); mark_dirty(dst1);
    }

    void xdbl_tail_uv(const Reg x_out, const Reg z_out, const Reg u_work, const Reg v_reg,
                      const Reg a24_mul, const Reg tmp_e_mul, const Reg tmp_v_mul) const override
    {
        if (!_paged) { _flat->xdbl_tail_uv(x_out, z_out, u_work, v_reg, a24_mul, tmp_e_mul, tmp_v_mul); return; }
        engine::xdbl_tail_uv(x_out, z_out, u_work, v_reg, a24_mul, tmp_e_mul, tmp_v_mul);
    }

    size_t get_register_data_size() const override { return _reg_bytes; }

    bool get_data(std::vector<char> & data, const Reg src) const override
    {
        if (data.size() != _reg_bytes) return false;
        if (!_paged) return _flat->get_data(data, src);
        const size_t l = size_t(src);
        if (l >= _logical_reg_count) return false;
        const size_t s = _logical_to_slot[l];
        if (s != npos) return _flat->get_data(data, (Reg)s);
        if (_valid[l]) { ensure_backing(l); std::memcpy(data.data(), _backing[l].data(), _reg_bytes); return true; }
        std::fill(data.begin(), data.end(), 0); return true;
    }

    bool set_data(const Reg dst, const std::vector<char> & data) const override
    {
        if (data.size() != _reg_bytes) return false;
        if (!_paged) return _flat->set_data(dst, data);
        const size_t l = size_t(dst);
        if (l >= _logical_reg_count) return false;
        ensure_backing(l);
        std::memcpy(_backing[l].data(), data.data(), _reg_bytes);
        _valid[l] = 1;
        const size_t s = _logical_to_slot[l];
        if (s != npos)
        {
            if (!_flat->set_data((Reg)s, data)) return false;
            _slot_dirty[s] = 0;
        }
        return true;
    }

    size_t get_checkpoint_size() const override { return _logical_reg_count * _reg_bytes; }

    bool get_checkpoint(std::vector<char> & data) const override
    {
        if (data.size() != get_checkpoint_size()) return false;
        if (!_paged) return _flat->get_checkpoint(data);
        flush_all();
        for (size_t l = 0; l < _logical_reg_count; ++l)
        {
            char* out = data.data() + l * _reg_bytes;
            if (_valid[l]) { ensure_backing(l); std::memcpy(out, _backing[l].data(), _reg_bytes); }
            else std::memset(out, 0, _reg_bytes);
        }
        return true;
    }

    bool set_checkpoint(const std::vector<char> & data) const override
    {
        if (data.size() != get_checkpoint_size()) return false;
        if (!_paged) return _flat->set_checkpoint(data);
        for (size_t l = 0; l < _logical_reg_count; ++l)
        {
            ensure_backing(l);
            std::memcpy(_backing[l].data(), data.data() + l * _reg_bytes, _reg_bytes);
            _valid[l] = 1;
            _logical_to_slot[l] = npos;
        }
        std::fill(_slot_to_logical.begin(), _slot_to_logical.end(), npos);
        std::fill(_slot_dirty.begin(), _slot_dirty.end(), 0);
        std::fill(_slot_clock.begin(), _slot_clock.end(), 0);
        return true;
    }
};
