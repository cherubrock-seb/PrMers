/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "engine.h"
#include "ibdwt.h"
#include "ocl.h"

#include "ocl/kernel.h"

#define CREATE_KERNEL_TRANSFORM(name) _##name = create_kernel_transform(#name);
#define CREATE_KERNEL_CARRY(name) _##name = create_kernel_carry(#name);

#define DEFINE_FORWARDx2(u) void forward##u##x2(const size_t src, const uint32 lm) { ek_fb(_forward##u##x2, src, lm, (u / 4) * _chunk##u); }
#define DEFINE_BACKWARDx2(u) void backward##u##x2(const size_t src, const uint32 lm) { ek_fb(_backward##u##x2, src, lm, (u / 4) * _chunk##u); }

#define DEFINE_FORWARDx2_5(u) void forward##u##x2_5(const size_t src, const uint32 lm) { ek_fb(_forward##u##x2_5, src, lm, 5 * (u / 4) * _chunk##u##_5); }
#define DEFINE_BACKWARDx2_5(u) void backward##u##x2_5(const size_t src, const uint32 lm) { ek_fb(_backward##u##x2_5, src, lm, 5 * (u / 4) * _chunk##u##_5); }

#define DEFINE_FORWARD_MULx2(u) void forward_mul##u##x2(const size_t src) { ek_fms(_forward_mul##u##x2, 8, src, (u / 4) * _blk##u); }
#define DEFINE_SQRx2(u) void sqr##u##x2(const size_t src) { ek_fms(_sqr##u##x2, 8, src, (u / 4) * _blk##u); }
#define DEFINE_MULx2(u) void mul##u##x2(const size_t dst, const size_t src) { ek_mul(_mul##u##x2, 8, dst, src, (u / 4) * _blk##u); }

#define DEFINE_FORWARD_MUL5(u) void forward_mul##u(const size_t src) { ek_fms(_forward_mul##u, 8, src, (u / 8) * _blk##u); }
#define DEFINE_SQR5(u) void sqr##u(const size_t src) { ek_fms(_sqr##u, 8, src, (u / 8) * _blk##u); }
#define DEFINE_MUL5(u) void mul##u(const size_t dst, const size_t src) { ek_mul(_mul##u, 8, dst, src, (u / 8) * _blk##u); }

class gpu : public ocl::device
{
private:
	const size_t _n;
	const bool _even;
	const size_t _reg_count;
	const int _lcwm_wg_size, _lcwm_wg_size2;
	const size_t _blk16, _blk64, _blk40, _blk160, _blk640;
	const size_t _chunk16, _chunk64, _chunk256, _chunk16_5, _chunk64_5;
	static const size_t _blk4 = 0, _blk256 = 1, _blk1024 = 1, _blk2560 = 1, _chunk4 = 0, _chunk1024 = 1, _chunk4_5 = 0, _chunk256_5 = 1;

	// reg is the weighted representation of registers R0, R1, ...
	cl_mem _reg = nullptr, _carry = nullptr, _root = nullptr, _weight = nullptr, _digit_width = nullptr;

	cl_kernel _forward4x2 = nullptr, _backward4x2 = nullptr;
	cl_kernel _forward16x2 = nullptr, _backward16x2 = nullptr;
	cl_kernel _forward64x2 = nullptr, _backward64x2 = nullptr;
	cl_kernel _forward256x2 = nullptr, _backward256x2 = nullptr;
	cl_kernel _forward1024x2 = nullptr, _backward1024x2 = nullptr;

	cl_kernel _forward_mul4 = nullptr, _sqr4 = nullptr, _mul4 = nullptr;
	cl_kernel _forward_mul4x2 = nullptr, _sqr4x2 = nullptr, _mul4x2 = nullptr;
	cl_kernel _forward_mul16x2 = nullptr, _sqr16x2 = nullptr, _mul16x2 = nullptr;
	cl_kernel _forward_mul64x2 = nullptr, _sqr64x2 = nullptr, _mul64x2 = nullptr;
	cl_kernel _forward_mul256x2 = nullptr, _sqr256x2 = nullptr, _mul256x2 = nullptr;
	cl_kernel _forward_mul1024x2 = nullptr, _sqr1024x2 = nullptr, _mul1024x2 = nullptr;

	cl_kernel _forward4x2_5 = nullptr, _backward4x2_5 = nullptr;
	cl_kernel _forward16x2_5 = nullptr, _backward16x2_5 = nullptr;
	cl_kernel _forward64x2_5 = nullptr, _backward64x2_5 = nullptr;
	cl_kernel _forward256x2_5 = nullptr, _backward256x2_5 = nullptr;

	// cl_kernel _forward_mul10 = nullptr, _sqr10 = nullptr, _mul10 = nullptr;
	cl_kernel _forward_mul40 = nullptr, _sqr40 = nullptr, _mul40 = nullptr;
	cl_kernel _forward_mul160 = nullptr, _sqr160 = nullptr, _mul160 = nullptr;
	cl_kernel _forward_mul640 = nullptr, _sqr640 = nullptr, _mul640 = nullptr;
	cl_kernel _forward_mul2560 = nullptr, _sqr2560 = nullptr, _mul2560 = nullptr;

	cl_kernel _carry_weight_mul_p1 = nullptr, _carry_weight_mul_p2 = nullptr, _carry_weight_mul2_p1 = nullptr, _carry_weight_mul2_p2 = nullptr;
	cl_kernel _copy = nullptr;
	cl_kernel _subtract = nullptr, _subtract2 = nullptr;

	std::vector<cl_kernel> _kernels;

public:
	gpu(const ocl::platform & platform, const size_t d, const size_t n, const bool even, const size_t reg_count, const bool verbose)
		: device(platform, d, verbose), _n(n), _even(even), _reg_count(reg_count),
		_lcwm_wg_size(ilog2_32(uint32_t(std::min(((n % 5 == 0) ? n / 5 : n) / 4, get_max_local_worksize(sizeof(uint64)))))),
		_lcwm_wg_size2(ilog2_32(uint32_t(std::min(((n % 5 == 0) ? n / 5 : n) / 8, get_max_local_worksize(2 * sizeof(uint64)))))),

		// We must have (u / 4) * BLKu <= n / 8
		_blk16((n >= 512) ? 16 : 1),	// 16 * BLK16 uint64_2 <= 4KB, workgroup size = (16 / 4) * BLK16 <= 64
		_blk64((n >= 512) ? 4 : 1),		// 64 * BLK64 uint64_2 <= 4KB, workgroup size = (64 / 4) * BLK64 <= 64
		// 256 uint64_2 = 4KB, workgroup size = 256 / 4 = 64; 1024 uint64_2 = 16KB, workgroup size <= 1024 / 4 = 256

		// We must have (u / 8) * BLKu <= n / 8
		_blk40((n >= 1280) ? 32 : 1),	// 20 * BLK40 uint64_2 <= 10KB, workgroup size = (40 / 8) * BLK40 <= 160 = 5 * 32
		_blk160((n >= 1280) ? 8 : 1),	// 80 * BLK160 uint64_2 <= 10KB, workgroup size = (160 / 8) * BLK160 <= 160
		_blk640((n >= 1280) ? 2 : 1),	// 320 * BLK640 uint64_2 <= 10KB, workgroup size = (640 / 8) * BLK640 <= 160
		// 2560: 1280 uint64_2 = 20KB, workgroup size = 2560 / 8 = 320

		// We must have (u / 4) * CHUNKu <= n / 8 and CHUNKu < m
		_chunk16(std::min(std::max(n / 8 * 4 / 16, size_t(1)), size_t(16))),	// 16 * CHUNK16 uint64_2 <= 4KB, workgroup size = (16 / 4) * CHUNK16 <= 64
		_chunk64(std::min(std::max(n / 8 * 4 / 64, size_t(1)), size_t(4))),		// 64 * CHUNK64 uint64_2 <= 4KB, workgroup size = (64 / 4) * CHUNK64 <= 64
		_chunk256(std::min(std::max(n / 8 * 4 / 256, size_t(1)), size_t(4))),	// 256 * CHUNK256 uint64_2 <= 16KB, workgroup size = (256 / 4) * CHUNK256 <= 256
		// 1024: 1024 uint64_2 = 16KB, workgroup size = 1024 / 4 = 256

		// We must have 5 * (u / 4) * CHUNKu <= n / 8
		_chunk16_5(std::min(std::max(n / 8 / 5 * 4 / 16, size_t(1)), size_t(8))),	// 5 * 16 * CHUNK16_5 uint64_2 <= 10KB, workgroup size = 5 * (16 / 4) * CHUNK16_5 <= 160 = 5 * 32
		_chunk64_5(std::min(std::max(n / 8 / 5 * 4 / 46, size_t(1)), size_t(2)))	// 5 * 64 * CHUNK64_5 uint64_2 <= 10KB, workgroup size = 5 * (64 / 4) * CHUNK64_5 <= 160
		// 256: 5 * 256 uint64_2 = 20KB, workgroup size = 5 * (256 / 4) = 320
 	{}
	virtual ~gpu() {}

	int get_lcwm_wg_size() const { return _lcwm_wg_size; }
	int get_lcwm_wg_size2() const { return _lcwm_wg_size2; }
	size_t get_blk16() const { return _blk16; }
	size_t get_blk64() const { return _blk64; }
	size_t get_blk40() const { return _blk40; }
	size_t get_blk160() const { return _blk160; }
	size_t get_blk640() const { return _blk640; }
	size_t get_chunk16() const { return _chunk16; }
	size_t get_chunk64() const { return _chunk64; }
	size_t get_chunk256() const { return _chunk256; }
	size_t get_chunk16_5() const { return _chunk16_5; }
	size_t get_chunk64_5() const { return _chunk64_5; }

///////////////////////////////

	void alloc_memory()
	{
#if defined(ocl_debug)
		std::cout << "Alloc gpu memory." << std::endl;
#endif
		const size_t n = _n;
		if (n != 0)
		{
			_reg = _create_buffer(CL_MEM_READ_WRITE, _reg_count * n * sizeof(uint64));
			_carry = _create_buffer(CL_MEM_READ_WRITE, n / 4 * sizeof(uint64));
			_root = _create_buffer(CL_MEM_READ_ONLY, 2 * n * sizeof(uint64));
			_weight = _create_buffer(CL_MEM_READ_ONLY, 3 * n * sizeof(uint64));
			_digit_width = _create_buffer(CL_MEM_READ_ONLY, n * sizeof(uint8));
		}
	}

	void free_memory()
	{
#if defined(ocl_debug)
		std::cout << "Free gpu memory." << std::endl;
#endif
		if (_n != 0)
		{
			_release_buffer(_reg); _release_buffer(_carry);
			_release_buffer(_root); _release_buffer(_weight); _release_buffer(_digit_width);
		}
	}

///////////////////////////////

	cl_kernel create_kernel_transform(const char * const kernel_name)
	{
		cl_kernel kernel = _create_kernel(kernel_name);
		_set_kernel_arg(kernel, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(kernel, 1, sizeof(cl_mem), &_root);
		_kernels.push_back(kernel);
		return kernel;
	}

	cl_kernel create_kernel_carry(const char * const kernel_name)
	{
		cl_kernel kernel = _create_kernel(kernel_name);
		_set_kernel_arg(kernel, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(kernel, 1, sizeof(cl_mem), &_carry);
		_set_kernel_arg(kernel, 2, sizeof(cl_mem), &_weight);
		_set_kernel_arg(kernel, 3, sizeof(cl_mem), &_digit_width);
		_kernels.push_back(kernel);
		return kernel;
	}

	cl_kernel create_kernel_subtract(const char * const kernel_name)
	{
		cl_kernel kernel = _create_kernel(kernel_name);
		_set_kernel_arg(kernel, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(kernel, 1, sizeof(cl_mem), &_weight);
		_set_kernel_arg(kernel, 2, sizeof(cl_mem), &_digit_width);
		_kernels.push_back(kernel);
		return kernel;
	}

	void create_kernels()
	{
#if defined(ocl_debug)
		std::cout << "Create ocl kernels." << std::endl;
#endif
		const size_t n = _n;
		if (n % 5 != 0)
		{
			if (n <= 1024)
			{
				CREATE_KERNEL_TRANSFORM(forward4x2);
				CREATE_KERNEL_TRANSFORM(backward4x2);
			}
			if ((n == 4096) || (n == 8192))
			{
				CREATE_KERNEL_TRANSFORM(forward16x2);
				CREATE_KERNEL_TRANSFORM(backward16x2);
			}
			if (n >= 16384)
			{
				CREATE_KERNEL_TRANSFORM(forward64x2);
				CREATE_KERNEL_TRANSFORM(backward64x2);
			}
			if (n >= 65536)
			{
				CREATE_KERNEL_TRANSFORM(forward256x2);
				CREATE_KERNEL_TRANSFORM(backward256x2);
			}
			if (n >= 262144)
			{
				CREATE_KERNEL_TRANSFORM(forward1024x2);
				CREATE_KERNEL_TRANSFORM(backward1024x2);
			}

			if (n == 4)
			{
				CREATE_KERNEL_TRANSFORM(forward_mul4);
				CREATE_KERNEL_TRANSFORM(sqr4);
				CREATE_KERNEL_TRANSFORM(mul4);
			}
			else if ((n == 8) || (n == 16))
			{
				CREATE_KERNEL_TRANSFORM(forward_mul4x2);
				CREATE_KERNEL_TRANSFORM(sqr4x2);
				CREATE_KERNEL_TRANSFORM(mul4x2);
			}
			else if ((n == 32) || (n == 64))
			{
				CREATE_KERNEL_TRANSFORM(forward_mul16x2);
				CREATE_KERNEL_TRANSFORM(sqr16x2);
				CREATE_KERNEL_TRANSFORM(mul16x2);
			}
			else if ((n == 128) || (n == 256))
			{
				CREATE_KERNEL_TRANSFORM(forward_mul64x2);
				CREATE_KERNEL_TRANSFORM(sqr64x2);
				CREATE_KERNEL_TRANSFORM(mul64x2);
			}
			else
			{
				CREATE_KERNEL_TRANSFORM(forward_mul256x2);
				CREATE_KERNEL_TRANSFORM(sqr256x2);
				CREATE_KERNEL_TRANSFORM(mul256x2);
				CREATE_KERNEL_TRANSFORM(forward_mul1024x2);
				CREATE_KERNEL_TRANSFORM(sqr1024x2);
				CREATE_KERNEL_TRANSFORM(mul1024x2);
			}
		}
		else
		{
			if ((n == 2560) || (n == 5120))
			{
				CREATE_KERNEL_TRANSFORM(forward4x2_5);
				CREATE_KERNEL_TRANSFORM(backward4x2_5);
			}
			if (n >= 10240)
			{
				CREATE_KERNEL_TRANSFORM(forward16x2_5);
				CREATE_KERNEL_TRANSFORM(backward16x2_5);
				CREATE_KERNEL_TRANSFORM(forward64x2_5);
				CREATE_KERNEL_TRANSFORM(backward64x2_5);
				if (get_max_workgroup_size() >= 5 * (256 / 4))
				{
					CREATE_KERNEL_TRANSFORM(forward256x2_5);
					CREATE_KERNEL_TRANSFORM(backward256x2_5);
				}
			}
			// CREATE_KERNEL_TRANSFORM(forward_mul10);
			// CREATE_KERNEL_TRANSFORM(sqr10);
			// CREATE_KERNEL_TRANSFORM(mul10);
			if ((n == 40) || (n == 80))
			{
				CREATE_KERNEL_TRANSFORM(forward_mul40);
				CREATE_KERNEL_TRANSFORM(sqr40);
				CREATE_KERNEL_TRANSFORM(mul40);
			}
			else if ((n == 160) || (n == 320))
			{
				CREATE_KERNEL_TRANSFORM(forward_mul160);
				CREATE_KERNEL_TRANSFORM(sqr160);
				CREATE_KERNEL_TRANSFORM(mul160);
			}
			else
			{
				CREATE_KERNEL_TRANSFORM(forward_mul640);
				CREATE_KERNEL_TRANSFORM(sqr640);
				CREATE_KERNEL_TRANSFORM(mul640);
				if (get_max_workgroup_size() >= 2560 / 8)
				{
					CREATE_KERNEL_TRANSFORM(forward_mul2560);
					CREATE_KERNEL_TRANSFORM(sqr2560);
					CREATE_KERNEL_TRANSFORM(mul2560);
				}
			}
		}

		if (_even)
		{
			CREATE_KERNEL_CARRY(carry_weight_mul_p1);
			CREATE_KERNEL_CARRY(carry_weight_mul_p2);
		}
		else
		{
			CREATE_KERNEL_CARRY(carry_weight_mul2_p1);
			CREATE_KERNEL_CARRY(carry_weight_mul2_p2);
		}

		_copy = _create_kernel("copy");
		_set_kernel_arg(_copy, 0, sizeof(cl_mem), &_reg);
		_kernels.push_back(_copy);

		if (_even) _subtract = create_kernel_subtract("subtract");
		else _subtract2 = create_kernel_subtract("subtract2");
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

	void read_regs(uint64 * const ptr) { _read_buffer(_reg, ptr, _reg_count * _n * sizeof(uint64)); }
	void write_regs(const uint64 * const ptr) { _write_buffer(_reg, ptr, _reg_count * _n * sizeof(uint64)); }
	void read_reg(uint64 * const ptr, const size_t index) { _read_buffer(_reg, ptr, _n * sizeof(uint64), index * _n * sizeof(uint64)); }
	void write_reg(const uint64 * const ptr, const size_t index) { _write_buffer(_reg, ptr, _n * sizeof(uint64), index * _n * sizeof(uint64)); }

	void write_root(const uint64 * const ptr) { _write_buffer(_root, ptr, 2 * _n * sizeof(uint64)); }
	void write_weight(const uint64 * const ptr) { _write_buffer(_weight, ptr, 3 * _n * sizeof(uint64)); }
	void write_width(const uint8 * const ptr) { _write_buffer(_digit_width, ptr, _n * sizeof(uint8)); }

///////////////////////////////

	void ek_fb(cl_kernel & kernel, const size_t src, const uint32 lm, const size_t local_size = 0)
	{
		const uint32 offset = uint32(src * _n);
		_set_kernel_arg(kernel, 2, sizeof(uint32), &offset);
		_set_kernel_arg(kernel, 3, sizeof(uint32), &lm);
		_execute_kernel(kernel, _n / 8, local_size);
	}

	void ek_fms(cl_kernel & kernel, const size_t step, const size_t src, const size_t local_size = 0)
	{
		const uint32 offset = uint32(src * _n);
		_set_kernel_arg(kernel, 2, sizeof(uint32), &offset);
		_execute_kernel(kernel, _n / step, local_size);
	}

	void ek_mul(cl_kernel & kernel, const size_t step, const size_t dst, const size_t src, const size_t local_size = 0)
	{
		const uint32 offset_y = uint32(src * _n);
		_set_kernel_arg(kernel, 3, sizeof(uint32), &offset_y);
		ek_fms(kernel, step, dst, local_size);
	}

	void ek_cwm(cl_kernel & kernel1, cl_kernel & kernel2, const size_t step, const int lcwm_wg_size, const size_t src, const uint32 a)
	{
		const uint32 offset = uint32(src * _n);
		_set_kernel_arg(kernel1, 4, sizeof(uint32), &a);
		_set_kernel_arg(kernel1, 5, sizeof(uint32), &offset);
		_execute_kernel(kernel1, _n / step, 1u << lcwm_wg_size);
		_set_kernel_arg(kernel2, 4, sizeof(uint32), &offset);
		_execute_kernel(kernel2, (_n / step) >> lcwm_wg_size);
	}

	DEFINE_FORWARDx2(4);
	DEFINE_BACKWARDx2(4);
	DEFINE_FORWARDx2(16);
	DEFINE_BACKWARDx2(16);
	DEFINE_FORWARDx2(64);
	DEFINE_BACKWARDx2(64);
	DEFINE_FORWARDx2(256);
	DEFINE_BACKWARDx2(256);
	DEFINE_FORWARDx2(1024);
	DEFINE_BACKWARDx2(1024);

	DEFINE_FORWARDx2_5(4);
	DEFINE_BACKWARDx2_5(4);
	DEFINE_FORWARDx2_5(16);
	DEFINE_BACKWARDx2_5(16);
	DEFINE_FORWARDx2_5(64);
	DEFINE_BACKWARDx2_5(64);
	DEFINE_FORWARDx2_5(256);
	DEFINE_BACKWARDx2_5(256);

	void forward_mul4(const size_t src) { ek_fms(_forward_mul4, 4, src); }
	void sqr4(const size_t src) { ek_fms(_sqr4, 4, src); }
	void mul4(const size_t dst, const size_t src) { ek_mul(_mul4, 4, dst, src); }

	DEFINE_FORWARD_MULx2(4);
	DEFINE_SQRx2(4);
	DEFINE_MULx2(4);
	DEFINE_FORWARD_MULx2(16);
	DEFINE_SQRx2(16);
	DEFINE_MULx2(16);
	DEFINE_FORWARD_MULx2(64);
	DEFINE_SQRx2(64);
	DEFINE_MULx2(64);
	DEFINE_FORWARD_MULx2(256);
	DEFINE_SQRx2(256);
	DEFINE_MULx2(256);
	DEFINE_FORWARD_MULx2(1024);
	DEFINE_SQRx2(1024);
	DEFINE_MULx2(1024);

	// void forward_mul10(const size_t src) { ek_fms(_forward_mul10, 8, src); }
	// void sqr10(const size_t src) { ek_fms(_sqr10, 8, src); }
	// void mul10(const size_t dst, const size_t src) { ek_mul(_mul10, 8, dst, src); }

	DEFINE_FORWARD_MUL5(40);
	DEFINE_SQR5(40);
	DEFINE_MUL5(40);
	DEFINE_FORWARD_MUL5(160);
	DEFINE_SQR5(160);
	DEFINE_MUL5(160);
	DEFINE_FORWARD_MUL5(640);
	DEFINE_SQR5(640);
	DEFINE_MUL5(640);
	DEFINE_FORWARD_MUL5(2560);
	DEFINE_SQR5(2560);
	DEFINE_MUL5(2560);

	void carry_weight_mul(const size_t src, const uint32 a)
	{
		if (_even) ek_cwm(_carry_weight_mul_p1, _carry_weight_mul_p2, 4, _lcwm_wg_size, src, a);
		else ek_cwm(_carry_weight_mul2_p1, _carry_weight_mul2_p2, 8, _lcwm_wg_size2, src, a);
	}

	void copy(const size_t dst, const size_t src)
	{
		const uint32 offset_y = uint32(dst * _n), offset_x = uint32(src * _n);
		_set_kernel_arg(_copy, 1, sizeof(uint32), &offset_y);
		_set_kernel_arg(_copy, 2, sizeof(uint32), &offset_x);
		_execute_kernel(_copy, _n);
	}

	void ek_sub(cl_kernel & kernel, const size_t src, const uint32 a)
	{
		const uint32 offset = uint32(src * _n);
		_set_kernel_arg(kernel, 3, sizeof(uint32), &offset);
		_set_kernel_arg(kernel, 4, sizeof(uint32), &a);
		_execute_kernel(kernel, 1);
	}

	void subtract(const size_t src, const uint32 a)
	{
		if (_even) ek_sub(_subtract, src, a);
		else ek_sub(_subtract2, src, a);
	}
};

class engine_gpu : public engine
{
private:
	const size_t _reg_count;
	const size_t _n;
	const bool _even;
	gpu * _gpu;
	std::vector<uint64> _weight;
	std::vector<uint8> _digit_width;

public:
	engine_gpu(const uint32_t q, const size_t reg_count, const size_t device, const bool verbose) : engine(),
		_reg_count(reg_count), _n(ibdwt::transform_size(q)), _even(ibdwt::is_even(_n))
	{
		const size_t n = _n;

		const ocl::platform eng_platform = ocl::platform();
		_gpu = new gpu(eng_platform, device, n, _even, _reg_count, verbose);

		std::ostringstream src;
		src << "#define N_SZ\t" << n << "u" << std::endl;

		const uint64 K = mod_root_nth(5), K2 = mod_sqr(K), K3 = mod_mul(K, K2), K4 = mod_sqr(K2); \
		const uint64 cosu = mod_half(mod_add(K, K4)), isinu = mod_half(mod_sub(K, K4)); \
		const uint64 cos2u = mod_half(mod_add(K2, K3)), isin2u = mod_half(mod_sub(K2, K3)); \
		const uint64 F1 = mod_sub(mod_half(mod_add(cosu, cos2u)), 1), F2 = mod_half(mod_sub(cosu, cos2u)); \
		const uint64 F3 = mod_add(isinu, isin2u), F4 = isin2u, F5 = mod_sub(isinu, isin2u); \
		src << "#define W_F1\t" << F1 << "ul" << std::endl;
		src << "#define W_F2\t" << F2 << "ul" << std::endl;
		src << "#define W_F3\t" << F3 << "ul" << std::endl;
		src << "#define W_F4\t" << F4 << "ul" << std::endl;
		src << "#define W_F5\t" << F5 << "ul" << std::endl;

		src << "#define BLK16\t" << _gpu->get_blk16() << "u" << std::endl;
		src << "#define BLK64\t" << _gpu->get_blk64() << "u" << std::endl;
		src << "#define BLK40\t" << _gpu->get_blk40() << "u" << std::endl;
		src << "#define BLK160\t" << _gpu->get_blk160() << "u" << std::endl;
		src << "#define BLK640\t" << _gpu->get_blk640() << "u" << std::endl;

		src << "#define CHUNK16\t" << _gpu->get_chunk16() << "u" << std::endl;
		src << "#define CHUNK64\t" << _gpu->get_chunk64() << "u" << std::endl;
		src << "#define CHUNK256\t" << _gpu->get_chunk256() << "u" << std::endl;
		src << "#define CHUNK16_5\t" << _gpu->get_chunk16_5() << "u" << std::endl;
		src << "#define CHUNK64_5\t" << _gpu->get_chunk64_5() << "u" << std::endl;

		if (_even) src << "#define CWM_WG_SZ\t" << (1u << _gpu->get_lcwm_wg_size()) << "u" << std::endl;
		else       src << "#define CWM_WG_SZ2\t" << (1u << _gpu->get_lcwm_wg_size2()) << "u" << std::endl;

		src << "#define MAX_WG_SZ\t" << _gpu->get_max_workgroup_size() << std::endl << std::endl;

		if (!_gpu->read_OpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_gpu->load_program(src.str());
		_gpu->alloc_memory();
		_gpu->create_kernels();

		std::vector<uint64> root(2 * n);
		ibdwt::roots(n, root.data());
		_gpu->write_root(root.data());

		_weight.resize(3 * n);
		_digit_width.resize(n);
		ibdwt::weights_widths(n, q, _weight.data(), _digit_width.data());
		_gpu->write_weight(_weight.data());
		_gpu->write_width(_digit_width.data());
	}

	virtual ~engine_gpu()
	{
		_gpu->release_kernels();
		_gpu->free_memory();
		_gpu->clear_program();

		delete _gpu;
	}

	size_t get_size() const override { return _n; }

	void set(const Reg dst, const uint64 a) const override
	{
		const size_t n = _n;
		std::vector<uint64> x(n);

		x[0] = a;	// weight[0] = 1
		for (size_t k = 1; k < n; ++k) x[k] = 0;

		// radix-2
		if (!_even) x[n / 2] = x[0];

		_gpu->write_reg(x.data(), size_t(dst));
	}

	void get(uint64 * const d, const Reg src) const override
	{
		const size_t n = _n;
		const uint64 * const wi = &_weight.data()[2 * n];
		const uint8 * const width = &_digit_width.data()[0];

		_gpu->read_reg(d, size_t(src));

		if (!_even)
		{
			// inverse radix-2
			for (size_t k = 0; k < n / 2; ++k)
			{
				const uint64 u0 = d[k + 0 * n / 2], u1 = d[k + 1 * n / 2];
				const uint64 v0 = mod_half(mod_add(u0, u1)), v1 = mod_half(mod_sub(u0, u1));
				d[k + 0 * n / 2] = v0; d[k + 1 * n / 2] = v1;
			}
		}

		// unweight, carry (strong)
		uint64 c = 0;
		for (size_t k = 0; k < n; ++k) d[k] = adc(mod_mul(d[k], wi[k]), width[k], c);

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

	bool is_equal(const Reg src1, const Reg src2) const override
	{
		const size_t n = _n;
		std::vector<uint64> x(n), y(n);

		_gpu->read_reg(x.data(), size_t(src1));
		_gpu->read_reg(y.data(), size_t(src2));

		for (size_t k = 0; k < n; ++k) if (y[k] != x[k]) return false;
		return true;
	}

	void set_multiplicand(const Reg rdst, const Reg rsrc) const override
	{
		if (rsrc != rdst) copy(rdst, rsrc);

		const size_t n = _n, dst = size_t(rdst), wg_size = _gpu->get_max_workgroup_size();
		switch (n)
		{
			case 1u <<  2: _gpu->forward_mul4(dst); break;
			case 1u <<  3: _gpu->forward_mul4x2(dst); break;
			case 1u <<  4: _gpu->forward4x2(dst, 1); _gpu->forward_mul4x2(dst); break;
			case 1u <<  5: _gpu->forward_mul16x2(dst); break;
			case 1u <<  6: _gpu->forward4x2(dst, 3); _gpu->forward_mul16x2(dst); break;
			case 1u <<  7: _gpu->forward_mul64x2(dst); break;
			case 1u <<  8: _gpu->forward4x2(dst, 5); _gpu->forward_mul64x2(dst); break;
			case 1u <<  9: _gpu->forward_mul256x2(dst); break;
			case 1u << 10: _gpu->forward4x2(dst, 7); _gpu->forward_mul256x2(dst); break;
			case 1u << 11: _gpu->forward_mul1024x2(dst); break;
			case 1u << 12:
			case 1u << 13: _gpu->forward16x2(dst, 7); _gpu->forward_mul256x2(dst); break;
			case 1u << 14:
			case 1u << 15: _gpu->forward64x2(dst, 7); _gpu->forward_mul256x2(dst); break;
			case 1u << 16:
			case 1u << 17: _gpu->forward256x2(dst, 7); _gpu->forward_mul256x2(dst); break;
			case 1u << 18:
			case 1u << 19: _gpu->forward256x2(dst, 9); _gpu->forward_mul1024x2(dst); break;
			case 1u << 20:
			case 1u << 21: _gpu->forward1024x2(dst, 9); _gpu->forward_mul1024x2(dst); break;
			case 1u << 22:
			case 1u << 23: _gpu->forward64x2(dst, 15); _gpu->forward256x2(dst, 7); _gpu->forward_mul256x2(dst); break;
			case 1u << 24:
			case 1u << 25: _gpu->forward256x2(dst, 15); _gpu->forward256x2(dst, 7); _gpu->forward_mul256x2(dst); break;
			case 1u << 26: _gpu->forward256x2(dst, 17); _gpu->forward256x2(dst, 9); _gpu->forward_mul1024x2(dst); break;

			case 5u <<  3:
			case 5u <<  4: _gpu->forward_mul40(dst); break;
			case 5u <<  5:
			case 5u <<  6: _gpu->forward_mul160(dst); break;
			case 5u <<  7:
			case 5u <<  8: _gpu->forward_mul640(dst); break;
			case 5u <<  9:
			case 5u << 10: if (wg_size >= 2560 / 8) _gpu->forward_mul2560(dst);
							else { _gpu->forward4x2_5(dst, 6); _gpu->forward_mul640(dst); }
							break;
			case 5u << 11:
			case 5u << 12: _gpu->forward16x2_5(dst, 6); _gpu->forward_mul640(dst); break;
			case 5u << 13:
			case 5u << 14: _gpu->forward64x2_5(dst, 6); _gpu->forward_mul640(dst); break;
			case 5u << 15:
			case 5u << 16: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(dst, 8); _gpu->forward_mul2560(dst); }
							else { _gpu->forward16x2_5(dst, 10); _gpu->forward16x2_5(dst, 6); _gpu->forward_mul640(dst);}
							break;
			case 5u << 17:
			case 5u << 18: if (wg_size >= 2560 / 8) { _gpu->forward256x2_5(dst, 8); _gpu->forward_mul2560(dst); }
							else { _gpu->forward16x2_5(dst, 12); _gpu->forward64x2_5(dst, 6); _gpu->forward_mul640(dst);}
							break;
			case 5u << 19:
			case 5u << 20: _gpu->forward64x2_5(dst, 12); _gpu->forward64x2_5(dst, 6); _gpu->forward_mul640(dst); break;
			case 5u << 21:
			case 5u << 22: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(dst, 14); _gpu->forward64x2_5(dst, 8); _gpu->forward_mul2560(dst); }
							else { _gpu->forward16x2_5(dst, 16); _gpu->forward16x2_5(dst, 12); _gpu->forward64x2_5(dst, 6); _gpu->forward_mul640(dst); }
							break;
			case 5u << 23:
			case 5u << 24: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(dst, 16); _gpu->forward256x2_5(dst, 8); _gpu->forward_mul2560(dst); }
							else { _gpu->forward16x2_5(dst, 18); _gpu->forward64x2_5(dst, 12); _gpu->forward64x2_5(dst, 6); _gpu->forward_mul640(dst); }
							break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}
	}

	void square_mul(const Reg rsrc, const uint32 a = 1) const override
	{
		const size_t n = _n, src = size_t(rsrc), wg_size = _gpu->get_max_workgroup_size();
		switch (n)
		{
			case 1u <<  2: _gpu->sqr4(src); break;
			case 1u <<  3: _gpu->sqr4x2(src); break;
			case 1u <<  4: _gpu->forward4x2(src, 1); _gpu->sqr4x2(src); _gpu->backward4x2(src, 1); break;
			case 1u <<  5: _gpu->sqr16x2(src); break;
			case 1u <<  6: _gpu->forward4x2(src, 3); _gpu->sqr16x2(src); _gpu->backward4x2(src, 3); break;
			case 1u <<  7: _gpu->sqr64x2(src); break;
			case 1u <<  8: _gpu->forward4x2(src, 5); _gpu->sqr64x2(src); _gpu->backward4x2(src, 5); break;
			case 1u <<  9: _gpu->sqr256x2(src); break;
			case 1u << 10: _gpu->forward4x2(src, 7); _gpu->sqr256x2(src); _gpu->backward4x2(src, 7); break;
			case 1u << 11: _gpu->sqr1024x2(src); break;
			case 1u << 12:
			case 1u << 13: _gpu->forward16x2(src, 7); _gpu->sqr256x2(src); _gpu->backward16x2(src, 7); break;
			case 1u << 14:
			case 1u << 15: _gpu->forward64x2(src, 7); _gpu->sqr256x2(src); _gpu->backward64x2(src, 7); break;
			case 1u << 16:
			case 1u << 17: _gpu->forward256x2(src, 7); _gpu->sqr256x2(src); _gpu->backward256x2(src, 7); break;
			case 1u << 18:
			case 1u << 19: _gpu->forward256x2(src, 9); _gpu->sqr1024x2(src); _gpu->backward256x2(src, 9); break;
			case 1u << 20:
			case 1u << 21: _gpu->forward1024x2(src, 9); _gpu->sqr1024x2(src); _gpu->backward1024x2(src, 9); break;
			case 1u << 22:
			case 1u << 23: _gpu->forward64x2(src, 15); _gpu->forward256x2(src, 7); _gpu->sqr256x2(src); _gpu->backward256x2(src, 7); _gpu->backward64x2(src, 15); break;
			case 1u << 24:
			case 1u << 25: _gpu->forward256x2(src, 15); _gpu->forward256x2(src, 7); _gpu->sqr256x2(src); _gpu->backward256x2(src, 7); _gpu->backward256x2(src, 15); break;
			case 1u << 26: _gpu->forward256x2(src, 17); _gpu->forward256x2(src, 9); _gpu->sqr1024x2(src); _gpu->backward256x2(src, 9); _gpu->backward256x2(src, 17); break;

			case 5u <<  3:
			case 5u <<  4: _gpu->sqr40(src); break;
			case 5u <<  5:
			case 5u <<  6: _gpu->sqr160(src); break;
			case 5u <<  7:
			case 5u <<  8: _gpu->sqr640(src); break;
			case 5u <<  9:
			case 5u << 10: if (wg_size >= 2560 / 8) _gpu->sqr2560(src); else { _gpu->forward4x2_5(src, 6); _gpu->sqr640(src); _gpu->backward4x2_5(src, 6); } break;
			case 5u << 11:
			case 5u << 12: _gpu->forward16x2_5(src, 6); _gpu->sqr640(src); _gpu->backward16x2_5(src, 6); break;
			case 5u << 13:
			case 5u << 14: _gpu->forward64x2_5(src, 6); _gpu->sqr640(src); _gpu->backward64x2_5(src, 6); break;
			case 5u << 15:
			case 5u << 16: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(src, 8); _gpu->sqr2560(src); _gpu->backward64x2_5(src, 8); }
							else { _gpu->forward16x2_5(src, 10); _gpu->forward16x2_5(src, 6); _gpu->sqr640(src); _gpu->backward16x2_5(src, 6); _gpu->backward16x2_5(src, 10); }
							break;
			case 5u << 17:
			case 5u << 18: if (wg_size >= 2560 / 8) { _gpu->forward256x2_5(src, 8); _gpu->sqr2560(src); _gpu->backward256x2_5(src, 8); }
							else { _gpu->forward16x2_5(src, 12); _gpu->forward64x2_5(src, 6); _gpu->sqr640(src); _gpu->backward64x2_5(src, 6); _gpu->backward16x2_5(src, 12); }
							break;
			case 5u << 19:
			case 5u << 20: _gpu->forward64x2_5(src, 12); _gpu->forward64x2_5(src, 6); _gpu->sqr640(src); _gpu->backward64x2_5(src, 6); _gpu->backward64x2_5(src, 12); break;
			case 5u << 21:
			case 5u << 22: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(src, 14); _gpu->forward64x2_5(src, 8); _gpu->sqr2560(src); _gpu->backward64x2_5(src, 8); _gpu->backward64x2_5(src, 14); }
							else { _gpu->forward16x2_5(src, 16); _gpu->forward16x2_5(src, 12); _gpu->forward64x2_5(src, 6); _gpu->sqr640(src); _gpu->backward64x2_5(src, 6); _gpu->backward16x2_5(src, 12); _gpu->backward16x2_5(src, 16); }
							break;
			case 5u << 23:
			case 5u << 24: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(src, 16); _gpu->forward256x2_5(src, 8); _gpu->sqr2560(src); _gpu->backward256x2_5(src, 8); _gpu->backward64x2_5(src, 16); }
							else { _gpu->forward16x2_5(src, 18); _gpu->forward64x2_5(src, 12); _gpu->forward64x2_5(src, 6); _gpu->sqr640(src); _gpu->backward64x2_5(src, 6); _gpu->backward64x2_5(src, 12); _gpu->backward16x2_5(src, 18); }
							break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}

		_gpu->carry_weight_mul(src, a);
	}

	void mul(const Reg rdst, const Reg rsrc) const override
	{
		const size_t n = _n, dst = size_t(rdst), src = size_t(rsrc), wg_size = _gpu->get_max_workgroup_size();
		switch (n)
		{
			case 1u <<  2: _gpu->mul4(dst, src); break;
			case 1u <<  3: _gpu->mul4x2(dst, src); break;
			case 1u <<  4: _gpu->forward4x2(dst, 1); _gpu->mul4x2(dst, src); _gpu->backward4x2(dst, 1); break;
			case 1u <<  5: _gpu->mul16x2(dst, src); break;
			case 1u <<  6: _gpu->forward4x2(dst, 3); _gpu->mul16x2(dst, src); _gpu->backward4x2(dst, 3); break;
			case 1u <<  7: _gpu->mul64x2(dst, src); break;
			case 1u <<  8: _gpu->forward4x2(dst, 5); _gpu->mul64x2(dst, src); _gpu->backward4x2(dst, 5); break;
			case 1u <<  9: _gpu->mul256x2(dst, src); break;
			case 1u << 10: _gpu->forward4x2(dst, 7); _gpu->mul256x2(dst, src); _gpu->backward4x2(dst, 7); break;
			case 1u << 11: _gpu->mul1024x2(dst, src); break;
			case 1u << 12:
			case 1u << 13: _gpu->forward16x2(dst, 7); _gpu->mul256x2(dst, src); _gpu->backward16x2(dst, 7); break;
			case 1u << 14:
			case 1u << 15: _gpu->forward64x2(dst, 7); _gpu->mul256x2(dst, src); _gpu->backward64x2(dst, 7); break;
			case 1u << 16:
			case 1u << 17: _gpu->forward256x2(dst, 7); _gpu->mul256x2(dst, src); _gpu->backward256x2(dst, 7); break;
			case 1u << 18:
			case 1u << 19: _gpu->forward256x2(dst, 9); _gpu->mul1024x2(dst, src); _gpu->backward256x2(dst, 9); break;
			case 1u << 20:
			case 1u << 21: _gpu->forward1024x2(dst, 9); _gpu->mul1024x2(dst, src); _gpu->backward1024x2(dst, 9); break;
			case 1u << 22:
			case 1u << 23: _gpu->forward64x2(dst, 15); _gpu->forward256x2(dst, 7); _gpu->mul256x2(dst, src); _gpu->backward256x2(dst, 7); _gpu->backward64x2(dst, 15); break;
			case 1u << 24:
			case 1u << 25: _gpu->forward256x2(dst, 15); _gpu->forward256x2(dst, 7); _gpu->mul256x2(dst, src); _gpu->backward256x2(dst, 7); _gpu->backward256x2(dst, 15); break;
			case 1u << 26: _gpu->forward256x2(dst, 17); _gpu->forward256x2(dst, 9); _gpu->mul1024x2(dst, src); _gpu->backward256x2(dst, 9); _gpu->backward256x2(dst, 17); break;

			case 5u <<  3:
			case 5u <<  4: _gpu->mul40(dst, src); break;
			case 5u <<  5:
			case 5u <<  6: _gpu->mul160(dst, src); break;
			case 5u <<  7:
			case 5u <<  8: _gpu->mul640(dst, src); break;
			case 5u <<  9:
			case 5u << 10: if (wg_size >= 2560 / 8) _gpu->mul2560(dst, src); else { _gpu->forward4x2_5(dst, 6); _gpu->mul640(dst, src); _gpu->backward4x2_5(dst, 6); } break;
			case 5u << 11:
			case 5u << 12: _gpu->forward16x2_5(dst, 6); _gpu->mul640(dst, src); _gpu->backward16x2_5(dst, 6); break;
			case 5u << 13:
			case 5u << 14: _gpu->forward64x2_5(dst, 6); _gpu->mul640(dst, src); _gpu->backward64x2_5(dst, 6); break;
			case 5u << 15:
			case 5u << 16: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(dst, 8); _gpu->mul2560(dst, src); _gpu->backward64x2_5(dst, 8); }
							else { _gpu->forward16x2_5(dst, 10); _gpu->forward16x2_5(dst, 6); _gpu->mul640(dst, src); _gpu->backward16x2_5(dst, 6); _gpu->backward16x2_5(dst, 10); }
							break;
			case 5u << 17:
			case 5u << 18: if (wg_size >= 2560 / 8) { _gpu->forward256x2_5(dst, 8); _gpu->mul2560(dst, src); _gpu->backward256x2_5(dst, 8); }
							else { _gpu->forward16x2_5(dst, 12); _gpu->forward64x2_5(dst, 6); _gpu->mul640(dst, src); _gpu->backward64x2_5(dst, 6); _gpu->backward16x2_5(dst, 12); }
							break;
			case 5u << 19:
			case 5u << 20: _gpu->forward64x2_5(dst, 12); _gpu->forward64x2_5(dst, 6); _gpu->mul640(dst, src); _gpu->backward64x2_5(dst, 6); _gpu->backward64x2_5(dst, 12); break;
			case 5u << 21:
			case 5u << 22: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(dst, 14); _gpu->forward64x2_5(dst, 8); _gpu->mul2560(dst, src); _gpu->backward64x2_5(dst, 8); _gpu->backward64x2_5(dst, 14); }
							else { _gpu->forward16x2_5(dst, 16); _gpu->forward16x2_5(dst, 12); _gpu->forward64x2_5(dst, 6); _gpu->mul640(dst, src); _gpu->backward64x2_5(dst, 6); _gpu->backward16x2_5(dst, 12); _gpu->backward16x2_5(dst, 16); }
							break;
			case 5u << 23:
			case 5u << 24: if (wg_size >= 2560 / 8) { _gpu->forward64x2_5(dst, 16); _gpu->forward256x2_5(dst, 8); _gpu->mul2560(dst, src); _gpu->backward256x2_5(dst, 8); _gpu->backward64x2_5(dst, 16); }
							else { _gpu->forward16x2_5(dst, 18); _gpu->forward64x2_5(dst, 12); _gpu->forward64x2_5(dst, 6); _gpu->mul640(dst, src); _gpu->backward64x2_5(dst, 6); _gpu->backward64x2_5(dst, 12); _gpu->backward16x2_5(dst, 18); }
							break;

			default: throw std::runtime_error("An unexpected error has occurred.");
		}

		_gpu->carry_weight_mul(dst, 1);
	}

	void sub(const Reg src, const uint32 a) const override { _gpu->subtract(size_t(src), a); }

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