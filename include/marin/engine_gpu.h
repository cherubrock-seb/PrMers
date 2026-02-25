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
	const int _lcwm_wg_size;
	const size_t _blk16, _blk32, _blk64, _blk128, _blk256, _blk512;
	const size_t _chunk16, _chunk20, _chunk64, _chunk80, _chunk256, _chunk320;
	static const size_t _blk4 = 0, _blk8 = 0, _blk1024 = 1, _blk2048 = 1, _chunk4 = 0, _chunk5 = 0, _chunk1024 = 1, _chunk1280 = 1;

	// reg is the weighted representation of registers R0, R1, ...
	cl_mem _reg = nullptr, _carry = nullptr, _root = nullptr, _weight = nullptr, _digit_width = nullptr;

	// cl_kernel _forward4 = nullptr, _backward4 = nullptr, _forward16 = nullptr, _backward16 = nullptr;
	cl_kernel _forward64 = nullptr, _backward64 = nullptr, _forward256 = nullptr, _backward256 = nullptr;
	// cl_kernel _forward1024 = nullptr, _backward1024 = nullptr;

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
	cl_kernel _carry_weight_mul_p1 = nullptr, _carry_weight_add_p1 = nullptr, _carry_weight_add_neg_p1 = nullptr, _carry_weight_p2 = nullptr, _carry_weight_addsub_p1 = nullptr, _carry_weight_p2x2 = nullptr, _carry_weight_mul_p1_copy = nullptr, _carry_weight_p2_copy = nullptr, _carry_weight_addsub_p1_copy = nullptr, _carry_weight_p2x2_copy = nullptr, _carry_weight_mul2_unit_p1 = nullptr;
	cl_kernel _copy = nullptr, _subtract = nullptr;
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

	virtual ~gpu() {}

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
			_root = _create_buffer(CL_MEM_READ_ONLY, 3 * n * sizeof(uint64));
			_weight = _create_buffer(CL_MEM_READ_ONLY, 2 * n * sizeof(uint64));
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

		// CREATE_KERNEL_TRANSFORM(forward1024);
		// CREATE_KERNEL_TRANSFORM(backward1024);
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
		CREATE_KERNEL_CARRY(carry_weight_addsub_p1);
		CREATE_KERNEL_CARRY(carry_weight_p2x2);
		CREATE_KERNEL_CARRY(carry_weight_mul_p1_copy);
		CREATE_KERNEL_CARRY(carry_weight_p2_copy);
		CREATE_KERNEL_CARRY(carry_weight_addsub_p1_copy);
		CREATE_KERNEL_CARRY(carry_weight_p2x2_copy);
		CREATE_KERNEL_CARRY(carry_weight_mul2_unit_p1);

		_copy = _create_kernel("copy");
		_set_kernel_arg(_copy, 0, sizeof(cl_mem), &_reg);
		_kernels.push_back(_copy);

		_subtract = _create_kernel("subtract");
		_set_kernel_arg(_subtract, 0, sizeof(cl_mem), &_reg);
		_set_kernel_arg(_subtract, 1, sizeof(cl_mem), &_weight);
		_set_kernel_arg(_subtract, 2, sizeof(cl_mem), &_digit_width);
		_kernels.push_back(_subtract);
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

	void write_root(const uint64 * const ptr) { _write_buffer(_root, ptr, 3 * _n * sizeof(uint64)); }
	void write_weight(const uint64 * const ptr) { _write_buffer(_weight, ptr, 2 * _n * sizeof(uint64)); }
	void write_width(const uint8 * const ptr) { _write_buffer(_digit_width, ptr, _n * sizeof(uint8)); }

///////////////////////////////

	void ek_fb(cl_kernel & kernel, const size_t src, const uint32 s, const uint32 lm, const size_t local_size = 0)
	{
		const uint32 offset = uint32(src * _n);
		_set_kernel_arg(kernel, 2, sizeof(uint32), &offset);
		_set_kernel_arg(kernel, 3, sizeof(uint32), &s);
		_set_kernel_arg(kernel, 4, sizeof(uint32), &lm);
		_execute_kernel(kernel, _n / 8, local_size);
	}

	void ek_fb_0(cl_kernel & kernel, const size_t step, const size_t src, const size_t local_size = 0)
	{
		const uint32 offset = uint32(src * _n);
		_set_kernel_arg(kernel, 2, sizeof(uint32), &offset);
		_execute_kernel(kernel, _n / step, local_size);
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

	// DEFINE_FORWARD(1024);
	// DEFINE_BACKWARD(1024);
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
	DEFINE_MUL(512);

	DEFINE_FORWARD_MUL(1024);
	DEFINE_SQR(1024);
	DEFINE_MUL(1024);

	// DEFINE_FORWARD_MUL(2048);
	// DEFINE_SQR(2048);
	// DEFINE_MUL(2048);

	void carry_weight_mul(const size_t src, const uint32 a)
	{
		const uint32 offset = uint32(src * _n);
		_set_kernel_arg(_carry_weight_mul_p1, 4, sizeof(uint32), &a);
		_set_kernel_arg(_carry_weight_mul_p1, 5, sizeof(uint32), &offset);
		_execute_kernel(_carry_weight_mul_p1, _n / 4, 1u << _lcwm_wg_size);
		_set_kernel_arg(_carry_weight_p2, 4, sizeof(uint32), &offset);
		_execute_kernel(_carry_weight_p2, (_n / 4) >> _lcwm_wg_size);
	}

	void carry_weight_mul_copy(const size_t src, const size_t dst, const uint32 a)
	{
		uint32 offS = (uint32)(src * _n);
		uint32 offD = (uint32)(dst * _n);
		_set_kernel_arg(_carry_weight_mul_p1_copy, 4, sizeof(uint32), &a);
		_set_kernel_arg(_carry_weight_mul_p1_copy, 5, sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_mul_p1_copy, 6, sizeof(uint32), &offD);
		_execute_kernel(_carry_weight_mul_p1_copy, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_p2_copy, 4, sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_p2_copy, 5, sizeof(uint32), &offD);
		_execute_kernel(_carry_weight_p2_copy, (_n / 4) >> _lcwm_wg_size);
	}

	void carry_weight_muladd(const size_t dst, const size_t add_src, const uint32 a)
	{
		const uint32 offY = uint32(dst * _n);
		const uint32 offX = uint32(add_src * _n);

		_set_kernel_arg(_carry_weight_muladd_p1, 4, sizeof(uint32), &a);
		_set_kernel_arg(_carry_weight_muladd_p1, 5, sizeof(uint32), &offY);
		_set_kernel_arg(_carry_weight_muladd_p1, 6, sizeof(uint32), &offX);
		_execute_kernel(_carry_weight_muladd_p1, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_muladd_p2, 4, sizeof(uint32), &offY);
		_execute_kernel(_carry_weight_muladd_p2, (_n / 4) >> _lcwm_wg_size);
	}
	
	void carry_weight_add(const size_t dst, const size_t src)
	{
		const uint32 offset_y = uint32(dst * _n), offset_x = uint32(src * _n);
		_set_kernel_arg(_carry_weight_add_p1, 4, sizeof(uint32), &offset_y);
		_set_kernel_arg(_carry_weight_add_p1, 5, sizeof(uint32), &offset_x);
		_execute_kernel(_carry_weight_add_p1, _n / 4, 1u << _lcwm_wg_size);
		_set_kernel_arg(_carry_weight_p2, 4, sizeof(uint32), &offset_y);
		_execute_kernel(_carry_weight_p2, (_n / 4) >> _lcwm_wg_size);

	}
	
	void carry_weight_sub(const size_t dst, const size_t src)
	{
		const uint32 offset_y = uint32(dst * _n), offset_x = uint32(src * _n);
		_set_kernel_arg(_carry_weight_add_neg_p1, 4, sizeof(uint32), &offset_y);
		_set_kernel_arg(_carry_weight_add_neg_p1, 5, sizeof(uint32), &offset_x);
		_execute_kernel(_carry_weight_add_neg_p1, _n / 4, 1u << _lcwm_wg_size);
		_set_kernel_arg(_carry_weight_p2, 4, sizeof(uint32), &offset_y);
		_execute_kernel(_carry_weight_p2, (_n / 4) >> _lcwm_wg_size);

	}

	
	void carry_weight_addsub(const size_t sum, const size_t diff, const size_t a, const size_t b)
	{
		uint32 offS = (uint32)(sum * _n);
		uint32 offD = (uint32)(diff * _n);
		uint32 offA = (uint32)(a   * _n);
		uint32 offB = (uint32)(b   * _n);
		_set_kernel_arg(_carry_weight_addsub_p1, 4, sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_addsub_p1, 5, sizeof(uint32), &offD);
		_set_kernel_arg(_carry_weight_addsub_p1, 6, sizeof(uint32), &offA);
		_set_kernel_arg(_carry_weight_addsub_p1, 7, sizeof(uint32), &offB);
		_execute_kernel(_carry_weight_addsub_p1, _n / 4, 1u << _lcwm_wg_size);
		_set_kernel_arg(_carry_weight_p2x2, 4, sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_p2x2, 5, sizeof(uint32), &offD);
		_execute_kernel(_carry_weight_p2x2, (_n / 4) >> _lcwm_wg_size);
	}

	void carry_weight_mul2_unit(const size_t dst0, const size_t dst1)
	{
		uint32 off0 = (uint32)(dst0 * _n);
		uint32 off1 = (uint32)(dst1 * _n);

		_set_kernel_arg(_carry_weight_mul2_unit_p1, 4, sizeof(uint32), &off0);
		_set_kernel_arg(_carry_weight_mul2_unit_p1, 5, sizeof(uint32), &off1);
		_execute_kernel(_carry_weight_mul2_unit_p1, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_p2x2, 4, sizeof(uint32), &off0);
		_set_kernel_arg(_carry_weight_p2x2, 5, sizeof(uint32), &off1);
		_execute_kernel(_carry_weight_p2x2, (_n / 4) >> _lcwm_wg_size);
	}

	void addsub_copy(const size_t sum, const size_t diff, const size_t sum_copy, const size_t diff_copy,
					const size_t a, const size_t b)
	{
		const uint32 offS  = (uint32)(sum  * _n);
		const uint32 offD  = (uint32)(diff * _n);
		const uint32 offSc = (uint32)(sum_copy  * _n);
		const uint32 offDc = (uint32)(diff_copy * _n);
		const uint32 offA  = (uint32)(a * _n);
		const uint32 offB  = (uint32)(b * _n);

		_set_kernel_arg(_carry_weight_addsub_p1_copy, 4, sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, 5, sizeof(uint32), &offD);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, 6, sizeof(uint32), &offSc);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, 7, sizeof(uint32), &offDc);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, 8, sizeof(uint32), &offA);
		_set_kernel_arg(_carry_weight_addsub_p1_copy, 9, sizeof(uint32), &offB);
		_execute_kernel(_carry_weight_addsub_p1_copy, _n / 4, 1u << _lcwm_wg_size);

		_set_kernel_arg(_carry_weight_p2x2_copy, 4, sizeof(uint32), &offS);
		_set_kernel_arg(_carry_weight_p2x2_copy, 5, sizeof(uint32), &offD);
		_set_kernel_arg(_carry_weight_p2x2_copy, 6, sizeof(uint32), &offSc);
		_set_kernel_arg(_carry_weight_p2x2_copy, 7, sizeof(uint32), &offDc);
		_execute_kernel(_carry_weight_p2x2_copy, (_n / 4) >> _lcwm_wg_size);
	}

	void copy(const size_t dst, const size_t src)
	{
		const uint32 offset_y = uint32(dst * _n), offset_x = uint32(src * _n);
		_set_kernel_arg(_copy, 1, sizeof(uint32), &offset_y);
		_set_kernel_arg(_copy, 2, sizeof(uint32), &offset_x);
		_execute_kernel(_copy, _n);
	}

	void subtract(const size_t src, const uint32 a)
	{
		const uint32 offset = uint32(src * _n);
		_set_kernel_arg(_subtract, 3, sizeof(uint32), &offset);
		_set_kernel_arg(_subtract, 4, sizeof(uint32), &a);
		_execute_kernel(_subtract, 1);
	}
};

class engine_gpu : public engine
{
private:
	const size_t _reg_count;
	const size_t _n;
	gpu * _gpu;
	std::vector<uint64> _weight;
	std::vector<uint8> _digit_width;

public:
	engine_gpu(const uint32_t q, const size_t reg_count, const size_t device, const bool verbose) : engine(),
		_reg_count(reg_count), _n(ibdwt::transform_size(q))
	{
		const size_t n = _n;

		const ocl::platform eng_platform = ocl::platform();
		_gpu = new gpu(eng_platform, device, n, _reg_count, verbose);

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

		src << "#define MAX_WG_SZ\t" << _gpu->get_max_workgroup_size() << "u" << std::endl << std::endl;

		if (!_gpu->read_OpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_gpu->load_program(src.str());
		_gpu->alloc_memory();
		_gpu->create_kernels();

		std::vector<uint64> root(3 * n);
		ibdwt::roots(n, root.data());
		_gpu->write_root(root.data());

		_weight.resize(2 * n);
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

	void set(const Reg dst, const uint32 a) const override
	{
		const size_t n = _n;
		std::vector<uint64> x(n);

		x[0] = a;	// weight[0] = 1
		for (size_t k = 1; k < n; ++k) x[k] = 0;

		_gpu->write_reg(x.data(), size_t(dst));
	}

	void set(const Reg dst, uint64 * const d) const override
	{
		const size_t n = _n;
		const uint64 * const weight = _weight.data();

		// weight
		std::vector<uint64> x(n);
		for (size_t k = 0; k < n; ++k)
		{
			const uint64 w = weight[2 * (k / 4 + (k % 4) * (n / 4)) + 0];
			x[k] = mod_mul(uint32(d[k]), w);
		}

		_gpu->write_reg(x.data(), size_t(dst));
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
		set_multiplicand2(dst0, dst0);
		set_multiplicand2(dst1, dst1);

		set_multiplicand(src0, src0);
		set_multiplicand(src1, src1);

		mul_new_core(dst0, src0);
		mul_new_core(dst1, src1);

		_gpu->carry_weight_mul2_unit(size_t(dst0), size_t(dst1));
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
		_gpu->carry_weight_sub(size_t(dst), size_t(src)); 
	}

	void addsub(const Reg sum_out, const Reg diff_out, const Reg a, const Reg b) const override
	{
		_gpu->carry_weight_addsub((size_t)sum_out, (size_t)diff_out, (size_t)a, (size_t)b);
	}

	void addsub_copy(const Reg sum, const Reg diff, const Reg sum_copy, const Reg diff_copy,
					const Reg a, const Reg b) const override
	{
		_gpu->addsub_copy((size_t)sum,(size_t)diff,(size_t)sum_copy,(size_t)diff_copy,(size_t)a,(size_t)b);
	}

	size_t get_register_data_size() const override { return _reg_count * _n * sizeof(uint64); }

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
