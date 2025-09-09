/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <vector>

#include "arith.h"

class engine
{
protected:
	// d is encoded: low 32-bit word is the value and high 32-bit word is the width of the base
	virtual void get(uint64 * const d, const size_t src) const = 0;	

public:
	engine() {}
	virtual ~engine() {}

	// a register
	typedef size_t Reg;

	// get transform size
	virtual size_t get_size() const = 0;
	// dst = a
	virtual void set(const Reg dst, const uint32 a) const = 0;
	// dst = src
	virtual void copy(const Reg dst, const Reg src) const = 0;
	// src1 ?= src2
	virtual bool is_equal(const Reg src1, const Reg src2) const = 0;
	// src = src * a
	virtual void square_mul(const Reg src, const uint32 a = 1) const = 0;
	// dst = multiplicand(src). A multiplicand is the src of the mul operation.
	virtual void set_multiplicand(const Reg dst, const Reg src) const = 0;
	// dst = dst * src. src must be a multiplicand, created with set_multiplicand.
	virtual void mul(const Reg dst, const Reg src) const = 0;
	// src = src - a
	virtual void sub(const Reg src, const uint32 a) const = 0;

	// get size in bytes of a register
	virtual size_t get_register_data_size() const = 0;
	// copy the content of src to data. The size of data must be equal to get_register_data_size().
	virtual bool get_data(std::vector<char> & data, const Reg src) const = 0;
	// copy the content of data to dst. The size of data must be equal to get_register_data_size().
	virtual bool set_data(const Reg dst, const std::vector<char> & data) const = 0;

	// get size in bytes of all registers
	virtual size_t get_checkpoint_size() const = 0;
	// copy all registers to data. The size of data must be equal to get_checkpoint_size().
	virtual bool get_checkpoint(std::vector<char> & data) const = 0;
	// copy data to all registers. The size of data must be equal to get_checkpoint_size().
	virtual bool set_checkpoint(const std::vector<char> & data) const = 0;

	// dst = src^e, src is erased
	void pow(const Reg dst, const Reg src, const uint64 e) const
	{
		set_multiplicand(src, src);
		set(dst, 1);
		if (e == 0) return;
		for (int i = std::bit_width(e) - 1; i >= 0; --i)
		{
			square_mul(dst);
			if ((e & (static_cast<uint64>(1) << i)) != 0) mul(dst, src);
		}
	}

	class digit
	{
	private:
		std::vector<uint64> _data;

	public:
		// unsigned digit representation of src using IBDWT base
		digit(engine * const eng, const Reg src)
		{
			_data.resize(eng->get_size());
			eng->get(_data.data(), src);
		}

		virtual ~digit() {}

		// get transform size
		size_t get_size() const { return _data.size(); }
		// digit[i]
		uint32 val(const size_t i) const { return uint32(_data[i]); }
		// base of digit[i] is 2^width[i]
		uint8 width(const size_t i) const { return uint8(_data[i] >> 32); }

		// 64-bit residue: src modulo 2^64
		uint64 res64() const
		{
			uint64 r64 = 0; uint8 s = 0;
			for (uint64 d :_data)
			{
				const uint64 u = uint32(d);
				const uint8 width = uint8(d >> 32);
				r64 += u << s;
				s += width;
				if (s >= 64) break;
			}
			return r64;
		}

		// src ?= a
		bool equal_to(const uint64 a) const
		{
			uint64 r = a;
			for (uint64 d :_data)
			{
				const uint64 u = uint32(d);
				const uint8 width = uint8(d >> 32);
				if ((r & ((uint64(1) << width) - 1)) != u) return false;
				r >>= width;
			}
			return true;
		}

		// src ?= 2^p - 1 (the Mersenne number)
		bool equal_to_Mp() const
		{
			for (uint64 d :_data)
			{
				const uint64 u = uint32(d);
				const uint8 width = uint8(d >> 32);
				if (u != (uint64(1) << width) - 1) return false;
			} 
			return true;
		}
	};

	static engine * create_gpu(const uint32_t q, const size_t reg_count, const size_t device, const bool verbose);
	static engine * create_cpu(const uint32_t q, const size_t reg_count);
};
