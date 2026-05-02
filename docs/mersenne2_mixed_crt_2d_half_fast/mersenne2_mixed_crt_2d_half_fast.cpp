/*
Original copyright 2025, Yves Gallot
Copyright 2026, modified version by Cherubrock (experimental mixed-radix CRT)

mersenne2.cpp is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

Original code by Yves Gallot:
https://github.com/galloty/mersenne2

Modified experimental version by Sébastien "cherubrock":
https://github.com/cherubrock-seb/PrMers/tree/main/docs/mersenne2_mixed_crt_2d_half_fast

This variant tries to use odd radix sizes by separating the odd axis with CRT indexing.
The original power-of-two half-real GF(p^2) transform is kept for the 2^m axis.

Odd roots are used as real scalars in GF(p):
(a+b*i)*r = (a*r)+(b*r)*i.

Radix 3, 7, 9, 11, 21, 33 and 63 use small specialized butterflies.
Radix 33 is mainly used for jump cases before the next radix-9 size.

Radix 11 alone is not used by default because 3*2^(m+2) is usually faster.
It can still be forced, for example:
./mersenne2_mixed_crt_2d_half_fast 194753086 11
*/

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>

static uint64_t inv_mod_u64(uint64_t a, const uint64_t m)
{
	int64_t t = 0, nt = 1;
	int64_t r = int64_t(m), nr = int64_t(a % m);
	while (nr != 0)
	{
		const int64_t q = r / nr;
		const int64_t tt = t - q * nt; t = nt; nt = tt;
		const int64_t rr = r - q * nr; r = nr; nr = rr;
	}
	if (r != 1) throw std::runtime_error("no modular inverse");
	if (t < 0) t += int64_t(m);
	return uint64_t(t);
}


// Z/{2^61 - 1}Z: the prime field of order p = 2^61 - 1
class Z61
{
private:
	static const uint64_t _p = (uint64_t(1) << 61) - 1;
	uint64_t _n;	// 0 <= n < p

	static uint64_t _add(const uint64_t a, const uint64_t b)
	{
		const uint64_t t = a + b;
		return t - ((t >= _p) ? _p : 0);
	}

	static uint64_t _sub(const uint64_t a, const uint64_t b)
	{
		const uint64_t t = a - b;
		return t + ((a < b) ? _p : 0);
	}

	static uint64_t _mul(const uint64_t a, const uint64_t b)
	{
		const __uint128_t t = a * __uint128_t(b);
		return _add(uint64_t(t) & _p, uint64_t(t >> 61));
	}

	static uint64_t _lshift(const uint64_t a, const uint8_t s)
	{
		const __uint128_t t = __uint128_t(a) << s;
		return _add(uint64_t(t) & _p, uint64_t(t >> 61));
	}

public:
	Z61() {}
	explicit Z61(const uint64_t n) : _n(n) {}

	uint64_t get() const { return _n; }

	bool operator==(const Z61 & rhs) const { return (_n == rhs._n); }
	bool operator!=(const Z61 & rhs) const { return (_n != rhs._n); }

	Z61 pow(uint64_t e) const
	{
		Z61 r(1), y(*this);
		while (e != 0) { if ((e & 1) != 0) r = r * y; e >>= 1; if (e != 0) y = y.sqr(); }
		return r;
	}

	static Z61 inv_uint(const uint64_t n) { return Z61(n).pow(_p - 2); }

	static Z61 root_odd_nth(const size_t n)
	{
		static Z61 cache[128];
		static bool has[128] = {};
		if (n == 1) return Z61(1);
		if (n < 128 && has[n]) return cache[n];
		if (((_p - 1) % n) != 0) throw std::runtime_error("unsupported odd radix for Z61");
		for (uint64_t g = 2; g < 1000000; ++g)
		{
			const Z61 z = Z61(g).pow((_p - 1) / n);
			if (z.pow(n) != Z61(1)) continue;
			bool primitive = true;
			size_t t = n;
			for (size_t f = 2; f <= t / f; ++f)
			{
				if ((t % f) == 0)
				{
					if (z.pow(n / f) == Z61(1)) primitive = false;
					while ((t % f) == 0) t /= f;
				}
			}
			if (t > 1 && z.pow(n / t) == Z61(1)) primitive = false;
			if (primitive) { if (n < 128) { cache[n] = z; has[n] = true; } return z; }
		}
		throw std::runtime_error("cannot find Z61 root");
	}

	// Z61 neg() const { return Z61((_n == 0) ? 0 : _p - _n); }
	// Z61 half() const { return Z61(((_n % 2 == 0) ? _n : (_n + _p)) / 2); }

	Z61 operator+(const Z61 & rhs) const { return Z61(_add(_n, rhs._n)); }
	Z61 operator-(const Z61 & rhs) const { return Z61(_sub(_n, rhs._n)); }
	Z61 operator*(const Z61 & rhs) const { return Z61(_mul(_n, rhs._n)); }

	Z61 sqr() const { return Z61(_mul(_n, _n)); }

	Z61 lshift(const uint8_t s) const { const uint8_t s61 = s % 61; return (s61 != 0) ? Z61(_lshift(_n, s61)) : *this; }
	Z61 rshift(const uint8_t s) const { const uint8_t s61 = s % 61; return (s61 != 0) ? Z61(_lshift(_n, 61 - s61)) : *this; }
};

// GF((2^61 - 1)^2): the prime field of order p^2, p = 2^61 - 1
class GF61
{
private:
	Z61 _s0, _s1;
	// a primitive root of order 2^62 which is a root of (0, 1).
	static const uint64_t _h_order = uint64_t(1) << 62;
	static const uint64_t _h_0 = 264036120304204ull, _h_1 = 4677669021635377ull;

public:
	GF61() {}
	explicit GF61(const Z61 & s0, const Z61 & s1) : _s0(s0), _s1(s1) {}
	explicit GF61(const uint64_t n0, const uint64_t n1) : _s0(n0), _s1(n1) {}

	const Z61 & s0() const { return _s0; }
	const Z61 & s1() const { return _s1; }

	void set0(const uint64_t n0) { _s0 = Z61(n0); }
	void set1(const uint64_t n1) { _s1 = Z61(n1); }

	bool operator!=(const GF61 & rhs) const { return ((_s0 != rhs._s0) || (_s1 != rhs._s1)); }

	// GF61 conj() const { return GF61(_s0, _s1.neg()); }
	// GF61 muli() const { return GF61(_s1.neg(), _s0); }
	// GF61 half() const { return GF61(_s0.half(), _s1.half()); }

	GF61 operator+(const GF61 & rhs) const { return GF61(_s0 + rhs._s0, _s1 + rhs._s1); }
	GF61 operator-(const GF61 & rhs) const { return GF61(_s0 - rhs._s0, _s1 - rhs._s1); }
	GF61 addconj(const GF61 & rhs) const { return GF61(_s0 + rhs._s0, _s1 - rhs._s1); }
	GF61 subconj(const GF61 & rhs) const { return GF61(_s0 - rhs._s0, _s1 + rhs._s1); }
	GF61 sub_conj(const GF61 & rhs) const { return GF61(_s0 - rhs._s0, rhs._s1 - _s1); }
	GF61 addi(const GF61 & rhs) const { return GF61(_s0 - rhs._s1, _s1 + rhs._s0); }
	GF61 subi(const GF61 & rhs) const { return GF61(_s0 + rhs._s1, _s1 - rhs._s0); }

	GF61 sqr() const { const Z61 t = _s0 * _s1; return GF61(_s0.sqr() - _s1.sqr(), t + t); }
	GF61 mul(const GF61 & rhs) const { return GF61(_s0 * rhs._s0 - _s1 * rhs._s1, _s1 * rhs._s0 + _s0 * rhs._s1); }
	GF61 mul_real(const Z61 & rhs) const { return GF61(_s0 * rhs, _s1 * rhs); }
	GF61 mulconj(const GF61 & rhs) const { return GF61(_s0 * rhs._s0 + _s1 * rhs._s1, _s1 * rhs._s0 - _s0 * rhs._s1); }

	GF61 lshift(const uint8_t ls0, const uint8_t ls1) const { return GF61(_s0.lshift(ls0), _s1.lshift(ls1)); }
	GF61 rshift(const uint8_t rs0, const uint8_t rs1) const { return GF61(_s0.rshift(rs0), _s1.rshift(rs1)); }

	GF61 pow(const uint64_t e) const
	{
		if (e == 0) return GF61(1u, 0u);
		GF61 r = GF61(1u, 0u), y = *this;
		for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF61 root_nth(const size_t n)
	{
		size_t odd = n; while ((odd & 1) == 0) odd >>= 1;
		const size_t pow2 = n / odd;
		GF61 r(1u, 0u);
		if (pow2 > 1) r = r.mul(GF61(Z61(_h_0), Z61(_h_1)).pow(_h_order / pow2));
		if (odd > 1) r = r.mul(GF61(Z61::root_odd_nth(odd), Z61(0)));
		return r;
	}
	static uint8_t log2_root_two(const size_t n) { return uint8_t(inv_mod_u64(n % 61, 61)); }
};

// Z/{2^31 - 1}Z: the prime field of order p = 2^31 - 1
class Z31
{
private:
	static const uint32_t _p = (uint32_t(1) << 31) - 1;
	uint32_t _n;	// 0 <= n < p

	static uint32_t _add(const uint32_t a, const uint32_t b)
	{
		const uint32_t t = a + b;
		return t - ((t >= _p) ? _p : 0);
	}

	static uint32_t _sub(const uint32_t a, const uint32_t b)
	{
		const uint32_t t = a - b;
		return t + ((a < b) ? _p : 0);
	}

	static uint32_t _mul(const uint32_t a, const uint32_t b)
	{
		const uint64_t t = a * uint64_t(b);
		return _add(uint32_t(t) & _p, uint32_t(t >> 31));
	}

	static uint32_t _lshift(const uint32_t a, const uint8_t s)
	{
		const uint64_t t = uint64_t(a) << s;
		return _add(uint32_t(t) & _p, uint32_t(t >> 31));
	}

public:
	Z31() {}
	explicit Z31(const uint32_t n) : _n(n) {}
	explicit Z31(const uint64_t n) : _n(n % _p) {}

	uint32_t get() const { return _n; }

	bool operator==(const Z31 & rhs) const { return (_n == rhs._n); }
	bool operator!=(const Z31 & rhs) const { return (_n != rhs._n); }

	Z31 pow(uint64_t e) const
	{
		Z31 r(1u), y(*this);
		while (e != 0) { if ((e & 1) != 0) r = r * y; e >>= 1; if (e != 0) y = y.sqr(); }
		return r;
	}

	static Z31 inv_uint(const uint64_t n) { return Z31(n).pow(_p - 2); }

	static Z31 root_odd_nth(const size_t n)
	{
		static Z31 cache[128];
		static bool has[128] = {};
		if (n == 1) return Z31(1u);
		if (n < 128 && has[n]) return cache[n];
		if (((_p - 1) % n) != 0) throw std::runtime_error("unsupported odd radix for Z31");
		for (uint32_t g = 2; g < 1000000; ++g)
		{
			const Z31 z = Z31(g).pow((_p - 1) / n);
			if (z.pow(n) != Z31(1u)) continue;
			bool primitive = true;
			size_t t = n;
			for (size_t f = 2; f <= t / f; ++f)
			{
				if ((t % f) == 0)
				{
					if (z.pow(n / f) == Z31(1u)) primitive = false;
					while ((t % f) == 0) t /= f;
				}
			}
			if (t > 1 && z.pow(n / t) == Z31(1u)) primitive = false;
			if (primitive) { if (n < 128) { cache[n] = z; has[n] = true; } return z; }
		}
		throw std::runtime_error("cannot find Z31 root");
	}

	// Z31 neg() const { return Z31((_n == 0) ? 0 : _p - _n); }
	// Z31 half() const { return Z31(((_n % 2 == 0) ? _n : (_n + _p)) / 2); }

	Z31 operator+(const Z31 & rhs) const { return Z31(_add(_n, rhs._n)); }
	Z31 operator-(const Z31 & rhs) const { return Z31(_sub(_n, rhs._n)); }
	Z31 operator*(const Z31 & rhs) const { return Z31(_mul(_n, rhs._n)); }

	Z31 sqr() const { return Z31(_mul(_n, _n)); }

	Z31 lshift(const uint8_t s) const { const uint8_t s31 = s % 31; return (s31 != 0) ? Z31(_lshift(_n, s31)) : *this; }
	Z31 rshift(const uint8_t s) const { const uint8_t s31 = s % 31; return (s31 != 0) ? Z31(_lshift(_n, 31 - s31)) : *this; }
};

// GF((2^31 - 1)^2): the prime field of order p^2, p = 2^31 - 1
class GF31
{
private:
	Z31 _s0, _s1;
	// a primitive root of order 2^32 which is a root of (0, 1).
	static const uint64_t _h_order = uint64_t(1) << 32;
	static const uint32_t _h_0 = 7735u, _h_1 = 748621u;

public:
	GF31() {}
	explicit GF31(const Z31 & s0, const Z31 & s1) : _s0(s0), _s1(s1) {}
	explicit GF31(const uint32_t n0, const uint32_t n1) : _s0(n0), _s1(n1) {}
	explicit GF31(const uint64_t n0, const uint64_t n1) : _s0(n0), _s1(n1) {}

	const Z31 & s0() const { return _s0; }
	const Z31 & s1() const { return _s1; }

	void set0(const uint32_t n0) { _s0 = Z31(n0); }
	void set1(const uint32_t n1) { _s1 = Z31(n1); }

	bool operator!=(const GF31 & rhs) const { return ((_s0 != rhs._s0) || (_s1 != rhs._s1)); }

	// GF31 conj() const { return GF31(_s0, _s1.neg()); }
	// GF31 muli() const { return GF31(_s1.neg(), _s0); }
	// GF31 half() const { return GF31(_s0.half(), _s1.half()); }

	GF31 operator+(const GF31 & rhs) const { return GF31(_s0 + rhs._s0, _s1 + rhs._s1); }
	GF31 operator-(const GF31 & rhs) const { return GF31(_s0 - rhs._s0, _s1 - rhs._s1); }
	GF31 addconj(const GF31 & rhs) const { return GF31(_s0 + rhs._s0, _s1 - rhs._s1); }
	GF31 subconj(const GF31 & rhs) const { return GF31(_s0 - rhs._s0, _s1 + rhs._s1); }
	GF31 sub_conj(const GF31 & rhs) const { return GF31(_s0 - rhs._s0, rhs._s1 - _s1); }
	GF31 addi(const GF31 & rhs) const { return GF31(_s0 - rhs._s1, _s1 + rhs._s0); }
	GF31 subi(const GF31 & rhs) const { return GF31(_s0 + rhs._s1, _s1 - rhs._s0); }

	GF31 sqr() const { const Z31 t = _s0 * _s1; return GF31(_s0.sqr() - _s1.sqr(), t + t); }
	GF31 mul(const GF31 & rhs) const { return GF31(_s0 * rhs._s0 - _s1 * rhs._s1, _s1 * rhs._s0 + _s0 * rhs._s1); }
	GF31 mul_real(const Z31 & rhs) const { return GF31(_s0 * rhs, _s1 * rhs); }
	GF31 mulconj(const GF31 & rhs) const { return GF31(_s0 * rhs._s0 + _s1 * rhs._s1, _s1 * rhs._s0 - _s0 * rhs._s1); }

	GF31 lshift(const uint8_t ls0, const uint8_t ls1) const { return GF31(_s0.lshift(ls0), _s1.lshift(ls1)); }
	GF31 rshift(const uint8_t rs0, const uint8_t rs1) const { return GF31(_s0.rshift(rs0), _s1.rshift(rs1)); }

	GF31 pow(const uint64_t e) const
	{
		if (e == 0) return GF31(1u, 0u);
		GF31 r = GF31(1u, 0u), y = *this;
		for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF31 root_nth(const size_t n)
	{
		size_t odd = n; while ((odd & 1) == 0) odd >>= 1;
		const size_t pow2 = n / odd;
		GF31 r(1u, 0u);
		if (pow2 > 1) r = r.mul(GF31(Z31(_h_0), Z31(_h_1)).pow(_h_order / pow2));
		if (odd > 1) r = r.mul(GF31(Z31::root_odd_nth(odd), Z31(0u)));
		return r;
	}
	static uint8_t log2_root_two(const size_t n) { return uint8_t(inv_mod_u64(n % 31, 31)); }
};

// Z/{2^61 - 1}Z x Z/{2^31 - 1}Z
class Z61_31
{
private:
	Z61 _n61;
	Z31 _n31;

public:
	Z61_31() {}
	explicit Z61_31(const Z61 & n61, const Z31 & n31) : _n61(n61), _n31(n31) {}

	const Z61 & n61() const { return _n61; }
	const Z31 & n31() const { return _n31; }

	Z61_31 operator+(const Z61_31 & rhs) const { return Z61_31(_n61 + rhs._n61, _n31 + rhs._n31); }
	Z61_31 operator-(const Z61_31 & rhs) const { return Z61_31(_n61 - rhs._n61, _n31 - rhs._n31); }
	Z61_31 operator*(const Z61_31 & rhs) const { return Z61_31(_n61 * rhs._n61, _n31 * rhs._n31); }

	Z61_31 sqr() const { return Z61_31(_n61.sqr(), _n31.sqr()); }
};

struct IBWeight
{
	uint8_t _w61, _w31;

	IBWeight() {}
	IBWeight(const uint8_t w61, const uint8_t w31) : _w61(w61), _w31(w31) {}
	IBWeight operator+(const uint8_t rhs) const { return IBWeight(_w61 + rhs, _w31 + rhs); }
};

// GF((2^61 - 1)^2) x GF((2^31 - 1)^2)
class GF61_31
{
private:
	GF61 _n61;
	GF31 _n31;

public:
	GF61_31() {}
	explicit GF61_31(const uint32_t n) : _n61(GF61(n, 0u)), _n31(GF31(n, 0u)) {}
	explicit GF61_31(const uint64_t n) : _n61(GF61(n, 0u)), _n31(GF31(n, uint64_t(0))) {}
	explicit GF61_31(const uint64_t n0, const uint64_t n1) : _n61(n0, n1), _n31(n0, n1) {}
	explicit GF61_31(const GF61 & n61, const GF31 & n31) : _n61(n61), _n31(n31) {}
	explicit GF61_31(const Z61_31 & s0, const Z61_31 & s1) : _n61(s0.n61(), s1.n61()), _n31(s0.n31(), s1.n31()) {}

	bool operator!=(const GF61_31 & rhs) const { return ((_n61 != rhs._n61) || (_n31 != rhs._n31)); }

	const Z61_31 s0() const { return Z61_31(_n61.s0(), _n31.s0()); }
	const Z61_31 s1() const { return Z61_31(_n61.s1(), _n31.s1()); }

	// GF61_31 conj() const { return GF61_31(_n61.conj(), _n31.conj()); }
	// GF61_31 muli() const { return GF61_31(_n61.muli(), _n31.muli()); }
	// GF61_31 half() const { return GF61_31(_n61.half(), _n31.half()); }

	GF61_31 operator+(const GF61_31 & rhs) const { return GF61_31(_n61 + rhs._n61, _n31 + rhs._n31); }
	GF61_31 operator-(const GF61_31 & rhs) const { return GF61_31(_n61 - rhs._n61, _n31 - rhs._n31); }
	GF61_31 addconj(const GF61_31 & rhs) const { return GF61_31(_n61.addconj(rhs._n61), _n31.addconj(rhs._n31)); }
	GF61_31 subconj(const GF61_31 & rhs) const { return GF61_31(_n61.subconj(rhs._n61), _n31.subconj(rhs._n31)); }
	GF61_31 sub_conj(const GF61_31 & rhs) const { return GF61_31(_n61.sub_conj(rhs._n61), _n31.sub_conj(rhs._n31)); }
	GF61_31 addi(const GF61_31 & rhs) const { return GF61_31(_n61.addi(rhs._n61), _n31.addi(rhs._n31)); }
	GF61_31 subi(const GF61_31 & rhs) const { return GF61_31(_n61.subi(rhs._n61), _n31.subi(rhs._n31)); }

	GF61_31 sqr() const { return GF61_31(_n61.sqr(), _n31.sqr()); }
	GF61_31 mul(const GF61_31 & rhs) const { return GF61_31(_n61.mul(rhs._n61), _n31.mul(rhs._n31)); }
	GF61_31 mul_real(const Z61_31 & rhs) const { return GF61_31(_n61.mul_real(rhs.n61()), _n31.mul_real(rhs.n31())); }
	GF61_31 mulconj(const GF61_31 & rhs) const { return GF61_31(_n61.mulconj(rhs._n61), _n31.mulconj(rhs._n31)); }

	GF61_31 lshift(const IBWeight ls0, const IBWeight ls1) const { return GF61_31(_n61.lshift(ls0._w61, ls1._w61), _n31.lshift(ls0._w31, ls1._w31)); }
	GF61_31 rshift(const IBWeight rs0, const IBWeight rs1) const { return GF61_31(_n61.rshift(rs0._w61, rs1._w61), _n31.rshift(rs0._w31, rs1._w31)); }

	GF61_31 pow(const uint64_t e) const { return GF61_31(_n61.pow(e), _n31.pow(e)); }

	static const GF61_31 root_nth(const size_t n) { return GF61_31(GF61::root_nth(n), GF31::root_nth(n)); }
	static const GF61_31 inv_uint(const uint64_t n)
	{
		return GF61_31(GF61(Z61::inv_uint(n), Z61(0)), GF31(Z31::inv_uint(n), Z31(0u)));
	}

	// Chinese remainder theorem
	void garner(__uint128_t & n_0, __uint128_t & n_1) const
	{
		const uint32_t n31_0 = _n31.s0().get(), n31_1 = _n31.s1().get();
		const GF61 n31 = GF61(n31_0, n31_1);
		GF61 u = _n61 - n31; 
		// The inverse of 2^31 - 1 mod 2^61 - 1 is 2^31 + 1
		u = u + u.lshift(31, 31);
		const uint64_t s_0 = u.s0().get(), s_1 = u.s1().get();
		n_0 = n31_0 + (__uint128_t(s_0) << 31) - s_0;
		n_1 = n31_1 + (__uint128_t(s_1) << 31) - s_1;
	}
};


struct TransformPlan
{
	uint8_t pow2_ln;
	size_t odd;
	size_t m;
	size_t n;
};

class mersenne
{
private:
	const TransformPlan _tp;
	const size_t _n;
	const size_t _odd;
	const size_t _m;
	const size_t _m2;
	GF61_31 * const _z;
	GF61_31 * const _w;
	IBWeight * const _w_ib;
	uint8_t * const _digit_width;
	size_t * const _j_of_coord;
	size_t * const _coord_of_j;
	GF61_31 * const _odd_fwd;
	GF61_31 * const _odd_inv;
	const GF61_31 _inv_odd;

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	static TransformPlan transformsize(const uint32_t q, const size_t forced_odd = 0)
	{
		// Default auto mode avoids radix-11 paths.
		// 33 = 3*11 is smaller in some jump ranges, but it is often slower than the next radix-9 size.
		// Radix-11 based paths can still be forced from the command line.
		const size_t odd_list_auto[] = {1, 3, 7, 9, 21, 63};
		const size_t odd_list_forced[] = {1, 3, 7, 9, 11, 21, 33, 63, 77, 99};
		const size_t * const odd_list = (forced_odd == 0) ? odd_list_auto : odd_list_forced;
		const size_t odd_count = (forced_odd == 0) ? (sizeof(odd_list_auto) / sizeof(odd_list_auto[0])) : (sizeof(odd_list_forced) / sizeof(odd_list_forced[0]));
		TransformPlan best{0, 1, 0, std::numeric_limits<size_t>::max()};

		for (uint8_t ln = 2; ln <= 30; ++ln)
		{
			const size_t m = size_t(1) << ln;
			for (size_t oi = 0; oi < odd_count; ++oi)
			{
				const size_t odd = odd_list[oi];
				if ((forced_odd != 0) && (odd != forced_odd)) continue;
				const size_t n = odd * m;
				if (n == 0 || n > q) continue;
				const long double log2n = (long double)ln + std::log2((long double)odd);
				const long double word_span = (long double)q / (long double)n;
				if (log2n + 2.0L * (word_span + 1.0L) < 92.0L)
				{
					if (n < best.n) best = TransformPlan{ln, odd, m, n};
				}
			}
		}

		if (best.n == std::numeric_limits<size_t>::max()) throw std::runtime_error("no safe transform size");
		return best;
	}

	static Z61_31 real_one()
	{
		return Z61_31(Z61(1), Z31(1u));
	}

	static Z61_31 real_pow(Z61_31 y, uint64_t e)
	{
		Z61_31 r = real_one();
		while (e != 0)
		{
			if ((e & 1) != 0) r = r * y;
			e >>= 1;
			if (e != 0) y = y.sqr();
		}
		return r;
	}

	void dft_small_matrix(const GF61_31 * const in, GF61_31 * const out, const GF61_31 * const matrix) const
	{
		for (size_t k = 0; k < _odd; ++k)
		{
			GF61_31 sum(0u);
			for (size_t j = 0; j < _odd; ++j) sum = sum + in[j].mul_real(matrix[k * _odd + j].s0());
			out[k] = sum;
		}
	}

	__attribute__((noinline)) static void dft3_direct(const GF61_31 & x0, const GF61_31 & x1, const GF61_31 & x2,
		const Z61_31 & root3, GF61_31 & y0, GF61_31 & y1, GF61_31 & y2)
	{
		const Z61_31 root3_2 = root3.sqr();
		y0 = x0 + x1 + x2;
		y1 = x0 + x1.mul_real(root3) + x2.mul_real(root3_2);
		y2 = x0 + x1.mul_real(root3_2) + x2.mul_real(root3);
	}

	__attribute__((noinline)) static void dft7_direct(const GF61_31 * const in, GF61_31 * const out, const Z61_31 & root7)
	{
		Z61_31 w[7];
		w[0] = real_one();
		for (size_t i = 1; i < 7; ++i) w[i] = w[i - 1] * root7;

		for (size_t k = 0; k < 7; ++k)
		{
			GF61_31 sum(0u);
			for (size_t j = 0; j < 7; ++j) sum = sum + in[j].mul_real(w[(j * k) % 7]);
			out[k] = sum;
		}
	}

	__attribute__((noinline)) static void dft11_direct(const GF61_31 * const in, GF61_31 * const out, const Z61_31 & root11)
	{
		Z61_31 w[11];
		w[0] = real_one();
		for (size_t i = 1; i < 11; ++i) w[i] = w[i - 1] * root11;

		for (size_t k = 0; k < 11; ++k)
		{
			GF61_31 sum(0u);
			for (size_t j = 0; j < 11; ++j) sum = sum + in[j].mul_real(w[(j * k) % 11]);
			out[k] = sum;
		}
	}

	__attribute__((noinline)) void dft9_radix3(const GF61_31 * const in, GF61_31 * const out, const Z61_31 & root9) const
	{
		const Z61_31 root3 = real_pow(root9, 3);
		const Z61_31 w1 = root9;
		const Z61_31 w2 = root9.sqr();
		const Z61_31 w4 = w2.sqr();

		GF61_31 a00, a01, a02, a10, a11, a12, a20, a21, a22;
		dft3_direct(in[0], in[3], in[6], root3, a00, a01, a02);
		dft3_direct(in[1], in[4], in[7], root3, a10, a11, a12);
		dft3_direct(in[2], in[5], in[8], root3, a20, a21, a22);

		GF61_31 y0, y1, y2;
		dft3_direct(a00, a10, a20, root3, y0, y1, y2);
		out[0] = y0; out[3] = y1; out[6] = y2;

		dft3_direct(a01, a11.mul_real(w1), a21.mul_real(w2), root3, y0, y1, y2);
		out[1] = y0; out[4] = y1; out[7] = y2;

		dft3_direct(a02, a12.mul_real(w2), a22.mul_real(w4), root3, y0, y1, y2);
		out[2] = y0; out[5] = y1; out[8] = y2;
	}

	__attribute__((noinline)) void dft21_radix3x7(const GF61_31 * const in, GF61_31 * const out, const Z61_31 & root21) const
	{
		const Z61_31 root7 = real_pow(root21, 3);
		const Z61_31 root3 = real_pow(root21, 7);
		GF61_31 tmp[21];
		GF61_31 vin[7], vout[7];

		for (size_t r = 0; r < 3; ++r)
		{
			for (size_t s = 0; s < 7; ++s) vin[s] = in[r + 3 * s];
			dft7_direct(vin, vout, root7);
			for (size_t ks = 0; ks < 7; ++ks) tmp[r * 7 + ks] = vout[ks];
		}

		Z61_31 tw1 = real_one();
		for (size_t ks = 0; ks < 7; ++ks)
		{
			const Z61_31 tw2 = tw1.sqr();
			GF61_31 y0, y1, y2;
			dft3_direct(tmp[0 * 7 + ks], tmp[1 * 7 + ks].mul_real(tw1), tmp[2 * 7 + ks].mul_real(tw2), root3, y0, y1, y2);
			out[ks + 7 * 0] = y0;
			out[ks + 7 * 1] = y1;
			out[ks + 7 * 2] = y2;
			tw1 = tw1 * root21;
		}
	}

	__attribute__((noinline)) void dft33_radix3x11(const GF61_31 * const in, GF61_31 * const out, const Z61_31 & root33) const
	{
		const Z61_31 root11 = real_pow(root33, 3);
		const Z61_31 root3 = real_pow(root33, 11);
		GF61_31 tmp[33];
		GF61_31 vin[11], vout[11];

		for (size_t r = 0; r < 3; ++r)
		{
			for (size_t s = 0; s < 11; ++s) vin[s] = in[r + 3 * s];
			dft11_direct(vin, vout, root11);
			for (size_t ks = 0; ks < 11; ++ks) tmp[r * 11 + ks] = vout[ks];
		}

		Z61_31 tw1 = real_one();
		for (size_t ks = 0; ks < 11; ++ks)
		{
			const Z61_31 tw2 = tw1.sqr();
			GF61_31 y0, y1, y2;
			dft3_direct(tmp[0 * 11 + ks], tmp[1 * 11 + ks].mul_real(tw1), tmp[2 * 11 + ks].mul_real(tw2), root3, y0, y1, y2);
			out[ks + 11 * 0] = y0;
			out[ks + 11 * 1] = y1;
			out[ks + 11 * 2] = y2;
			tw1 = tw1 * root33;
		}
	}

	__attribute__((noinline)) void dft77_radix7x11(const GF61_31 * const in, GF61_31 * const out, const Z61_31 & root77) const
	{
		const Z61_31 root11 = real_pow(root77, 7);
		const Z61_31 root7 = real_pow(root77, 11);
		GF61_31 tmp[77];
		GF61_31 vin11[11], vout11[11];

		for (size_t r = 0; r < 7; ++r)
		{
			for (size_t s = 0; s < 11; ++s) vin11[s] = in[r + 7 * s];
			dft11_direct(vin11, vout11, root11);
			for (size_t ks = 0; ks < 11; ++ks) tmp[r * 11 + ks] = vout11[ks];
		}

		GF61_31 vin7[7], vout7[7];
		Z61_31 step = real_one();
		for (size_t ks = 0; ks < 11; ++ks)
		{
			Z61_31 tw = real_one();
			for (size_t r = 0; r < 7; ++r)
			{
				vin7[r] = tmp[r * 11 + ks].mul_real(tw);
				tw = tw * step;
			}
			dft7_direct(vin7, vout7, root7);
			for (size_t kr = 0; kr < 7; ++kr) out[ks + 11 * kr] = vout7[kr];
			step = step * root77;
		}
	}

	__attribute__((noinline)) void dft99_radix9x11(const GF61_31 * const in, GF61_31 * const out, const Z61_31 & root99) const
	{
		const Z61_31 root11 = real_pow(root99, 9);
		const Z61_31 root9 = real_pow(root99, 11);
		GF61_31 tmp[99];
		GF61_31 vin11[11], vout11[11];

		for (size_t r = 0; r < 9; ++r)
		{
			for (size_t s = 0; s < 11; ++s) vin11[s] = in[r + 9 * s];
			dft11_direct(vin11, vout11, root11);
			for (size_t ks = 0; ks < 11; ++ks) tmp[r * 11 + ks] = vout11[ks];
		}

		GF61_31 vin9[9], vout9[9];
		Z61_31 step = real_one();
		for (size_t ks = 0; ks < 11; ++ks)
		{
			Z61_31 tw = real_one();
			for (size_t r = 0; r < 9; ++r)
			{
				vin9[r] = tmp[r * 11 + ks].mul_real(tw);
				tw = tw * step;
			}
			dft9_radix3(vin9, vout9, root9);
			for (size_t kr = 0; kr < 9; ++kr) out[ks + 11 * kr] = vout9[kr];
			step = step * root99;
		}
	}

	__attribute__((noinline)) void dft63_radix7x9(const GF61_31 * const in, GF61_31 * const out, const Z61_31 & root63) const
	{
		const Z61_31 root9 = real_pow(root63, 7);
		const Z61_31 root7 = real_pow(root63, 9);
		GF61_31 tmp[63];
		GF61_31 vin9[9], vout9[9];

		for (size_t r = 0; r < 7; ++r)
		{
			for (size_t s = 0; s < 9; ++s) vin9[s] = in[r + 7 * s];
			dft9_radix3(vin9, vout9, root9);
			for (size_t ks = 0; ks < 9; ++ks) tmp[r * 9 + ks] = vout9[ks];
		}

		GF61_31 vin7[7], vout7[7];
		Z61_31 step = real_one();
		for (size_t ks = 0; ks < 9; ++ks)
		{
			Z61_31 tw = real_one();
			for (size_t r = 0; r < 7; ++r)
			{
				vin7[r] = tmp[r * 9 + ks].mul_real(tw);
				tw = tw * step;
			}
			dft7_direct(vin7, vout7, root7);
			for (size_t kr = 0; kr < 7; ++kr) out[ks + 9 * kr] = vout7[kr];
			step = step * root63;
		}
	}

	__attribute__((noinline)) void dft_small_fast(const GF61_31 * const in, GF61_31 * const out, const GF61_31 * const matrix) const
	{
		if (_odd == 3)
		{
			dft3_direct(in[0], in[1], in[2], matrix[4].s0(), out[0], out[1], out[2]);
			return;
		}
		if (_odd == 7)
		{
			dft7_direct(in, out, matrix[8].s0());
			return;
		}
		if (_odd == 9)
		{
			dft9_radix3(in, out, matrix[10].s0());
			return;
		}
		if (_odd == 11)
		{
			dft11_direct(in, out, matrix[12].s0());
			return;
		}
		if (_odd == 21)
		{
			dft21_radix3x7(in, out, matrix[22].s0());
			return;
		}
		if (_odd == 33)
		{
			dft33_radix3x11(in, out, matrix[34].s0());
			return;
		}
		if (_odd == 63)
		{
			dft63_radix7x9(in, out, matrix[64].s0());
			return;
		}
		if (_odd == 77)
		{
			dft77_radix7x11(in, out, matrix[78].s0());
			return;
		}
		if (_odd == 99)
		{
			dft99_radix9x11(in, out, matrix[100].s0());
			return;
		}
		dft_small_matrix(in, out, matrix);
	}

	size_t slot(const size_t a, const size_t k) const { return a * _m2 + k; }
	size_t coord(const size_t a, const size_t b) const { return a * _m + b; }

	void build_crt_maps()
	{
		if (_odd == 1)
		{
			for (size_t b = 0; b < _m; ++b)
			{
				_j_of_coord[coord(0, b)] = b;
				_coord_of_j[b] = coord(0, b);
			}
			return;
		}

		const uint64_t inv_m = inv_mod_u64(_m % _odd, _odd);
		for (size_t a = 0; a < _odd; ++a)
		{
			for (size_t b = 0; b < _m; ++b)
			{
				const size_t bm = b % _odd;
				const size_t delta = (a + _odd - bm) % _odd;
				const size_t t = (delta * inv_m) % _odd;
				const size_t j = b + _m * t;
				_j_of_coord[coord(a, b)] = j;
				_coord_of_j[j] = coord(a, b);
			}
		}
	}

	void set_digit_by_j(const size_t j, const uint64_t value) const
	{
		const size_t c = _coord_of_j[j];
		const size_t a = c / _m;
		const size_t b = c - a * _m;
		const size_t k = b / 2;
		const GF61_31 old = _z[slot(a, k)];
		const uint64_t other = ((b & 1) == 0) ? old.s1().n61().get() : old.s0().n61().get();
		_z[slot(a, k)] = ((b & 1) == 0) ? GF61_31(value, other) : GF61_31(other, value);
	}

	uint64_t get_digit_by_j(const size_t j) const
	{
		const size_t c = _coord_of_j[j];
		const size_t a = c / _m;
		const size_t b = c - a * _m;
		const size_t k = b / 2;
		const GF61_31 v = _z[slot(a, k)];
		return ((b & 1) == 0) ? v.s0().n61().get() : v.s1().n61().get();
	}

	void forward_odd() const
	{
		if (_odd == 1) return;
		GF61_31 in[128], out[128];
		for (size_t k = 0; k < _m2; ++k)
		{
			for (size_t a = 0; a < _odd; ++a) in[a] = _z[slot(a, k)];
			dft_small_fast(in, out, _odd_fwd);
			for (size_t a = 0; a < _odd; ++a) _z[slot(a, k)] = out[a];
		}
	}

	void backward_odd_norm() const
	{
		if (_odd == 1) return;
		GF61_31 in[128], out[128];
		for (size_t k = 0; k < _m2; ++k)
		{
			for (size_t a = 0; a < _odd; ++a) in[a] = _z[slot(a, k)];
			dft_small_fast(in, out, _odd_inv);
			for (size_t a = 0; a < _odd; ++a) _z[slot(a, k)] = out[a].mul(_inv_odd);
		}
	}

	void forward4(GF61_31 * const z, const size_t m, const size_t s) const
	{
		const GF61_31 * const w = _w;
		const GF61_31 * const wr4 = &_w[_m / 2];

		for (size_t j = 0; j < s; ++j)
		{
			const GF61_31 w1 = w[s + j], w2 = w[2 * (s + j)], w3 = wr4[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GF61_31 u0 = z[k + 0 * m], u1 = z[k + 1 * m].mul(w2), u2 = z[k + 2 * m].mul(w1), u3 = z[k + 3 * m].mul(w3);
				const GF61_31 v0 = u0 + u2, v1 = u1 + u3, v2 = u0 - u2, v3 = u1 - u3;
				z[k + 0 * m] = v0 + v1; z[k + 1 * m] = v0 - v1;
				z[k + 2 * m] = v2.addi(v3); z[k + 3 * m] = v2.subi(v3);
			}
		}
	}

	void backward4(GF61_31 * const z, const size_t m, const size_t s) const
	{
		const GF61_31 * const w = _w;
		const GF61_31 * const wr4 = &_w[_m / 2];

		for (size_t j = 0; j < s; ++j)
		{
			const GF61_31 w1 = w[s + j], w2 = w[2 * (s + j)], w3 = wr4[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GF61_31 u0 = z[k + 0 * m], u1 = z[k + 1 * m], u2 = z[k + 2 * m], u3 = z[k + 3 * m];
				const GF61_31 v0 = u0 + u1, v1 = u0 - u1, v2 = u2 + u3, v3 = u3 - u2;
				z[k + 0 * m] = v0 + v2; z[k + 2 * m] = (v0 - v2).mulconj(w1);
				z[k + 1 * m] = v1.addi(v3).mulconj(w2); z[k + 3 * m] = v1.subi(v3).mulconj(w3);
			}
		}
	}

	void sqr_row(GF61_31 * const z) const
	{
		const GF61_31 * const w = &_w[_m / 4];

		for (size_t j = 0, m_4 = _m / 4; j < m_4; ++j)
		{
			const size_t k = 2 * j, mk = (k != 0) ? (size_t(3) << (63 - __builtin_clzll((unsigned long long)k))) - k - 1 : 0;
			const GF61_31 zk = z[k], zmk = z[mk];
			const GF61_31 u0 = zk.addconj(zmk), u1 = zk.subconj(zmk);
			const GF61_31 v0 = u0.sqr() - u1.sqr().mul(w[j]), v1 = u0.mul(u1 + u1);
			z[k] = v0 + v1;
			if (k == 0) z[1] = (z[1] + z[1]).sqr();
			else z[mk] = v0.sub_conj(v1);
		}
	}

	void sqr2_row(GF61_31 * const z) const
	{
		const size_t m_4 = _m / 4;
		const GF61_31 * const w = _w;

		for (size_t j = 0; j < m_4; ++j)
		{
			const GF61_31 u0 = z[2 * j + 0], u1 = z[2 * j + 1].mul(w[m_4 + j]);
			z[2 * j + 0] = u0 + u1; z[2 * j + 1] = u0 - u1;
		}

		sqr_row(z);

		for (size_t j = 0; j < m_4; ++j)
		{
			const GF61_31 u0 = z[2 * j + 0], u1 = z[2 * j + 1];
			z[2 * j + 0] = u0 + u1; z[2 * j + 1] = (u0 - u1).mulconj(w[m_4 + j]);
		}
	}

	void half_square_rows() const
	{
		for (size_t a = 0; a < _odd; ++a)
		{
			GF61_31 * const row = &_z[slot(a, 0)];
			size_t m = _m / 4, s = 1;
			for (; m > 1; m /= 4, s *= 4) forward4(row, m / 2, s);
			if (m == 1) sqr2_row(row); else sqr_row(row);
			for (m = (m == 1) ? 4 : 2, s /= 4; s >= 1; m *= 4, s /= 4) backward4(row, m / 2, s);
		}
	}

	void weight() const
	{
		for (size_t a = 0; a < _odd; ++a)
		{
			for (size_t k = 0; k < _m2; ++k)
			{
				const size_t b0 = 2 * k + 0, b1 = 2 * k + 1;
				const size_t j0 = _j_of_coord[coord(a, b0)];
				const size_t j1 = _j_of_coord[coord(a, b1)];
				_z[slot(a, k)] = _z[slot(a, k)].lshift(_w_ib[j0], _w_ib[j1]);
			}
		}
	}

	void unweight_norm() const
	{
		const uint8_t ln1 = _tp.pow2_ln + 1;
		for (size_t a = 0; a < _odd; ++a)
		{
			for (size_t k = 0; k < _m2; ++k)
			{
				const size_t b0 = 2 * k + 0, b1 = 2 * k + 1;
				const size_t j0 = _j_of_coord[coord(a, b0)];
				const size_t j1 = _j_of_coord[coord(a, b1)];
				_z[slot(a, k)] = _z[slot(a, k)].rshift(_w_ib[j0] + ln1, _w_ib[j1] + ln1);
			}
		}
	}

	static constexpr uint64_t digit_adc(const __uint128_t lhs, const uint8_t digit_width, __uint128_t & carry)
	{
		const __uint128_t s = lhs + carry;
		const __uint128_t c = (s < lhs) ? 1 : 0;
		carry = (s >> digit_width) + (c << (128 - digit_width));
		return uint64_t(s) & ((uint64_t(1) << digit_width) - 1);
	}

	void carry() const
	{
		std::vector<__uint128_t> coeff(_n);
		for (size_t a = 0; a < _odd; ++a)
		{
			for (size_t k = 0; k < _m2; ++k)
			{
				__uint128_t l0, l1; _z[slot(a, k)].garner(l0, l1);
				coeff[_j_of_coord[coord(a, 2 * k + 0)]] = l0;
				coeff[_j_of_coord[coord(a, 2 * k + 1)]] = l1;
			}
		}

		std::vector<uint64_t> out(_n);
		__uint128_t c = 0;
		for (size_t j = 0; j < _n; ++j) out[j] = digit_adc(coeff[j], _digit_width[j], c);

		while (c != 0)
		{
			for (size_t j = 0; j < _n; ++j)
			{
				out[j] = digit_adc(out[j], _digit_width[j], c);
				if (c == 0) break;
			}
		}

		for (size_t a = 0; a < _odd; ++a)
		{
			for (size_t k = 0; k < _m2; ++k)
			{
				const uint64_t n0 = out[_j_of_coord[coord(a, 2 * k + 0)]];
				const uint64_t n1 = out[_j_of_coord[coord(a, 2 * k + 1)]];
				_z[slot(a, k)] = GF61_31(n0, n1);
			}
		}
	}

public:
	mersenne(const uint32_t q, const size_t forced_odd = 0) : _tp(transformsize(q, forced_odd)), _n(_tp.n), _odd(_tp.odd), _m(_tp.m), _m2(_tp.m / 2),
		_z(new GF61_31[_odd * _m2]), _w(new GF61_31[_m]), _w_ib(new IBWeight[_n]), _digit_width(new uint8_t[_n]),
		_j_of_coord(new size_t[_n]), _coord_of_j(new size_t[_n]), _odd_fwd(new GF61_31[_odd * _odd]), _odd_inv(new GF61_31[_odd * _odd]), _inv_odd(GF61_31::inv_uint(_tp.odd))
	{
		build_crt_maps();

		const GF61_31 odd_root = GF61_31::root_nth(_odd);
		const GF61_31 odd_root_inv = odd_root.pow(uint64_t((_odd == 1) ? 0 : (_odd - 1)));
		for (size_t k = 0; k < _odd; ++k)
		{
			for (size_t j = 0; j < _odd; ++j)
			{
				_odd_fwd[k * _odd + j] = odd_root.pow(uint64_t(j * k));
				_odd_inv[k * _odd + j] = odd_root_inv.pow(uint64_t(j * k));
			}
		}

		for (size_t s = 1; s <= _m / 4; s *= 2)
		{
			const GF61_31 r_s = GF61_31::root_nth(2 * s);
			for (size_t j = 0; j < s; ++j) _w[s + j] = r_s.pow(bitrev(j, s));
		}

		for (size_t s = 1; s <= _m / 4; s *= 2)
		{
			for (size_t j = 0; j < s; ++j) _w[_m / 2 + s + j] = _w[s + j].mul(_w[2 * (s + j)]);
		}

		const uint8_t lr2_61 = GF61::log2_root_two(_n);
		const uint8_t lr2_31 = GF31::log2_root_two(_n);

		uint32_t o = 0;
		for (size_t j = 0; j <= _n; ++j)
		{
			const uint64_t qj = uint64_t(q) * j;
			const uint32_t ceil_qj_n = (j == 0) ? 0 : uint32_t((qj + _n - 1) / _n);

			if (j > 0)
			{
				const uint8_t c = uint8_t(ceil_qj_n - o);
				_digit_width[j - 1] = c;

				if (j < _n)
				{
					const uint32_t r = uint32_t(qj % _n);
					const uint8_t w61 = uint8_t((lr2_61 * ((_n - r) % _n)) % 61);
					const uint8_t w31 = uint8_t((lr2_31 * ((_n - r) % _n)) % 31);
					_w_ib[j] = IBWeight(w61, w31);
				}
			}
			o = ceil_qj_n;
		}
		_w_ib[0] = IBWeight(0, 0);

		std::cout << "p=" << q << ", transform=" << _tp.odd << "*2^" << int(_tp.pow2_ln)
			<< " = " << _n << ", storage=" << (_odd * _m2) << " complex values"
			<< " (2D odd radix + power-of-two half-real CRT)" << std::endl;
	}

	virtual ~mersenne()
	{
		delete[] _z;
		delete[] _w;
		delete[] _w_ib;
		delete[] _digit_width;
		delete[] _j_of_coord;
		delete[] _coord_of_j;
		delete[] _odd_fwd;
		delete[] _odd_inv;
	}

	size_t transform_length() const { return _n; }
	size_t storage_length() const { return _odd * _m2; }
	size_t odd_radix() const { return _odd; }
	uint8_t pow2_ln() const { return _tp.pow2_ln; }

	void init(const uint64_t a) const
	{
		for (size_t k = 0; k < _odd * _m2; ++k) _z[k] = GF61_31(0u);
		set_digit_by_j(0, a);
	}

	void square() const
	{
		weight();
		forward_odd();
		half_square_rows();
		backward_odd_norm();
		unweight_norm();
		carry();
	}

	static constexpr uint64_t digit_sbc(const uint64_t lhs, const uint8_t digit_width, uint32_t & carry)
	{
		const bool borrow = (lhs < carry);
		const uint64_t r = lhs - carry + (borrow ? (uint64_t(1) << digit_width) : 0);
		carry = borrow ? 1 : 0;
		return r;
	}

	void sub(const uint32_t a) const
	{
		std::vector<uint64_t> out(_n);
		for (size_t j = 0; j < _n; ++j) out[j] = get_digit_by_j(j);

		uint32_t c = a;
		while (c != 0)
		{
			for (size_t j = 0; j < _n; ++j)
			{
				out[j] = digit_sbc(out[j], _digit_width[j], c);
				if (c == 0) break;
			}
		}

		for (size_t j = 0; j < _n; ++j) set_digit_by_j(j, out[j]);
	}

	bool is_zero() const
	{
		for (size_t j = 0; j < _n; ++j) if (get_digit_by_j(j) != 0u) return false;
		return true;
	}

	bool is_Mp() const
	{
		for (size_t j = 0; j < _n; ++j)
		{
			const uint64_t expected = (uint64_t(1) << _digit_width[j]) - 1;
			if (get_digit_by_j(j) != expected) return false;
		}
		return true;
	}
};

struct RunResult
{
	uint32_t p;
	size_t odd;
	size_t n;
	size_t storage;
	double elapsed;
	double it_s;
	bool prime;
};

static RunResult run_test(const uint32_t p, const size_t forced_odd, const std::string & label, const bool progress, const uint32_t max_iters)
{
	mersenne m(p, forced_odd);
	const size_t odd = m.odd_radix();
	const size_t n = m.transform_length();
	const size_t storage = m.storage_length();

	m.init(4);
	const auto t0 = std::chrono::steady_clock::now();
	auto last = t0;

	const uint32_t full_total = (p >= 2) ? (p - 2) : 0;
	const uint32_t total = ((max_iters != 0) && (max_iters < full_total)) ? max_iters : full_total;
	if (progress) std::cout << "[" << label << "] iter 0/" << full_total << " (0.00%), elapsed 0.00 s, it/s 0.00" << std::endl;
	for (uint32_t i = 0; i < total; ++i)
	{
		m.square();
		m.sub(2);

		if (progress)
		{
			const auto now = std::chrono::steady_clock::now();
			const double since_last = std::chrono::duration<double>(now - last).count();
			if ((since_last >= 1.0) || (i + 1 == total))
			{
				const double elapsed = std::chrono::duration<double>(now - t0).count();
				const double ips = double(i + 1) / elapsed;
				const double pct = (total != 0) ? 100.0 * double(i + 1) / double(total) : 100.0;
				std::cout << "[" << label << "] iter " << (i + 1) << "/" << total
					<< " (" << std::fixed << std::setprecision(2) << pct << "%)"
					<< ", elapsed " << std::setprecision(2) << elapsed << " s"
					<< ", it/s " << std::setprecision(2) << ips << std::endl;
				last = now;
			}
		}
	}

	const auto t1 = std::chrono::steady_clock::now();
	const double elapsed = std::chrono::duration<double>(t1 - t0).count();
	const double ips = (elapsed > 0.0) ? double(total) / elapsed : 0.0;
	const bool partial = (total != full_total);
	const bool prime = (!partial) && (m.is_zero() || m.is_Mp());

	if (partial) std::cout << "[" << label << "] partial run: " << total << "/" << full_total << " iterations";
	else std::cout << "[" << label << "] result: " << p << (prime ? " is prime" : " is composite");
	std::cout << ", elapsed " << std::fixed << std::setprecision(3) << elapsed << " s"
		<< ", it/s " << std::setprecision(2) << ips << std::endl;

	return RunResult{p, odd, n, storage, elapsed, ips, prime};
}

static void print_usage(const char * exe)
{
	std::cout << "Usage:\n"
		<< "  " << exe << " <p>                 auto choose best mixed size\n"
		<< "  " << exe << " <p> <odd>           force odd radix: 1,3,7,9,11,21,33,63,77,99\n"
		<< "  " << exe << " <p> --original      force original power-of-two size\n"
		<< "  " << exe << " <p> --compare       run auto then original and print perf\n"
		<< "  " << exe << " <p> --no-progress   disable progress lines\n"
		<< "  " << exe << " <p> --max-iters N   stop after N iterations for benchmarking\n";
}

int main(int argc, char ** argv)
{
	std::vector<uint32_t> tests;
	bool compare = false;
	bool progress = true;
	uint32_t max_iters = 0;
	size_t forced_odd = 0;

	if (argc > 1)
	{
		const std::string first = argv[1];
		if ((first == "--help") || (first == "-h")) { print_usage(argv[0]); return EXIT_SUCCESS; }
		tests.push_back(uint32_t(std::strtoul(argv[1], nullptr, 10)));
	}
	else tests = {3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127};

	for (int i = 2; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--compare") compare = true;
		else if ((arg == "--original") || (arg == "--pow2")) forced_odd = 1;
		else if (arg == "--auto") forced_odd = 0;
		else if (arg == "--no-progress") progress = false;
		else if (arg == "--progress") progress = true;
		else if (arg == "--max-iters")
		{
			if (i + 1 >= argc) throw std::runtime_error("--max-iters needs a value");
			max_iters = uint32_t(std::strtoul(argv[++i], nullptr, 10));
		}
		else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return EXIT_SUCCESS; }
		else forced_odd = size_t(std::strtoull(argv[i], nullptr, 10));
	}

	for (const uint32_t p : tests)
	{
		if (compare)
		{
			std::cout << "--- auto mixed mode ---" << std::endl;
			const RunResult a = run_test(p, 0, "auto", progress, max_iters);
			std::cout << "--- original power-of-two mode ---" << std::endl;
			const RunResult b = run_test(p, 1, "original", progress, max_iters);

			const double size_gain = double(b.storage) / double(a.storage);
			const double speed_gain = (b.it_s > 0.0) ? (a.it_s / b.it_s) : 0.0;
			std::cout << "comparison:" << std::endl;
			std::cout << "  auto     odd=" << a.odd << ", transform=" << a.n << ", storage=" << a.storage
				<< ", it/s=" << std::fixed << std::setprecision(2) << a.it_s << std::endl;
			std::cout << "  original odd=" << b.odd << ", transform=" << b.n << ", storage=" << b.storage
				<< ", it/s=" << std::fixed << std::setprecision(2) << b.it_s << std::endl;
			std::cout << "  storage gain original/auto = x" << std::fixed << std::setprecision(3) << size_gain << std::endl;
			std::cout << "  speed gain auto/original   = x" << std::fixed << std::setprecision(3) << speed_gain << std::endl;
		}
		else
		{
			const std::string label = (forced_odd == 1) ? "original" : ((forced_odd == 0) ? "auto" : ("odd" + std::to_string(forced_odd)));
			run_test(p, forced_odd, label, progress, max_iters);
		}
	}

	return EXIT_SUCCESS;
}
