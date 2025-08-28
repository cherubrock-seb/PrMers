/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <memory>
#include <chrono>
#include <sys/stat.h>

#include <gmp.h>

#include "engine.h"
#include "file.h"

class Mersenne
{
private:
	volatile bool _quit = false;
	uint32_t _display_i;
	std::string _ckpt_file;
	const size_t R0 = 0, R1 = 1, R2 = 2;

private:
	struct deleter { void operator()(const Mersenne * const p) { delete p; } };

	static std::string get_string(const uint64_t u)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u;
		return ss.str();
	}

	static std::string format_time(const double time)
	{
		uint64_t seconds = uint64_t(time), minutes = seconds / 60, hours = minutes / 60;
		seconds -= minutes * 60; minutes -= hours * 60;

		std::stringstream ss;
		ss << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
		return ss.str();
	}

	static void clearline() { std::cout << "                                                            \r"; }

	uint32 display_progress(const uint32 i, const double percent, const double display_time, const double elapsed_time)
	{
		if (_display_i == i) return 1;
		const double iter_time = display_time / (i - _display_i); _display_i = i;
		const uint32 count = std::max(uint32(1.0 / iter_time), 2u);
		const double expected_time = elapsed_time / percent, remaining_time = std::max(expected_time - elapsed_time, 0.0);
		if ((i > 1) && (display_time > 0.5))
		{
			std::ostringstream ss; ss << std::setprecision(3) << percent * 100.0 << "% done, " << format_time(remaining_time)
				<< "/" << format_time(expected_time) << " remaining, " << elapsed_time / i * 1e3 << " ms/iter.        \r";
			std::cout << ss.str();
		}
		return count;
	}

	void set_checkpoint_filename(const uint32_t p)
	{
		std::ostringstream ss; ss << "m_" << p << ".ckpt";
		_ckpt_file = ss.str();
	}

	int _read_checkpoint(engine * const eng, const uint32_t p, const std::string & filename, uint32 & i, double & elapsed_time)
	{
		File file(filename);
		if (!file.exists()) return -1;

		int version = 0;
		if (!file.read(reinterpret_cast<char *>(&version), sizeof(version))) return -2;
		if (version != 1) return -2;
		uint32_t rp = 0;
		if (!file.read(reinterpret_cast<char *>(&rp), sizeof(rp))) return -2;
		if (rp != p) return -2;
		if (!file.read(reinterpret_cast<char *>(&i), sizeof(i))) return -2;
		if (!file.read(reinterpret_cast<char *>(&elapsed_time), sizeof(elapsed_time))) return -2;
		const size_t checkpoint_size = eng->get_checkpoint_size();
		std::vector<char> data(checkpoint_size);
		if (!file.read(data.data(), checkpoint_size)) return -2;
		if (!eng->set_checkpoint(data)) return -2;
		if (!file.check_crc32()) return -2;
		return 0;
	}

	bool read_checkpoint(engine * const eng, const uint32_t p, uint32 & i, double & elapsed_time)
	{
		std::string ctx_file = _ckpt_file;
		int error = _read_checkpoint(eng, p, ctx_file, i, elapsed_time);
		if (error < -1)
		{
			std::ostringstream ss; ss << ctx_file << ": invalid checkpoint.";
			std::cout << ss.str() << std::endl;
		}
		ctx_file += ".old";
		if (error < 0)
		{
			error = _read_checkpoint(eng, p, ctx_file, i, elapsed_time);
			if (error < -1)
			{
				std::ostringstream ss; ss << ctx_file << ": invalid checkpoint.";
				std::cout << ss.str() << std::endl;
			}
		}
		return (error == 0);
	}

	void save_checkpoint(engine * const eng, const uint32_t p, const uint32 i, const double elapsed_time) const
	{
		std::cout << std::endl << "Saving checkpoint..." << std::endl;
		const std::string old_ckpt_file = _ckpt_file + ".old", new_ckpt_file = _ckpt_file + ".new";

		{
			File file(new_ckpt_file, "wb");
			int version = 1;
			if (!file.write(reinterpret_cast<const char *>(&version), sizeof(version))) return;
			if (!file.write(reinterpret_cast<const char *>(&p), sizeof(p))) return;
			if (!file.write(reinterpret_cast<const char *>(&i), sizeof(i))) return;
			if (!file.write(reinterpret_cast<const char *>(&elapsed_time), sizeof(elapsed_time))) return;
			const size_t checkpoint_size = eng->get_checkpoint_size();
			std::vector<char> data(checkpoint_size);
			if (!eng->get_checkpoint(data)) return;
			if (!file.write(data.data(), checkpoint_size)) return;
			file.write_crc32();
		}

		std::remove(old_ckpt_file.c_str());

		struct stat s;
		if ((stat(_ckpt_file.c_str(), &s) == 0) && (std::rename(_ckpt_file.c_str(), old_ckpt_file.c_str()) != 0))	// file exists and cannot rename it
		{
			std::cout << "Error: cannot save checkpoint." << std::endl;
			return;
		}

		if (std::rename(new_ckpt_file.c_str(), _ckpt_file.c_str()) != 0)
		{
			std::cout << "Error: cannot save checkpoint." << std::endl;
			return;
		}
	}

public:
	Mersenne() {}
	~Mersenne() {}

	static Mersenne & get_instance()
	{
		static std::unique_ptr<Mersenne, deleter> instance(new Mersenne());
		return *instance;
	}

	void quit() { _quit = true; }

	bool check(const uint32_t p, const size_t device, const bool verbose = true, const bool test_GL = false)
	{
		engine * const eng =
#if defined(GPU)
			engine::create_gpu(p, 3, device, verbose);	// 3 registers
#else
			engine::create_cpu(p);
			(void)device;
#endif
		if (verbose) std::cout << "Testing 2^" << p << " - 1, " << eng->get_size() << " 64-bit words..." << std::endl;

		set_checkpoint_filename(p);
		uint32_t ri = 0; double restored_time = 0;
		const bool found = read_checkpoint(eng, p, ri, restored_time);
		if (!found)
		{
			ri = 0; restored_time = 0;
			eng->set(R0, 1);	// result = 1
			eng->set(R1, 1);	// d(t) = 1
		}
		else
		{
			std::cout << "Resuming from a checkpoint." << std::endl;
		}

		// Gerbicz-Li error checking
		const uint32_t B_GL = std::max(uint32_t(std::sqrt(p)), 2u);

		const auto start_clock = std::chrono::high_resolution_clock::now();
		uint32_t display_count = 2, display_count_reset = display_count;
		auto display_clock = start_clock;
		_display_i = ri;

		// 3-prp test, left-to-right binary exponentiation
		for (uint32_t i = ri, j = p - 1 - i; i < p; ++i, --j)
		{
			if (_quit)
			{
				const auto now_clock = std::chrono::high_resolution_clock::now();
				const double elapsed_time = std::chrono::duration<double>(now_clock - start_clock).count() + restored_time;
				save_checkpoint(eng, p, i, elapsed_time);
				delete eng;
				return false;
			}

			if (verbose && (--display_count == 0))
			{
				const auto now_clock = std::chrono::high_resolution_clock::now();
				const double display_time = std::chrono::duration<double>(now_clock - display_clock).count();
				if (display_time < 1) display_count = display_count_reset;
				else
				{
					const double elapsed_time = std::chrono::duration<double>(now_clock - start_clock).count() + restored_time;
					display_clock = now_clock;
					display_count_reset = display_count = display_progress(i, i / double(p), display_time, elapsed_time);
				}
			}

			eng->square_mul(R0, (j != 0) ? 3 : 1);
			if ((j == 0) && test_GL) eng->error();	// test Gerbicz-Li

			if ((j % B_GL == 0) && (j != 0))
			{
				eng->set_multiplicand(R2, R0);
				eng->mul(R1, R2);	// d(t + 1) = d(t) * result
			}
		}

		// Probable prime?
		std::vector<uint64> d(eng->get_size());
		eng->get(d.data(), R0);
		uint64_t res64 = 0; const bool is_prp = eng->is_one(d, res64);
		d.clear();

		// d(t + 1) = d(t) * result
		eng->set_multiplicand(R2, R1);
		eng->mul(R0, R2);

		// d(t)^{2^B}
		for (uint32_t i = 0; i < B_GL; ++i)
		{
			eng->square_mul(R1);
			if (_quit) { delete eng; return false; }
		}

		// Exponent of double check process
		// See: An Efficient Modular Exponentiation Proof Scheme, §2, Darren Li, Yves Gallot, https://arxiv.org/abs/2209.15623

		mpz_t res, t;
		mpz_init_set_ui(res, p / B_GL); mpz_mul_2exp(res, res, B_GL);
		mpz_init_set_ui(t, 1); mpz_mul_2exp(t, t, p % B_GL); 
		mpz_add(res, res, t); mpz_sub_ui(res, res, p / B_GL + 2);
		mpz_clear(t);

		// 3^res
		eng->set(R2, 1);
		for (uint32_t i = uint32_t(mpz_sizeinbase(res, 2)); i > 0; --i)
		{
			eng->square_mul(R2, (mpz_tstbit(res, i - 1) != 0) ? 3 : 1);
			if (_quit) { delete eng; return false; }
		}

		mpz_clear(res);

		// d(t)^{2^B} * 3^res
		eng->set_multiplicand(R2, R2);
		eng->mul(R1, R2);

		if (verbose) clearline();

		// d(t + 1) = d(t)^{2^B} * 3^res?
		if (!eng->is_equal(R0, R1)) throw std::runtime_error("Gerbicz-Li error checking failed!");

		if (verbose)
		{
			const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
			std::cout << "2^" << p << " - 1 is ";
			if (is_prp) std::cout << "a probable prime";
			else std::cout << "composite, res64 = " << get_string(res64);
			std::cout << ", time = " << format_time(elapsed_time) << "." << std::endl << std::endl;
		}
		else
		{
			if (is_prp) std::cout << "2^" << p << " - 1" << " (" << eng->get_size() << ")" << std::endl;
		}

		delete eng;
		return true;
	}

	bool valid(const uint32_t p, const size_t device)
	{
		engine * const eng =
#if defined(GPU)
			engine::create_gpu(p, 3, device, true);	// 3 registers
#else
			engine::create_cpu(p);
			(void)device;
#endif

		eng->set(R0, 1);	// result = 1
		eng->set(R1, 1);	// d(t) = 1

		// Gerbicz-Li error checking
		const uint32_t B_GL = 7;

		// 3^exponent, left-to-right binary exponentiation
		const uint32_t exponent = 1234567809u;
		for (uint32_t i = 0, j = 30; i < 31; ++i, --j)
		{
			if (_quit) { delete eng; return false; }

			const uint32_t b_j = (exponent >> j) & 1;
			eng->square_mul(R0, (b_j != 0) ? 3 : 1);

			if ((j % B_GL == 0) && (j != 0))
			{
				eng->set_multiplicand(R2, R0);
				eng->mul(R1, R2);	// d(t + 1) = d(t) * result
			}
		}

		// d(t + 1) = d(t) * result
		eng->set_multiplicand(R2, R1);
		eng->mul(R0, R2);

		// d(t)^{2^B}
		for (uint32_t i = 0; i < B_GL; ++i)
		{
			eng->square_mul(R1);
			if (_quit) { delete eng; return false; }
		}

		const uint32_t res = 174;

		// 3^res
		eng->set(R2, 1);
		for (uint32_t i = 8; i > 0; --i)
		{
			const uint32_t b = (res >> (i - 1)) & 1;
			eng->square_mul(R2, (b != 0) ? 3 : 1);
			if (_quit) { delete eng; return false; }
		}

		// d(t)^{2^B} * 3^res
		eng->set_multiplicand(R2, R2);
		eng->mul(R1, R2);

		// d(t + 1) = d(t)^{2^B} * 3^res?
		if (!eng->is_equal(R0, R1)) throw std::runtime_error("Gerbicz-Li error checking failed!");

		delete eng;
		return true;
	}

	void valid_gpu(const size_t device)
	{
		static const uint32_t prm[] = { 113, 239, 463, 919, 1153, 1789, 2239, 3583, 4463, 6911, 8629, 13807, 17257, 26597, 33247,
			53239, 66553, 102397, 127997, 204797, 255989, 393209, 491503, 786431, 982981, 1507321, 1884133, 3014653, 3768311,
			5767129, 7208951, 11534329, 14417881, 22020091, 27525109, 44040187, 55050217, 83886053, 104857589, 167772107,
			209715199, 318767093, 398458859, 637534199, 796917757, 1207959503, 1509949421 };

		for (size_t i = 0; i < sizeof(prm) / sizeof(uint32_t); ++i)
		{
			const uint32_t p = prm[i];
			if (i != 0) std::cout << ", " << std::flush;
			if (!valid(p, device)) return;
			else
			{
				engine * const eng = new engine_info(p);
				const size_t n = eng->get_size();
				delete eng;
 				std::cout << prm[i] << " (" << n << ")";
			}
		}

		std::cout << std::endl << "OK!" << std::endl;
	}

	void test(const size_t device)
	{
#if defined(GPU)
		// min sizes: 3, 127, 241, 467, 929, 1163, 1801, 2243, 3593, 4481, 6917, 8641, 13829, 17291, 26627,
		// 33287, 53267, 66569, 102407, 128021, 204803, 256019, 393241, 491527, 786433, 983063, 1507369, 1884193, 3014659,
		// 3768341, 5767169, 7208977, 11534351, 14417927, 22020127, 27525131, 44040253, 55050253, 83886091, 104857601,
		// 167772161, 209715263, 318767107, 398458889, 637534277, 796917763, 1207959559

		static const uint32_t prm[] = { 3, 127, 401, 521, 1009, 1279, 2203, 2281, 4253, 5003, 7001, 9689, 14009, 19937, 30011,
			44497, 60013, 86243, 110503, 132049, 216091, 300007, 400009, 756839, 859433, 1257787, 1600033, 2976221, 3021377,
			4000037, 6972593, 8000009, 13466917, 20996011, 24036583, 30402457, 50000017, 57885161, 90000000, 136279841,
			167772161, 209715263, 318767107, 398458889, 637534277, 796917763, 1207959559, 1509949421 };

		for (size_t i = 0; i < sizeof(prm) / sizeof(uint32_t); ++i)
		{
			if (!check(prm[i], device)) return;
		}
#else
		// 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209,
		// 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583,
		// 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933, 136279841

		// If p > 1509949439 then size is 2^27 and (p - 1) / 192 = 2^26 * 5 * 17 * 257. If p <= 1509949439 then transform_size <= 83886080
		for (uint32_t p = 3; p <= 1509949439u; p += 2)
		{
			bool isprime = true;
			for (uint32_t d = 3; p / d >= d; d += 2) if (p % d == 0) { isprime = false; break; }
			if (!isprime) continue;
			if (!check(p, device, false, p == 11239)) return;
		}
#endif
	}

	static bool isprime(const uint32 p)
	{
		for (uint32_t d = 3; p / d >= d; d += 2) if (p % d == 0) return false;
		return true;
	}

	static void display_info(const uint32_t p)
	{
		engine * const eng = new engine_info(p);
		const size_t n = eng->get_size();
		const bool even = eng->get_even();
		// const int ln_max = eng->get_ln_max();
		delete eng;

		const size_t sqr_size = (n % 5 == 0) ? 10 : 4;
		size_t r = n / sqr_size;
		const uint32_t n5 = uint32_t((n % 5 == 0) ? n / 5 : n);
		const int lcwm_wg_size = ilog2_32(std::min(n5 / (even ? 4 : 8), 1024u));

		std::cout << p << ", " << n << " = ";
		std::cout << sqr_size << " * ";
		if (!even) { r /= 2; std::cout << "2 * "; }
		std::cout << "4^" << ilog2_32(r) / 2;
		// std::cout << " (ln_max = " << ln_max << ")";
		std::cout << ", " << ((n / 4) >> lcwm_wg_size) << " * 2^" << lcwm_wg_size << std::endl;
	}

	void info() const
	{
		size_t n2 = 4, n5 = 40, next_n = n2;

		while (next_n <= (size_t(1) << 27))
		{
			uint64_t p_min = 3, p_max = 3;
			size_t n_max = next_n;
			while (n_max <= next_n)
			{
				p_max = 2 * p_max + 1;
				engine * const eng = new engine_info(p_max);
				n_max = eng->get_size();
				delete eng;
			}

			while (p_max - p_min > 2)
			{
				uint64_t p = (p_min + p_max) / 2; if (p % 2 == 0) --p;
				engine * const eng = new engine_info(p);
				const size_t n = eng->get_size();
				delete eng;
				if (n < next_n) p_min = p; else p_max = p;
			}

			engine * const eng = new engine_info(p_min);
			const size_t n_min = eng->get_size();
			delete eng;
	
			if (next_n != 4)
			{
				uint64_t p = (n_min == next_n) ? p_min : p_max; p -= 2; while (!isprime(p)) p -= 2;
				display_info(p);
			}
			if (next_n != (size_t(1) << 27))
			{
				uint64_t p = (n_min == next_n) ? p_min : p_max; while (!isprime(p)) p += 2;
				display_info(p);
			}

			if (next_n == n2) n2 *= 2; else n5 *= 2;
			next_n = std::min(n2, n5);
		}
	}
};