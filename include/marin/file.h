/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>

class File
{
private:
	const std::string _filename;
	FILE * const _cfile;
	uint32_t _crc32 = 0;

public:
	File(const std::string & filename, const char * const mode)
		: _filename(filename), _cfile(std::fopen(filename.c_str(), mode)), _crc32(0)
	{
		if (_cfile == nullptr) std::cout << std::endl << "Cannot open file: "<< filename << "." << std::endl;
	}

	File(const std::string & filename)
		: _filename(filename), _cfile(std::fopen(filename.c_str(), "rb")), _crc32(0)
	{
		// _cfile may be null
	}

	virtual ~File()
	{
		if (_cfile != nullptr)
		{
			if (std::fclose(_cfile) != 0) std::cout << std::endl << "Cannot close file." << std::endl;
		}
	}

	bool exists() const { return (_cfile != nullptr); }

	uint32_t crc32() const { return _crc32; }

	// Rosetta Code, CRC-32, C
	static uint32_t rc_crc32(const uint32_t crc32, const char * const buf, const size_t len)
	{
		static uint32_t table[256];
		static bool have_table = false;
	
		// This check is not thread safe; there is no mutex
		if (!have_table)
		{
			// Calculate CRC table
			for (size_t i = 0; i < 256; ++i)
			{
				uint32_t rem = uint32_t(i);  // remainder from polynomial division
				for (size_t j = 0; j < 8; ++j)
				{
					if (rem & 1)
					{
						rem >>= 1;
						rem ^= 0xedb88320;
					}
					else rem >>= 1;
				}
				table[i] = rem;
			}
			have_table = true;
		}

		uint32_t crc = ~crc32;
		for (size_t i = 0; i < len; ++i)
		{
			const uint8_t octet = uint8_t(buf[i]);  // cast to unsigned octet
			crc = (crc >> 8) ^ table[(crc & 0xff) ^ octet];
		}
		return ~crc;
	}

	bool read(char * const ptr, const size_t size)
	{
		const size_t ret = std::fread(ptr , sizeof(char), size, _cfile);
		_crc32 = rc_crc32(_crc32, ptr, size);
		return (ret == size * sizeof(char));
	}

	bool write(const char * const ptr, const size_t size)
	{
		const size_t ret = std::fwrite(ptr , sizeof(char), size, _cfile);
		_crc32 = rc_crc32(_crc32, ptr, size);
		return (ret == size * sizeof(char));
	}

	void write_crc32()
	{
		uint32_t crc32 = ~_crc32 ^ 0xa23777ac;
		write(reinterpret_cast<const char *>(&crc32), sizeof(crc32));
	}

	bool check_crc32()
	{
		uint32_t crc32 = 0, ocrc32 = ~_crc32 ^ 0xa23777ac;	// before the read operation
		read(reinterpret_cast<char *>(&crc32), sizeof(crc32));
		const bool success = (crc32 == ocrc32);
		if (!success) std::cout << std::endl << "Bad file (crc32)." << std::endl;
		return success;
	}
};
