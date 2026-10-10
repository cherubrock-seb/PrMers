/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <iomanip>

#include <sys/stat.h>

// A File that could not be opened is safe to use: read(), write() and the CRC
// helpers return false instead of touching a null stream.  A file opened for
// writing ("w...") must be finished with close(), which returns true only when
// every write, the flush and the close succeeded; a file whose write or close
// failed is removed by close()/the destructor so that a truncated file is never
// left behind to be renamed over a good one.
class File
{
private:
	const std::string _filename;
	FILE * _cfile;
	uint32_t _crc32 = 0;
	const bool _truncating;	// opened with "w": a failed file is removed
	bool _failed = false;

public:
	File(const std::string & filename, const char * const mode)
		: _filename(filename), _cfile(std::fopen(filename.c_str(), mode)), _crc32(0), _truncating(mode[0] == 'w')
	{
		if (_cfile == nullptr)
		{
			_failed = true;
			std::cout << std::endl << "Cannot open file: "<< filename << ": " << std::strerror(errno) << "." << std::endl;
		}
	}

	File(const std::string & filename)
		: _filename(filename), _cfile(std::fopen(filename.c_str(), "rb")), _crc32(0), _truncating(false)
	{
		// _cfile may be null
	}

	File(const File &) = delete;
	File & operator=(const File &) = delete;

	virtual ~File() { close(); }

	// Flushes and closes the file.  Returns true if the file was opened and no
	// write, flush or close failed.  A failed file opened with "w" is removed.
	// The file is closed afterwards (exists() is false); closing twice is harmless.
	bool close()
	{
		if (_cfile != nullptr)
		{
			if (std::fflush(_cfile) != 0 || std::ferror(_cfile) != 0) _failed = true;
			if (std::fclose(_cfile) != 0)
			{
				_failed = true;
				std::cout << std::endl << "Cannot close file: " << _filename << "." << std::endl;
			}
			_cfile = nullptr;
			struct stat st;
			if (_failed && _truncating && stat(_filename.c_str(), &st) == 0)
			{
#ifdef _WIN32
				const bool regular_file = (st.st_mode & _S_IFMT) == _S_IFREG;
#else
				const bool regular_file = S_ISREG(st.st_mode);
#endif
				if (regular_file) std::remove(_filename.c_str());
			}
		}
		return !_failed;
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
		if (_cfile == nullptr) return false;
		const size_t ret = std::fread(ptr , sizeof(char), size, _cfile);
		_crc32 = rc_crc32(_crc32, ptr, size);
		return (ret == size * sizeof(char));
	}

	bool write(const char * const ptr, const size_t size)
	{
		if (_cfile == nullptr) { _failed = true; return false; }
		const size_t ret = std::fwrite(ptr , sizeof(char), size, _cfile);
		_crc32 = rc_crc32(_crc32, ptr, size);
		const bool success = (ret == size * sizeof(char));
		if (!success) _failed = true;
		return success;
	}

	bool write_crc32()
	{
		uint32_t crc32 = ~_crc32 ^ 0xa23777ac;
		return write(reinterpret_cast<const char *>(&crc32), sizeof(crc32));
	}

	bool check_crc32()
	{
		uint32_t crc32 = 0, ocrc32 = ~_crc32 ^ 0xa23777ac;	// before the read operation
		const bool read_ok = read(reinterpret_cast<char *>(&crc32), sizeof(crc32));
		const bool success = read_ok && (crc32 == ocrc32);
		if (!success) std::cout << std::endl << "Bad file (crc32)." << std::endl;
		return success;
	}
};