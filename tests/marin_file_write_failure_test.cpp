/*
 * Host test for marin::File failure handling (no GPU needed).
 *
 * A checkpoint that cannot be created (missing or read-only directory) used to
 * crash File::write in fwrite(nullptr).  Writers now see false from every call
 * and from close(), and a file whose write failed is removed rather than left
 * truncated.  Usage: marin_file_write_failure_test <scratch directory>
 */

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include "marin/file.h"

static int failures = 0;

static void check(const bool ok, const char * const what)
{
	std::cout << (ok ? "ok   " : "FAIL ") << what << std::endl;
	if (!ok) ++failures;
}

static bool exists_on_disk(const std::string & path)
{
	struct stat st;
	return stat(path.c_str(), &st) == 0;
}

int main(int argc, char ** argv)
{
	if (argc != 2) { std::cerr << "usage: " << argv[0] << " <scratch dir>" << std::endl; return 2; }
	const std::string dir = argv[1];
	const int v = 12345;
	const char * const pv = reinterpret_cast<const char *>(&v);

	// Missing directory: every call must fail cleanly.
	{
		File f(dir + "/no/such/dir/x.ckpt.new", "wb");
		check(!f.exists(), "missing directory: open fails");
		check(!f.write(pv, sizeof(v)), "missing directory: write returns false");
		check(!f.write_crc32(), "missing directory: write_crc32 returns false");
		check(!f.close(), "missing directory: close returns false");
	}

	// Read-only directory (skipped when the process may write anyway, e.g. root).
	const std::string rodir = dir + "/ro";
	mkdir(rodir.c_str(), 0555);
	if (access(rodir.c_str(), W_OK) != 0)
	{
		File f(rodir + "/x.ckpt.new", "wb");
		check(!f.exists(), "read-only directory: open fails");
		check(!f.write(pv, sizeof(v)), "read-only directory: write returns false");
		check(!f.write_crc32(), "read-only directory: write_crc32 returns false");
		check(!f.close(), "read-only directory: close returns false");
	}
	else std::cout << "skip read-only directory (directory is writable)" << std::endl;

	// A reader on a missing file must not crash either.
	{
		File f(dir + "/no/such/file");
		int x = 0;
		check(!f.exists(), "missing file: not found");
		check(!f.read(reinterpret_cast<char *>(&x), sizeof(x)), "missing file: read returns false");
		check(!f.check_crc32(), "missing file: check_crc32 returns false");
	}

	// Successful round trip with CRC.
	const std::string good = dir + "/good.ckpt";
	{
		File f(good, "wb");
		check(f.exists(), "round trip: open for write");
		check(f.write(pv, sizeof(v)) && f.write_crc32(), "round trip: write + crc");
		check(f.close(), "round trip: close returns true");
		check(!f.exists(), "round trip: closed file no longer exists()");
		check(f.close(), "round trip: closing twice is harmless");
	}
	{
		File f(good);
		int x = 0;
		check(f.exists() && f.read(reinterpret_cast<char *>(&x), sizeof(x)) && x == v, "round trip: read back");
		check(f.check_crc32(), "round trip: crc matches");
	}

	// Truncated file: the CRC check must fail without crashing.
	{
		const std::string trunc = dir + "/trunc.ckpt";
		{ File f(trunc, "wb"); f.write(pv, sizeof(v)); f.close(); }
		File f(trunc);
		int x = 0;
		check(f.read(reinterpret_cast<char *>(&x), sizeof(x)) && !f.check_crc32(), "truncated: crc check fails");
	}

	// Write failure on a regular file (file-size limit): the failure is reported
	// and the partial file is removed.  /dev/full is deliberately not used,
	// since it must never be removed.
	{
		std::signal(SIGXFSZ, SIG_IGN);
		struct rlimit old_limit;
		getrlimit(RLIMIT_FSIZE, &old_limit);
		struct rlimit lim = old_limit;
		lim.rlim_cur = 1000;
		setrlimit(RLIMIT_FSIZE, &lim);
		const std::string big = dir + "/big.ckpt.new";
		bool wrote = true, closed = true;
		{
			File f(big, "wb");
			std::vector<char> data(100000, 'x');
			wrote = f.write(data.data(), data.size());
			closed = f.close();
		}
		setrlimit(RLIMIT_FSIZE, &old_limit);
		check(!(wrote && closed), "file size limit: write or close reports failure");
		check(!exists_on_disk(big), "file size limit: partial file is removed");
	}

	if (failures == 0) std::cout << "All File failure tests passed." << std::endl;
	return failures == 0 ? 0 : 1;
}
