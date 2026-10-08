// Host test (no OpenCL device needed): the legacy transform size and legacy checkpoint loading.
//
// 1. math::transformsize(): every convolution coefficient must stay below the NTT prime. Digits are below
//    2^(w + 1) (w = floor(p / n)), the IBDWT weight contributes a factor 1 or 2 to each product and n products
//    are summed, so the largest coefficient is n * 2 * (2^(w + 1) - 1)^2 and must be < MOD_P = 2^64 - 2^32 + 1.
//    The exponents pinned below sat in the top of a radix-5 range and used to select a size where it was not.
// 2. core::BackupManager: a legacy checkpoint is a raw image of the n-word digit vector, so a state file of any
//    other length (written for another transform size, truncated or extended) must be rejected, not misread.
#include "core/BackupManager.hpp"
#include "math/Precompute.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int fails = 0;

constexpr unsigned __int128 MODP = (unsigned __int128)0xFFFFFFFF00000001ull;

bool valid(const uint64_t n, const uint64_t p)
{
	const unsigned __int128 w1 = p / n + 1;
	if (w1 > 40) return false;
	const unsigned __int128 d = ((unsigned __int128)1 << w1) - 1;
	return (unsigned __int128)n * 2 * d * d < MODP;
}

void check_size(const uint64_t p)
{
	const uint64_t n = math::transformsize(p);
	if (!valid(n, p)) { ++fails; std::printf("FAIL p=%llu: transformsize=%llu admits a coefficient >= MOD_P\n", (unsigned long long)p, (unsigned long long)n); }
}

void expect_size(const uint64_t p, const uint64_t n)
{
	const uint64_t got = math::transformsize(p);
	if (got != n) { ++fails; std::printf("FAIL p=%llu: transformsize=%llu, expected %llu\n", (unsigned long long)p, (unsigned long long)got, (unsigned long long)n); }
}

void expect(const bool ok, const std::string & what)
{
	if (!ok) { ++fails; std::printf("FAIL %s\n", what.c_str()); }
}

void write_words(const fs::path & f, const size_t words, const uint64_t fill)
{
	std::ofstream o(f, std::ios::binary);
	std::vector<uint64_t> v(words, fill);
	o.write(reinterpret_cast<const char *>(v.data()), std::streamsize(words * sizeof(uint64_t)));
}

void write_text(const fs::path & f, const std::string & s) { std::ofstream o(f); o << s; }

void write_bytes(const fs::path & f, const size_t bytes)
{
	std::ofstream o(f, std::ios::binary);
	const std::string z(bytes, '\x5a');
	o.write(z.data(), std::streamsize(bytes));
}

core::BackupManager make(const fs::path & dir, const size_t n, const char * mode = "prp", const uint64_t b1 = 0, const uint64_t b2 = 0)
{
	return core::BackupManager(nullptr, 60, n, dir.string(), 4451u, mode, b1, b2, false, false);
}

void test_sizes()
{
	// the exponents the old radix-5 bound let wrap now take the next power of two
	struct { uint64_t p, n; } wrap[] = {
		{ 4451, 256 }, { 4463, 256 }, { 17153, 1024 }, { 17159, 1024 }, { 66049, 4096 }, { 66071, 4096 },
		{ 4423, 256 }, { 16651, 1024 }, { 65537, 4096 } };	// the last three happened to be exact but fail the bound
	for (const auto & e : wrap) { expect_size(e.p, e.n); check_size(e.p - 1); check_size(e.p + 1); }

	// the largest exponents that still fit radix-5 sizes keep them
	struct { uint64_t p, n; } keep[] = { { 4319, 160 }, { 16639, 640 }, { 63999, 2560 }, { 245759, 10240 }, { 199229439, 10485760 } };
	for (const auto & e : keep) { expect_size(e.p, e.n); if (math::transformsize(e.p + 1) == e.n) { ++fails; std::printf("FAIL p=%llu must not keep n=%llu\n", (unsigned long long)(e.p + 1), (unsigned long long)e.n); } }

	// dense sweep of the small range, a sparse sweep up to the largest exponents, and the neighbourhood of every
	// radix-5 digit-width boundary p = 5 * 2^k * w
	for (uint64_t p = 5; p < 2000000; ++p) check_size(p);
	for (uint64_t p = 2000000; p < 1300000000ull; p += 99991) check_size(p);
	for (uint64_t k = 4; k <= 26; ++k)
		for (uint64_t w = 1; w <= 40; ++w)
			for (int d = -3; d <= 3; ++d) {
				const uint64_t p = (5ull << k) * w + d;
				if (p > 4) check_size(p);
			}
}

void test_checkpoints()
{
	const fs::path root = fs::temp_directory_path() / ("prmers-legacy-ckpt-" + std::to_string(::getpid()));
	fs::remove_all(root);
	fs::create_directories(root);

	const size_t n = 256;	// the size p = 4451 now uses
	const std::string base = "4451prp";

	// a correct-size state resumes
	{
		const fs::path d = root / "ok"; fs::create_directories(d);
		auto b = make(d, n);
		write_words(d / (base + ".mers"), n, 0x1234);
		write_text(d / (base + ".loop"), "77");
		std::vector<uint64_t> x(n, 0);
		expect(b.loadState(x) == 77 && x[0] == 0x1234 && x[n - 1] == 0x1234, "exact-size state resumes");
	}
	// every wrong size is rejected and x is the fresh start state: the old n (160 words), one word short, one word
	// long, a byte short, a byte long, a partial word, double size, empty file
	{
		const size_t bytes[] = { 160 * 8, n * 8 - 8, n * 8 + 8, n * 8 - 1, n * 8 + 1, 3, n * 16, 0 };
		int i = 0;
		for (const size_t sz : bytes) {
			const fs::path d = root / ("bad" + std::to_string(i++)); fs::create_directories(d);
			auto b = make(d, n);
			write_bytes(d / (base + ".mers"), sz);
			write_text(d / (base + ".loop"), "77");
			std::vector<uint64_t> x(n, 9);
			const uint64_t r = b.loadState(x);
			expect(r == 0 && x[0] == 3 && x[1] == 0 && x[n - 1] == 0, "wrong-size state rejected, bytes=" + std::to_string(sz));
		}
	}
	// loop file contents that are not a valid resume point, leading zeros/whitespace, 2^32 boundary values
	{
		struct { const char * text; uint64_t want; } loops[] = {
			{ "", 0 }, { "0", 0 }, { "garbage", 0 }, { "  12  ", 12 }, { "0012", 12 }, { "4294967295", 4294967295ull },
			{ "4294967296", 4294967296ull }, { "18446744073709551616", 0 } };
		int i = 0;
		for (const auto & l : loops) {
			const fs::path d = root / ("loop" + std::to_string(i++)); fs::create_directories(d);
			auto b = make(d, n);
			write_words(d / (base + ".mers"), n, 5);
			write_text(d / (base + ".loop"), l.text);
			std::vector<uint64_t> x(n, 9);
			const uint64_t r = b.loadState(x);
			expect(r == l.want, std::string("loop file \"") + l.text + "\" resume=" + std::to_string(r));
			if (r == 0) expect(x[0] == 3 && x[1] == 0, "fresh state when the loop file is unusable");
		}
		const fs::path d = root / "nomers"; fs::create_directories(d);
		auto b = make(d, n);
		write_text(d / (base + ".loop"), "5");
		std::vector<uint64_t> x(n, 9);
		expect(b.loadState(x) == 0 && x[0] == 3, "missing state file rejected");
	}
	// a state "file" that is a directory is rejected, not misread
	{
		const fs::path d = root / "dir"; fs::create_directories(d);
		auto b = make(d, n);
		fs::create_directories(d / (base + ".mers"));
		write_text(d / (base + ".loop"), "5");
		std::vector<uint64_t> x(n, 9);
		expect(b.loadState(x) == 0 && x[0] == 3, "directory as state file rejected");
	}
	// Gerbicz-Li side files: wrong size -> defaults, right size -> loaded; counters of a rejected state are ignored
	{
		const fs::path d = root / "gl"; fs::create_directories(d);
		auto b = make(d, n);
		write_words(d / (base + ".bufd"), 160, 7);
		write_words(d / (base + ".gli"), n + 1, 7);
		write_words(d / (base + ".lbufd"), n, 7);
		write_text(d / (base + ".isav"), "100");
		write_text(d / (base + ".jsav"), "200");
		std::vector<uint64_t> a(n, 9), c(n, 9), l(n, 9);
		b.loadGerbiczLiBufDState(a); b.loadGerbiczLiCorrectState(c); b.loadGerbiczLiCorrectBufDState(l);
		expect(a[0] == 1 && a[1] == 0 && a[n - 1] == 0, "short bufd rejected");
		expect(c[0] == 3 && c[1] == 0 && c[n - 1] == 0, "long gli rejected");
		expect(l[0] == 7 && l[n - 1] == 7, "exact lbufd loaded");
		expect(b.loadGerbiczIterSave() == 100 && b.loadGerbiczJSave() == 200, "counters read while the state is usable");
		write_bytes(d / (base + ".mers"), 160 * 8);
		write_text(d / (base + ".loop"), "5");
		std::vector<uint64_t> x(n, 0);
		expect(b.loadState(x) == 0, "state of the old size rejected");
		expect(b.loadGerbiczIterSave() == 0 && b.loadGerbiczJSave() == 0, "counters ignored after a rejected state");
	}
	// P-1 stage-2 buffers: wrong size or a missing file -> restart stage 2 from the beginning
	{
		const std::string b2base = "4451pm1100_200";
		struct { size_t hq, q; bool hq_present, q_present, ok; } cases[] = {
			{ n * 8, n * 8, true, true, true }, { 160 * 8, n * 8, true, true, false }, { n * 8, 160 * 8, true, true, false },
			{ n * 8 + 8, n * 8, true, true, false }, { n * 8, 0, true, true, false },
			{ 0, n * 8, false, true, false }, { n * 8, 0, true, false, false } };
		int i = 0;
		for (const auto & c : cases) {
			const fs::path d = root / ("s2_" + std::to_string(i++)); fs::create_directories(d);
			auto b = make(d, n, "pm1", 100, 200);
			if (c.hq_present) write_bytes(d / (b2base + ".hq"), c.hq);
			if (c.q_present) write_bytes(d / (b2base + ".q"), c.q);
			write_text(d / (b2base + ".loop2"), "42");
			// a null queue makes clEnqueueWriteBuffer fail harmlessly: only the resume decision is checked
			const uint64_t r = b.loadStatePM1S2(nullptr, nullptr, n * 8);
			expect(c.ok ? r == 42 : r == 0, "stage-2 buffers case " + std::to_string(i - 1));
		}
	}

	fs::remove_all(root);
}

}	// namespace

int main()
{
	test_sizes();
	test_checkpoints();
	std::printf("legacy transform size / checkpoint test: %s\n", fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
