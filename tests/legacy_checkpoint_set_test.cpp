// Host test (no OpenCL device needed): the legacy (-marin) checkpoint is saved as one consistent set.
//
// The state (.mers), its iteration (.loop) and the Gerbicz-Li files (.bufd/.lbufd/.gli/.isav/.jsav), or the
// P-1 stage-2 files (.hq/.q/.loop2), are written as "<file>.new", committed by a manifest "<base>.state"
// that records the size and CRC of each, then renamed into place with the previous copies kept as ".old".
// The reader must use only a complete set that matches a manifest (the current one, then ".state.old"),
// whatever mix of current, ".new" and ".old" files a kill left, and start over when no set matches. Files of
// an earlier version (no manifest) are read as before.
#include "core/BackupManager.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int fails = 0;
const size_t n = 256;
const unsigned P = 4451;

void expect(const bool ok, const std::string & what)
{
	if (!ok) { ++fails; std::printf("FAIL %s\n", what.c_str()); }
}

core::BackupManager make(const fs::path & dir, const char * mode = "prp", const uint64_t b1 = 0, const uint64_t b2 = 0, const size_t words = n, const unsigned p = P)
{
	return core::BackupManager(nullptr, 60, words, dir.string(), p, mode, b1, b2, false, false);
}

std::vector<uint64_t> vec(const uint64_t seed)
{
	std::vector<uint64_t> v(n);
	for (size_t i = 0; i < n; ++i) v[i] = seed * 1000003u + i;
	return v;
}

core::BackupManager::GerbiczLiHost gl(const uint64_t seed)
{
	core::BackupManager::GerbiczLiHost g;
	g.bufd = vec(seed + 1); g.lastBufd = vec(seed + 2); g.correct = vec(seed + 3);
	g.itersave = seed * 10; g.jsave = seed * 20;
	return g;
}

std::string read(const fs::path & f)
{
	std::ifstream in(f, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write(const fs::path & f, const std::string & s) { std::ofstream o(f, std::ios::binary | std::ios::trunc); o << s; }

// CRC-32 (IEEE), as the manifest records it, to forge a manifest that is intact but of another version
std::string crc_hex(const std::string & s)
{
	uint32_t c = 0xFFFFFFFFu;
	for (const unsigned char b : s) { c ^= b; for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1; }
	char buf[16]; std::snprintf(buf, sizeof buf, "%08x", c ^ 0xFFFFFFFFu);
	return buf;
}

std::string reseal(const std::string & manifest)
{
	const std::string body = manifest.substr(0, manifest.rfind("end "));
	return body + "end " + crc_hex(body) + "\n";
}

fs::path fresh_dir(const fs::path & root, const std::string & name)
{
	const fs::path d = root / name;
	fs::remove_all(d);
	fs::create_directories(d);
	return d;
}

const std::string base = "4451prp";
const char * const parts[] = { "mers", "loop", "bufd", "lbufd", "gli", "isav", "jsav" };

// load and check: the resume value, x and the Gerbicz-Li part
void expect_set(const fs::path & d, const uint64_t want, const uint64_t seed, const bool with_gl, const std::string & what)
{
	auto b = make(d);
	std::vector<uint64_t> x(n, 9);
	const uint64_t r = b.loadState(x);
	expect(r == want, what + ": resume " + std::to_string(r) + ", expected " + std::to_string(want));
	if (want == 0) {
		expect(x[0] == 3 && x[1] == 0 && x[n - 1] == 0, what + ": fresh state");
	} else {
		expect(x == vec(seed), what + ": residue of the set");
	}
	std::vector<uint64_t> a(n, 9), c(n, 9), l(n, 9);
	b.loadGerbiczLiBufDState(a); b.loadGerbiczLiCorrectState(c); b.loadGerbiczLiCorrectBufDState(l);
	const uint64_t is = b.loadGerbiczIterSave(), js = b.loadGerbiczJSave();
	if (want != 0 && with_gl) {
		const auto g = gl(seed);
		expect(a == g.bufd && l == g.lastBufd && c == g.correct, what + ": Gerbicz-Li buffers of the set");
		expect(is == g.itersave && js == g.jsave, what + ": Gerbicz-Li counters of the set");
	} else {
		expect(a[0] == 1 && a[1] == 0 && c[0] == 3 && c[1] == 0 && l[0] == 1 && l[1] == 0, what + ": default Gerbicz-Li buffers");
		expect(is == 0 && js == 0, what + ": default Gerbicz-Li counters");
	}
}

// Two saves: A (seed 1, iteration 77) then B (seed 2, iteration 100). B is current, A is ".old".
fs::path two_saves(const fs::path & root, const std::string & name, const bool with_gl = true)
{
	const fs::path d = fresh_dir(root, name);
	auto b = make(d);
	const auto ga = gl(1), gb = gl(2);
	expect(b.saveStateHost(vec(1), 76, with_gl ? &ga : nullptr), name + ": save A");
	expect(b.saveStateHost(vec(2), 99, with_gl ? &gb : nullptr), name + ": save B");
	return d;
}

void test_round_trip(const fs::path & root)
{
	{
		const fs::path d = fresh_dir(root, "rt");
		auto b = make(d);
		const auto g = gl(1);
		expect(b.saveStateHost(vec(1), 76, &g), "round trip: save");
		expect(read(d / (base + ".loop")) == "77", "the .loop file still holds the plain count");
		expect(fs::file_size(d / (base + ".mers")) == n * 8, "the .mers file is still the raw residue");
		expect(fs::exists(d / (base + ".state")), "manifest written");
		for (const char * s : parts) expect(!fs::exists(d / (base + "." + s + ".new")), std::string("no .new left: ") + s);
		expect_set(d, 77, 1, true, "round trip");
	}
	{
		const fs::path d = two_saves(root, "rt2");
		for (const char * s : parts) expect(fs::exists(d / (base + "." + s + ".old")), std::string("previous copy kept: ") + s);
		expect(fs::exists(d / (base + ".state.old")), "previous manifest kept");
		expect_set(d, 100, 2, true, "second save");
	}
	{
		const fs::path d = two_saves(root, "nogl", false);
		expect(!fs::exists(d / (base + ".bufd")), "no Gerbicz-Li files without Gerbicz-Li");
		expect_set(d, 100, 2, false, "set without Gerbicz-Li");
	}
	{
		// an interrupt before the first iteration records 0: a fresh start
		const fs::path d = fresh_dir(root, "zero");
		auto b = make(d);
		expect(b.saveStateHost(vec(1), UINT64_MAX, nullptr), "save at 0");
		expect_set(d, 0, 0, false, "set at iteration 0");
	}
}

// What a kill at each step of a save leaves, rebuilt by hand from a completed save B over A.
void test_interrupted_saves(const fs::path & root)
{
	// step 1, before the commit: some "<file>.new" written (one torn), the manifest still names A
	for (size_t k = 0; k <= std::size(parts); ++k) {
		const fs::path d = two_saves(root, "pre" + std::to_string(k));
		// undo B: A back in place, B's files as ".new", A's manifest current
		for (const char * s : parts) {
			const fs::path f = d / (base + "." + s);
			fs::rename(f, fs::path(f.string() + ".new"));
			fs::rename(fs::path(f.string() + ".old"), f);
		}
		fs::rename(d / (base + ".state.old"), d / (base + ".state"));
		// only the first k ".new" files were written, the k-th torn
		for (size_t i = 0; i < std::size(parts); ++i) {
			const fs::path f = d / (base + "." + parts[i] + ".new");
			if (i > k) fs::remove(f);
			else if (i == k) { const std::string s = read(f); write(f, s.substr(0, s.size() / 2)); }
		}
		expect_set(d, 77, 1, true, "kill before the commit, k=" + std::to_string(k));
	}
	// step 3, after the commit: the files of B moved into place one by one; each file is either
	// still ".new" (A in place), being moved (A already ".old", nothing in place) or in place
	for (size_t k = 0; k < std::size(parts); ++k) {
		for (int stage = 0; stage < 2; ++stage) {
			const fs::path d = two_saves(root, "post" + std::to_string(k) + "_" + std::to_string(stage));
			for (size_t i = k; i < std::size(parts); ++i) {
				const fs::path f = d / (base + "." + parts[i]);
				fs::rename(f, fs::path(f.string() + ".new"));
				if (i == k && stage == 1) continue;  // A already moved to ".old", B not yet in place
				fs::rename(fs::path(f.string() + ".old"), f);
			}
			expect_set(d, 100, 2, true, "kill after the commit, k=" + std::to_string(k) + " stage=" + std::to_string(stage));
		}
	}
}

void test_inconsistent(const fs::path & root)
{
	{
		// the .mers of A with the .loop of B (the old non-atomic save's failure): B does not match,
		// A (.state.old) does with its .loop.old
		const fs::path d = two_saves(root, "mix");
		fs::copy_file(d / (base + ".mers.old"), d / (base + ".mers"), fs::copy_options::overwrite_existing);
		expect_set(d, 77, 1, true, "residue of A with the loop of B");
	}
	{
		// a .loop that does not hold the manifest's iteration
		const fs::path d = two_saves(root, "loopval");
		write(d / (base + ".loop"), "50");
		fs::remove(d / (base + ".loop.old"));
		expect_set(d, 0, 0, true, "loop value not the set's, no previous copy");
	}
	{
		// truncated residue, with and without the previous set
		const fs::path d = two_saves(root, "trunc");
		const std::string s = read(d / (base + ".mers"));
		write(d / (base + ".mers"), s.substr(0, 100));
		expect_set(d, 77, 1, true, "truncated residue -> previous set");
		fs::remove(d / (base + ".mers.old"));
		expect_set(d, 0, 0, true, "truncated residue, no previous copy -> start over");
	}
	{
		// one flipped byte in each file in turn
		for (const char * part : parts) {
			const fs::path d = two_saves(root, std::string("crc_") + part);
			const fs::path f = d / (base + "." + part);
			std::string s = read(f);
			s[s.size() / 2] ^= 0x01;
			write(f, s);
			expect_set(d, 77, 1, true, std::string("bad checksum of .") + part + " -> previous set");
		}
	}
	{
		// a Gerbicz-Li file of A with the rest of B: never mixed
		const fs::path d = two_saves(root, "glmix");
		fs::copy_file(d / (base + ".gli.old"), d / (base + ".gli"), fs::copy_options::overwrite_existing);
		fs::remove(d / (base + ".gli.old"));
		expect_set(d, 77, 1, true, "Gerbicz-Li file of A with the rest of B -> set A");
		fs::remove(d / (base + ".mers.old"));
		expect_set(d, 0, 0, true, "Gerbicz-Li file of A, no residue of A -> start over");
	}
	{
		// manifest damaged, truncated, or of another format version -> previous set
		const fs::path d = two_saves(root, "man");
		const std::string good = read(d / (base + ".state"));
		std::string bad = good; bad[good.find("iteration") + 10] ^= 0x01;
		write(d / (base + ".state"), bad);
		expect_set(d, 77, 1, true, "damaged manifest -> previous set");
		write(d / (base + ".state"), good.substr(0, good.size() - 12));
		expect_set(d, 77, 1, true, "truncated manifest -> previous set");
		write(d / (base + ".state"), "");
		expect_set(d, 77, 1, true, "empty manifest -> previous set");
		expect(reseal(good) == good, "the test's CRC matches the manifest's");
		std::string v2 = good; v2.replace(good.find(" 1\n"), 3, " 2\n");
		write(d / (base + ".state"), reseal(v2));
		expect_set(d, 77, 1, true, "manifest of another version -> previous set");
		fs::remove(d / (base + ".state.old"));
		expect_set(d, 0, 0, true, "no usable manifest -> start over");
		fs::remove(d / (base + ".state"));
		// no manifest at all: the current files are read as an earlier version's
		expect_set(d, 100, 2, true, "manifest removed -> files read as an earlier version wrote them");
	}
	{
		// a set saved for another exponent or transform size is not used
		const fs::path d = two_saves(root, "other");
		auto b = make(d, "prp", 0, 0, n * 2);
		std::vector<uint64_t> x(n * 2, 9);
		expect(b.loadState(x) == 0 && x[0] == 3, "set of another transform size rejected");
	}
}

void test_old_format(const fs::path & root)
{
	{
		// files of an earlier version, including Gerbicz-Li ones, resume as before
		const fs::path d = fresh_dir(root, "old");
		const auto g = gl(1);
		auto words = [&](const std::string & s, const std::vector<uint64_t> & v) {
			write(d / (base + "." + s), std::string(reinterpret_cast<const char *>(v.data()), v.size() * 8));
		};
		words("mers", vec(1)); words("bufd", g.bufd); words("lbufd", g.lastBufd); words("gli", g.correct);
		write(d / (base + ".loop"), "77"); write(d / (base + ".isav"), "10"); write(d / (base + ".jsav"), "20");
		expect_set(d, 77, 1, true, "earlier version's files");

		// the first save of this version, killed before its commit: the earlier files are untouched
		for (const char * s : parts) write(d / (base + "." + s + ".new"), "torn");
		expect_set(d, 77, 1, true, "earlier version's files with a torn first save");

		// and once it is committed, the new set is used
		auto b = make(d);
		const auto g2 = gl(2);
		expect(b.saveStateHost(vec(2), 99, &g2), "first save over an earlier version's files");
		expect_set(d, 100, 2, true, "first save over an earlier version's files");
		expect(read(d / (base + ".loop.old")) == "77", "the earlier .loop is kept as .old");
	}
	{
		// an earlier version's truncated residue still starts over, Gerbicz-Li files included
		const fs::path d = fresh_dir(root, "oldtrunc");
		const auto g = gl(1);
		write(d / (base + ".mers"), "short");
		write(d / (base + ".bufd"), std::string(reinterpret_cast<const char *>(g.bufd.data()), n * 8));
		write(d / (base + ".loop"), "77"); write(d / (base + ".isav"), "10");
		expect_set(d, 0, 0, true, "earlier version's truncated residue");
	}
}

void test_stage2(const fs::path & root)
{
	const std::string b2 = "4451pm1100_200";
	auto load = [&](const fs::path & d, std::vector<uint64_t> & hq, std::vector<uint64_t> & q) {
		auto b = make(d, "pm1", 100, 200);
		hq.assign(n, 9); q.assign(n, 9);
		return b.loadStatePM1S2Host(hq, q);
	};
	const fs::path d = fresh_dir(root, "s2");
	{
		auto b = make(d, "pm1", 100, 200);
		expect(b.saveStatePM1S2Host(vec(1), vec(2), 41), "stage 2: save A");
		expect(b.saveStatePM1S2Host(vec(3), vec(4), 99), "stage 2: save B");
	}
	expect(read(d / (b2 + ".loop2")) == "100", "the .loop2 file still holds the plain count");
	std::vector<uint64_t> hq, q;
	expect(load(d, hq, q) == 100 && hq == vec(3) && q == vec(4), "stage 2: set B");
	fs::copy_file(d / (b2 + ".q.old"), d / (b2 + ".q"), fs::copy_options::overwrite_existing);
	expect(load(d, hq, q) == 42 && hq == vec(1) && q == vec(2), "stage 2: q of A with the rest of B -> set A");
	fs::remove(d / (b2 + ".hq.old"));
	expect(load(d, hq, q) == 0, "stage 2: no complete set -> stage 2 from the beginning");
}

void test_clear(const fs::path & root)
{
	const fs::path d = two_saves(root, "clear");
	for (const char * s : parts) write(d / (base + "." + s + ".new"), "x");
	write(d / (base + ".state.new"), "x");
	auto b = make(d);
	b.clearState();
	expect(fs::is_empty(d), "clearState removes the set, its .new/.old copies and the manifests");
}

}	// namespace

int main()
{
	const fs::path root = fs::temp_directory_path() / ("prmers-legacy-set-" + std::to_string(::getpid()));
	fs::remove_all(root);
	fs::create_directories(root);
	test_round_trip(root);
	test_interrupted_saves(root);
	test_inconsistent(root);
	test_old_format(root);
	test_stage2(root);
	test_clear(root);
	fs::remove_all(root);
	std::printf("legacy checkpoint set test: %s\n", fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
