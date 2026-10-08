// Host-only test of the P-1 stage-1 checkpoint header (core/Pm1Checkpoint.hpp):
//   * the resume counter round-trips above 2^32 (version 4, 64-bit field);
//   * a version 3 file (32-bit counter) still loads, widened, with the rest of the
//     file (payload and CRC) aligned;
//   * truncated files, unknown versions and a foreign exponent are rejected;
//   * what an older binary (accepts version 3 only) does with a version 4 file;
//   * the B1 / -maxe limit checks.
#include "core/Pm1Checkpoint.hpp"
#include "marin/file.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <streambuf>
#include <string>
#include <vector>

namespace pc = core::pm1ckpt;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; ++failures; } } while (0)

static std::string g_dir;

static const std::vector<char>& payload() {
    static std::vector<char> p;
    if (p.empty()) for (int i = 0; i < 100; ++i) p.push_back(static_cast<char>(i * 7 + 3));
    return p;
}
static const uint64_t kTrailer = 0x1122334455667788ULL;

// Writes the whole stage-1 layout the driver writes: header, payload, one more
// field and the CRC.  `legacy` writes a version-3 file by hand.
static std::string writeFile(const std::string& name, uint32_t p, uint64_t counter, double et, bool legacy) {
    const std::string path = g_dir + "/" + name;
    File f(path, "wb");
    if (legacy) {
        const int version = pc::kStage1VersionLegacy32;
        const uint32_t c32 = static_cast<uint32_t>(counter);
        CHECK(f.write(reinterpret_cast<const char*>(&version), sizeof(version)));
        CHECK(f.write(reinterpret_cast<const char*>(&p), sizeof(p)));
        CHECK(f.write(reinterpret_cast<const char*>(&c32), sizeof(c32)));
        CHECK(f.write(reinterpret_cast<const char*>(&et), sizeof(et)));
    } else {
        CHECK(pc::writeHeader(f, p, counter, et));
    }
    CHECK(f.write(payload().data(), payload().size()));
    CHECK(f.write(reinterpret_cast<const char*>(&kTrailer), sizeof(kTrailer)));
    CHECK(f.write_crc32());
    CHECK(f.close());
    return path;
}

struct Loaded { bool ok = false; int version = 0; uint64_t counter = 0; double et = 0; };

// Mirrors a driver reader: header, payload, trailing field, CRC.
static Loaded readFile(const std::string& path, uint32_t expectedP) {
    Loaded r;
    File f(path);
    if (!f.exists()) return r;
    if (!pc::readHeader(f, expectedP, r.counter, r.et, &r.version)) return r;
    std::vector<char> data(payload().size());
    if (!f.read(data.data(), data.size())) return r;
    if (data != payload()) return r;
    uint64_t t = 0;
    if (!f.read(reinterpret_cast<char*>(&t), sizeof(t)) || t != kTrailer) return r;
    r.ok = f.check_crc32();
    return r;
}

// What the previous release's reader did: version must be exactly 3, counter is
// 32 bits.  Returns true when it would accept the file.
static bool oldReaderAccepts(const std::string& path, uint32_t expectedP) {
    File f(path);
    if (!f.exists()) return false;
    int version = 0;
    if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return false;
    if (version != 3) return false;
    uint32_t rp = 0, ri = 0; double et = 0;
    if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp)) || rp != expectedP) return false;
    if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return false;
    if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return false;
    std::vector<char> data(payload().size());
    if (!f.read(data.data(), data.size())) return false;
    uint64_t t = 0;
    if (!f.read(reinterpret_cast<char*>(&t), sizeof(t))) return false;
    return f.check_crc32();
}

static std::string slurp(const std::string& path) {
    std::string s;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return s;
    char buf[4096]; size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) s.append(buf, n);
    std::fclose(fp);
    return s;
}
static void spit(const std::string& path, const std::string& s) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp) { std::fwrite(s.data(), 1, s.size(), fp); std::fclose(fp); }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <scratch dir>\n"; return 2; }
    g_dir = argv[1];
    // File::check_crc32 prints "Bad file (crc32)." for every damaged file used below.
    struct NullBuf : std::streambuf { int overflow(int c) override { return c; } } nullBuf;
    std::streambuf* const savedCout = std::cout.rdbuf(&nullBuf);
    const uint32_t P = 269;

    // 1. Version 4 round trip, including counters beyond 2^32.
    const uint64_t counters[] = {0, 1, 4294967295ULL, 4294967296ULL, 4294967296ULL + 7, (1ULL << 40) + 3,
                                 (1ULL << 63), 0xFFFFFFFFFFFFFFFFULL};
    for (uint64_t c : counters) {
        const std::string path = writeFile("v4.ckpt", P, c, 12.5, false);
        const Loaded r = readFile(path, P);
        CHECK(r.ok);
        CHECK(r.version == 4);
        CHECK(r.counter == c);
        CHECK(r.et == 12.5);
        // The old reader must not accept it (it would truncate the counter to 32 bits).
        CHECK(!oldReaderAccepts(path, P));
        // Exact on-disk layout: 4 + 4 + 8 + 8 bytes of header.
        const std::string raw = slurp(path);
        int v; std::memcpy(&v, raw.data(), 4); CHECK(v == 4);
        uint64_t c64; std::memcpy(&c64, raw.data() + 8, 8); CHECK(c64 == c);
    }

    // 2. Version 3 files (old format) still load, widened, with the rest aligned.
    const uint64_t counters32[] = {0, 1, 123456789, 4294967295ULL};
    for (uint64_t c : counters32) {
        const std::string path = writeFile("v3.ckpt", P, c, 3.25, true);
        const Loaded r = readFile(path, P);
        CHECK(r.ok);
        CHECK(r.version == 3);
        CHECK(r.counter == c);
        CHECK(r.et == 3.25);
        CHECK(oldReaderAccepts(path, P));   // unchanged for the old reader
    }

    // 3. Hostile: wrong exponent, unknown versions.
    {
        const std::string path = writeFile("p.ckpt", P, 99, 1.0, false);
        CHECK(!readFile(path, P + 2).ok);
        const std::string v4 = slurp(path);
        for (int bad : {0, 1, 2, 5, 6, -1, 0x7fffffff}) {
            std::string raw = v4;
            std::memcpy(&raw[0], &bad, 4);
            spit(g_dir + "/bad.ckpt", raw);
            CHECK(!readFile(g_dir + "/bad.ckpt", P).ok);
        }
        // A version-4 body relabelled as 3 (or the reverse) is misaligned: the CRC
        // check at the end must reject it rather than yield a wrong counter.
        std::string raw = v4; int three = 3; std::memcpy(&raw[0], &three, 4);
        spit(g_dir + "/relabel.ckpt", raw);
        CHECK(!readFile(g_dir + "/relabel.ckpt", P).ok);
        const std::string v3 = slurp(writeFile("v3b.ckpt", P, 99, 1.0, true));
        std::string raw3 = v3; int four = 4; std::memcpy(&raw3[0], &four, 4);
        spit(g_dir + "/relabel3.ckpt", raw3);
        CHECK(!readFile(g_dir + "/relabel3.ckpt", P).ok);
    }

    // 4. Hostile: truncation at every length, for both versions.
    for (bool legacy : {false, true}) {
        const std::string path = writeFile("t.ckpt", P, legacy ? 77 : (4294967296ULL + 5), 2.0, legacy);
        const std::string full = slurp(path);
        CHECK(readFile(path, P).ok);
        for (size_t n = 0; n < full.size(); ++n) {
            spit(g_dir + "/trunc.ckpt", full.substr(0, n));
            CHECK(!readFile(g_dir + "/trunc.ckpt", P).ok);
        }
        // One flipped bit anywhere is caught by the CRC (or the header checks).
        for (size_t n = 0; n < full.size(); n += 7) {
            std::string raw = full; raw[n] = static_cast<char>(raw[n] ^ 0x10);
            spit(g_dir + "/flip.ckpt", raw);
            const Loaded r = readFile(g_dir + "/flip.ckpt", P);
            CHECK(!r.ok);
        }
    }
    CHECK(!readFile(g_dir + "/does-not-exist.ckpt", P).ok);

    // 5. Limits.
    const uint64_t u32 = 4294967296ULL;
    const uint64_t defMaxe = 268435456ULL;
    // Default chunked Marin stage 1 accepts B1 around and beyond 2^32.
    CHECK(pc::limitError(u32 - 1, 0, defMaxe, 269, false).empty());
    CHECK(pc::limitError(u32, 0, defMaxe, 269, false).empty());
    CHECK(pc::limitError(u32 + 1, 0, defMaxe, 269, false).empty());
    CHECK(pc::limitError(100000000000ULL, 0, defMaxe, 269, false).empty());
    // Chunk size beyond the supported one, also when the multiplication overflowed.
    CHECK(pc::limitError(1000, 0, pc::maxEBits(), 269, false).empty());
    CHECK(!pc::limitError(1000, 0, pc::maxEBits() + 1, 269, false).empty());
    CHECK(!pc::limitError(1000, 0, 0xFFFFFFFFFFFFFFFFULL, 269, false).empty());
    // B1 above the sanity bound.
    CHECK(pc::limitError(pc::kMaxB1, 0, defMaxe, 269, false).empty());
    CHECK(!pc::limitError(pc::kMaxB1 + 1, 0, defMaxe, 269, false).empty());
    CHECK(!pc::limitError(0xFFFFFFFFFFFFFFFFULL, 0, defMaxe, 269, false).empty());
    // Whole-E paths (legacy, -torus, -b1old extension, Gaussian-Mersenne): E must fit.
    CHECK(pc::limitError(u32, 0, defMaxe, 269, true).empty());        // ~6.2e9 bits
    CHECK(!pc::limitError(50000000000ULL, 0, defMaxe, 269, true).empty());  // ~7.2e10 bits
    // An extension only builds the delta.
    CHECK(pc::limitError(50000000000ULL, 49000000000ULL, defMaxe, 269, true).empty());
    CHECK(!pc::limitError(50000000000ULL, 1000, defMaxe, 269, true).empty());
    CHECK(!pc::limitError(50000000000ULL, 0, defMaxe, 269, true).empty());
    // The message names the problem.
    CHECK(pc::limitError(50000000000ULL, 0, defMaxe, 269, true).find("B1=50000000000") != std::string::npos);
    CHECK(pc::limitError(1000, 0, 0xFFFFFFFFFFFFFFFFULL, 269, false).find("-maxe") != std::string::npos);

    std::cout.rdbuf(savedCout);
    if (failures) { std::cerr << failures << " check(s) failed\n"; return 1; }
    std::cout << "pm1 stage-1 checkpoint counter test passed\n";
    return 0;
}
