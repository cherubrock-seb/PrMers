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

    // 5. Limits for a 64-bit bit index (Linux/macOS), whatever platform this runs on.
    const uint64_t u32 = 4294967296ULL;
    const uint64_t defMaxe = 268435456ULL;
    // Default chunked Marin stage 1 accepts B1 around and beyond 2^32.
    CHECK(pc::limitError<uint64_t>(u32 - 1, 0, defMaxe, 269, false).empty());
    CHECK(pc::limitError<uint64_t>(u32, 0, defMaxe, 269, false).empty());
    CHECK(pc::limitError<uint64_t>(u32 + 1, 0, defMaxe, 269, false).empty());
    CHECK(pc::limitError<uint64_t>(100000000000ULL, 0, defMaxe, 269, false).empty());
    // Chunk size beyond the supported one, also when the multiplication overflowed.
    CHECK(pc::limitError<uint64_t>(1000, 0, pc::maxEBitsFor<uint64_t>(), 269, false).empty());
    CHECK(!pc::limitError<uint64_t>(1000, 0, pc::maxEBitsFor<uint64_t>() + 1, 269, false).empty());
    CHECK(!pc::limitError<uint64_t>(1000, 0, 0xFFFFFFFFFFFFFFFFULL, 269, false).empty());
    // B1 above the sanity bound.
    CHECK(pc::limitError<uint64_t>(pc::maxB1For<uint64_t>(), 0, defMaxe, 269, false).empty());
    CHECK(!pc::limitError<uint64_t>(pc::maxB1For<uint64_t>() + 1, 0, defMaxe, 269, false).empty());
    CHECK(!pc::limitError<uint64_t>(0xFFFFFFFFFFFFFFFFULL, 0, defMaxe, 269, false).empty());
    // Whole-E paths (legacy, -torus, -b1old extension, Gaussian-Mersenne): E must fit.
    CHECK(pc::limitError<uint64_t>(u32, 0, defMaxe, 269, true).empty());        // ~6.2e9 bits
    CHECK(!pc::limitError<uint64_t>(50000000000ULL, 0, defMaxe, 269, true).empty());  // ~7.2e10 bits
    // An extension only builds the delta.
    CHECK(pc::limitError<uint64_t>(50000000000ULL, 49000000000ULL, defMaxe, 269, true).empty());
    CHECK(!pc::limitError<uint64_t>(50000000000ULL, 1000, defMaxe, 269, true).empty());
    CHECK(!pc::limitError<uint64_t>(50000000000ULL, 0, defMaxe, 269, true).empty());
    // The message names the problem.
    CHECK(pc::limitError<uint64_t>(50000000000ULL, 0, defMaxe, 269, true).find("B1=50000000000") != std::string::npos);
    CHECK(pc::limitError<uint64_t>(1000, 0, 0xFFFFFFFFFFFFFFFFULL, 269, false).find("-maxe") != std::string::npos);

    // 6. The same bounds for a 32-bit bit index (Windows: unsigned long / mp_bitcnt_t
    //    is 32 bits), computed by instantiating the checks on uint32_t.
    {
        using I32 = uint32_t;
        CHECK(pc::maxEBitsFor<I32>() == (1ULL << 31));
        CHECK(pc::maxEBitsFor<uint64_t>() == (1ULL << 36));
        CHECK(pc::maxEBitsFor<unsigned long>() == pc::maxEBits());
        CHECK(pc::maxEBits() == (sizeof(unsigned long) >= 8 ? (1ULL << 36) : (1ULL << 31)));
        CHECK(pc::kMaxB1 == (sizeof(unsigned long) >= 8 ? (1ULL << 62) : 4294967295ULL));
        // the default instantiation is the platform's unsigned long
        CHECK(pc::limitError(4294967296ULL, 0, 268435456ULL, 269, false).empty() == (sizeof(unsigned long) >= 8));
        CHECK(pc::maxB1For<I32>() == 4294967295ULL);
        CHECK(pc::maxB1For<uint64_t>() == (1ULL << 62));
        const uint64_t defMaxe32 = 268435456ULL;              // 2^28 bits: fine in 32 bits
        CHECK(pc::limitError<I32>(4294967295ULL, 0, defMaxe32, 269, false).empty());
        CHECK(!pc::limitError<I32>(4294967296ULL, 0, defMaxe32, 269, false).empty());
        CHECK(pc::limitError<I32>(4294967296ULL, 0, defMaxe32, 269, false).find("too large") != std::string::npos);
        CHECK(pc::limitError<I32>(1000, 0, 1ULL << 31, 269, false).empty());
        CHECK(!pc::limitError<I32>(1000, 0, (1ULL << 31) + 1, 269, false).empty());
        CHECK(!pc::limitError<I32>(1000, 0, 1ULL << 32, 269, false).empty());   // index would wrap
        // Whole-E paths: 1.4e9 -> ~2.02e9 bits fits, 1.5e9 -> ~2.16e9 bits does not.
        CHECK(pc::limitError<I32>(1400000000ULL, 0, defMaxe32, 269, true).empty());
        CHECK(!pc::limitError<I32>(1500000000ULL, 0, defMaxe32, 269, true).empty());
        CHECK(pc::limitError<I32>(1500000000ULL, 0, defMaxe32, 269, true).find("2147483648 bits") != std::string::npos);
        // the same B1 is fine with a 64-bit index
        CHECK(pc::limitError<uint64_t>(1500000000ULL, 0, defMaxe32, 269, true).empty());
        // an extension only needs the delta to fit
        CHECK(pc::limitError<I32>(3600000000ULL, 2000000000ULL, defMaxe32, 269, true).find("exponent") != std::string::npos);
        CHECK(pc::limitError<I32>(3000000000ULL, 2900000000ULL, defMaxe32, 269, true).empty());
        // -tbits
        CHECK(pc::tbitsError<I32>(1ULL << 31).empty());
        CHECK(!pc::tbitsError<I32>((1ULL << 31) + 1).empty());
        CHECK(pc::tbitsError<uint64_t>(1ULL << 36).empty());
        CHECK(!pc::tbitsError<uint64_t>((1ULL << 36) + 1).empty());
        CHECK(pc::tbitsError<uint64_t>(1ULL << 36).empty());
    }

    // 7. Option-level check (all the paths that index E), with a stand-in for CliOptions.
    {
        struct Opts {
            std::string mode = "pm1";
            bool marin = true, torus = false, pm1_lowmem = false, pm1_ultralowmem = false;
            uint64_t B1 = 1000, B1old = 0, B2 = 0, max_e_bits = 268435456ULL, exponent = 269, tbits = 500000;
        };
        auto err = [](const Opts& o) { return pc::optionsLimitError<uint64_t>(o); };
        auto err32 = [](const Opts& o) { return pc::optionsLimitError<uint32_t>(o); };
        Opts o;
        CHECK(err(o).empty());
        o.B1 = 4294967296ULL; CHECK(err(o).empty()); CHECK(!err32(o).empty());
        Opts legacy = o; legacy.B1 = 50000000000ULL; legacy.marin = false; CHECK(!err(legacy).empty());
        Opts torus = o; torus.B1 = 50000000000ULL; torus.torus = true; CHECK(!err(torus).empty());
        Opts ext = o; ext.B1 = 50000000000ULL; ext.B1old = 1000; CHECK(!err(ext).empty());
        ext.B1old = 49999000000ULL; CHECK(err(ext).empty());
        Opts gm = o; gm.mode = "gm-pm1"; gm.B1 = 50000000000ULL; gm.B1old = 49999000000ULL; CHECK(!err(gm).empty());
        // ultra-low-memory stage 2 builds E(B1)*Q up to B2 in one piece
        Opts ulm; ulm.B1 = 1000; ulm.B2 = 50000000000ULL;
        CHECK(err(ulm).empty());
        ulm.pm1_lowmem = ulm.pm1_ultralowmem = true;
        CHECK(!err(ulm).empty());
        CHECK(err(ulm).find("ultra-low-memory") != std::string::npos);
        ulm.B2 = 1000000;
        CHECK(err(ulm).empty());
        // 32-bit index: B2 = 1.5e9 already overflows the product exponent
        ulm.B2 = 1500000000ULL; CHECK(!err32(ulm).empty()); CHECK(err(ulm).empty());
        Opts tb; tb.tbits = (1ULL << 36) + 1; CHECK(!err(tb).empty());
        tb.tbits = 1ULL << 36; CHECK(err(tb).empty());
        tb.tbits = 1ULL << 31; CHECK(err32(tb).empty());
        tb.tbits = (1ULL << 31) + 1; CHECK(!err32(tb).empty()); CHECK(err(tb).empty());
        // other modes are not checked
        Opts prp = o; prp.mode = "prp"; prp.B1 = 0xFFFFFFFFFFFFFFFFULL; prp.tbits = 0xFFFFFFFFFFFFFFFFULL; CHECK(err(prp).empty());
    }

    std::cout.rdbuf(savedCout);
    if (failures) { std::cerr << failures << " check(s) failed\n"; return 1; }
    std::cout << "pm1 stage-1 checkpoint counter test passed\n";
    return 0;
}
