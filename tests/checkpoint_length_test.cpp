// Host test for core::ckptlen::readLenPrefixed: a length read from a checkpoint
// before its CRC is checked must never size an allocation.  A garbage length
// (up to 4 GiB) is rejected, like a CRC failure, without allocating.  No GPU.
#include "core/CheckpointLength.hpp"
#include "marin/file.h"

#include <sys/resource.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <new>
#include <string>

namespace cl = core::ckptlen;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; ++g_fail; } } while (0)

static std::string make_file(const std::string& path, uint32_t len, const std::string& body) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&len), sizeof(len));
    f << body;
    return path;
}

// Current virtual size in bytes (Linux), 0 if unknown.
static size_t current_vsz() {
    std::ifstream f("/proc/self/statm");
    size_t pages = 0;
    if (!(f >> pages)) return 0;
    return pages * static_cast<size_t>(sysconf(_SC_PAGESIZE));
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <scratch dir>\n"; return 2; }
    const std::string dir = argv[1];
    const std::string path = dir + "/len.bin";

    // Ordinary fields.
    {
        make_file(path, 5, "abcde");
        File f(path);
        std::string s = "stale";
        CHECK(cl::readLenPrefixed(f, cl::fileSizeOf(path), s));
        CHECK(s == "abcde");
    }
    {
        make_file(path, 0, "");
        File f(path);
        std::string s = "stale";
        CHECK(cl::readLenPrefixed(f, cl::fileSizeOf(path), s));
        CHECK(s.empty());
    }
    {   // two fields back to back, as the loaders read them
        std::ofstream o(path, std::ios::binary | std::ios::trunc);
        const uint32_t a = 3, b = 2;
        o.write(reinterpret_cast<const char*>(&a), 4); o << "1f0";
        o.write(reinterpret_cast<const char*>(&b), 4); o << "ab";
        o.close();
        File f(path);
        const uint64_t fsz = cl::fileSizeOf(path);
        CHECK(fsz == 13);
        std::string x, y;
        CHECK(cl::readLenPrefixed(f, fsz, x) && x == "1f0");
        CHECK(cl::readLenPrefixed(f, fsz, y) && y == "ab");
    }

    // Bad fields: rejected, nothing left in `out`.
    const uint32_t bad[] = {0xFFFFFFFFu, 0x80000000u, 0x7FFFFFFFu, 0x10000000u, 21u, 100u};
    for (uint32_t len : bad) {
        make_file(path, len, std::string(16, 'x'));   // 20-byte file
        File f(path);
        std::string s = "stale";
        CHECK(!cl::readLenPrefixed(f, cl::fileSizeOf(path), s));
        CHECK(s.empty());
    }
    {   // length within the file size but more than the bytes that follow
        make_file(path, 18, std::string(16, 'x'));
        File f(path);
        std::string s;
        CHECK(!cl::readLenPrefixed(f, cl::fileSizeOf(path), s));
        CHECK(s.empty());
    }
    {   // truncated length field
        std::ofstream o(path, std::ios::binary | std::ios::trunc);
        o << "ab"; o.close();
        File f(path);
        std::string s;
        CHECK(!cl::readLenPrefixed(f, cl::fileSizeOf(path), s));
    }
    {   // unknown file size (0): only an empty field passes
        make_file(path, 4, "abcd");
        File f(path);
        std::string s;
        CHECK(!cl::readLenPrefixed(f, 0, s));
    }
    CHECK(cl::fileSizeOf(dir + "/does-not-exist") == 0);

    // No allocation on a garbage length: cap the address space just above what
    // the process uses now.  The old code (std::string(len, '\0')) throws
    // bad_alloc here; the helper must return false and not throw.
    const size_t vsz = current_vsz();
    if (vsz != 0) {
        struct rlimit old{};
        getrlimit(RLIMIT_AS, &old);
        struct rlimit lim = old;
        lim.rlim_cur = vsz + (64u << 20);
        if (setrlimit(RLIMIT_AS, &lim) == 0) {
            bool legacyThrew = false;
            try {
                uint32_t len = 0xFFFFFFFFu;
                std::string s(len, '\0');
                legacyThrew = s.empty();   // not reached
            } catch (const std::bad_alloc&) { legacyThrew = true; }
            CHECK(legacyThrew);            // the limit really stops a 4 GiB allocation

            make_file(path, 0xFFFFFFFFu, std::string(16, 'x'));
            bool threw = false, ok = true;
            try {
                File f(path);
                std::string s;
                ok = cl::readLenPrefixed(f, cl::fileSizeOf(path), s);
                CHECK(s.capacity() < 4096);
            } catch (...) { threw = true; }
            CHECK(!threw);
            CHECK(!ok);
            setrlimit(RLIMIT_AS, &old);
        } else {
            std::cout << "note: setrlimit(RLIMIT_AS) failed; allocation-limit check skipped\n";
        }
    } else {
        std::cout << "note: /proc/self/statm unavailable; allocation-limit check skipped\n";
    }

    std::remove(path.c_str());
    if (g_fail) { std::cerr << g_fail << " check(s) failed\n"; return 1; }
    std::cout << "checkpoint length tests passed\n";
    return 0;
}
