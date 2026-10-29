// Host test for core::ckptcommit::commit and the checkpoint writer contract
// `write_crc32() && close()`: a checkpoint that cannot be written completely
// (full disk, quota, I/O error at the last write or at flush/close) or swapped
// into place must never cost the previous good checkpoint.  No GPU needed.
#include "core/CheckpointCommit.hpp"
#include "marin/file.h"

#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;
namespace cc = core::ckptcommit;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; ++g_fail; } } while (0)

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
static void spit(const std::string& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << s;
}
static bool exists(const std::string& p) { std::error_code ec; return fs::exists(p, ec); }

// Writes `payload` bytes plus the CRC to `<ckpt>.new` with the file size limit
// set to `limit` bytes; returns what the new checkpoint writers test.
static bool write_new_limited(const std::string& ckpt, size_t payload, rlim_t limit) {
    struct rlimit old{};
    getrlimit(RLIMIT_FSIZE, &old);
    struct rlimit lim = old;
    lim.rlim_cur = limit;
    setrlimit(RLIMIT_FSIZE, &lim);
    bool ok = false;
    {
        File f(ckpt + ".new", "wb");
        const std::string data(payload, 'n');
        ok = f.write(data.data(), data.size());      // buffered: still succeeds
        ok = ok && f.write_crc32() && f.close();      // the flush hits the limit
    }
    setrlimit(RLIMIT_FSIZE, &old);
    return ok;
}

// The sequence the stage-1 writers used before: nothing is checked.
static void legacy_swap(const std::string& ckpt) {
    std::error_code ec;
    fs::remove(ckpt + ".old", ec);
    fs::rename(ckpt, ckpt + ".old", ec);
    fs::rename(ckpt + ".new", ckpt, ec);
    fs::remove(ckpt + ".old", ec);
}

static void clean(const std::string& base) {
    std::error_code ec;
    fs::remove_all(base, ec); fs::remove_all(base + ".old", ec); fs::remove_all(base + ".new", ec);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <scratch dir>\n"; return 2; }
    std::signal(SIGXFSZ, SIG_IGN);          // EFBIG from write() instead of a fatal signal
    const std::string dir = argv[1];
    const std::string ck = dir + "/x.ckpt";

    // 0. Scenario check: with the old unchecked sequence a write that fails at
    //    flush/close deletes the only good checkpoint.
    {
        clean(ck);
        spit(ck, "good");
        CHECK(!write_new_limited(ck, 5000, 1000));
        CHECK(!exists(ck + ".new"));        // File::close removed the failed file
        legacy_swap(ck);
        CHECK(!exists(ck));                  // ... and so the legacy swap lost "good"
        CHECK(!exists(ck + ".old"));
        clean(ck);
    }

    // 1. Writer failure at the CRC / flush: the caller stops, nothing is touched.
    {
        clean(ck);
        spit(ck, "good");
        spit(ck + ".old", "older");
        // 998 payload bytes + 4 CRC bytes stay in the stdio buffer; the limit
        // (1000) is hit by the flush in close().
        const bool ok = write_new_limited(ck, 998, 1000);
        CHECK(!ok);
        CHECK(!exists(ck + ".new"));
        CHECK(slurp(ck) == "good");
        CHECK(slurp(ck + ".old") == "older");
        // Even if a caller did call commit, a missing .new changes nothing.
        CHECK(!cc::commit(ck));
        CHECK(slurp(ck) == "good");
        CHECK(slurp(ck + ".old") == "older");
        CHECK(!exists(ck + ".new"));
        clean(ck);
    }

    // 2. Successful commits.
    {
        clean(ck);
        spit(ck, "v1"); spit(ck + ".old", "stale"); spit(ck + ".new", "v2");
        CHECK(cc::commit(ck));                       // stage 1: .old is dropped
        CHECK(slurp(ck) == "v2");
        CHECK(!exists(ck + ".old") && !exists(ck + ".new"));
        spit(ck + ".new", "v3");
        CHECK(cc::commit(ck, true));                 // stage 2: previous kept
        CHECK(slurp(ck) == "v3");
        CHECK(slurp(ck + ".old") == "v2");
        CHECK(!exists(ck + ".new"));
        clean(ck);
        spit(ck + ".new", "first");                  // very first save, no ckpt yet
        CHECK(cc::commit(ck));
        CHECK(slurp(ck) == "first");
        CHECK(!exists(ck + ".old") && !exists(ck + ".new"));
        clean(ck);
    }

    // 3. First rename fails (.old is a non-empty directory): ckpt stays.
    {
        clean(ck);
        spit(ck, "good"); spit(ck + ".new", "new");
        fs::create_directory(ck + ".old");
        spit(ck + ".old/keep", "k");
        CHECK(!cc::commit(ck));
        CHECK(slurp(ck) == "good");
        CHECK(!exists(ck + ".new"));
        CHECK(exists(ck + ".old/keep"));
        clean(ck);
    }

    // 4. Second rename fails (injected): the previous checkpoint is put back.
    {
        for (bool keepOld : {false, true}) {
            clean(ck);
            spit(ck, "good"); spit(ck + ".new", "new"); spit(ck + ".old", "stale");
            int calls = 0;
            const bool ok = cc::commitWith(ck, keepOld,
                [&](const std::string& a, const std::string& b, std::error_code& ec) {
                    if (++calls == 2) { ec = std::make_error_code(std::errc::io_error); return; }
                    std::filesystem::rename(a, b, ec);
                });
            CHECK(!ok);
            CHECK(calls == 2);
            CHECK(slurp(ck) == "good");
            CHECK(!exists(ck + ".new"));
            CHECK(!exists(ck + ".old"));
            clean(ck);
        }
        // Same failure on the very first save: nothing to restore, no debris.
        spit(ck + ".new", "new");
        CHECK(!cc::commitWith(ck, false,
            [&](const std::string&, const std::string&, std::error_code& ec) {
                ec = std::make_error_code(std::errc::io_error);
            }));
        CHECK(!exists(ck) && !exists(ck + ".new") && !exists(ck + ".old"));
        clean(ck);
    }

    // 4b. A `.new` that is not a regular file (a directory) is never swapped in.
    {
        clean(ck);
        spit(ck, "good");
        fs::create_directory(ck + ".new");
        CHECK(!cc::commit(ck));
        CHECK(slurp(ck) == "good");
        CHECK(!exists(ck + ".old"));
        clean(ck);
    }

    // 5. Repeated failures never accumulate state: good stays good.
    {
        clean(ck);
        spit(ck, "good");
        for (int i = 0; i < 3; ++i) CHECK(!write_new_limited(ck, 998, 1000) && !cc::commit(ck));
        CHECK(slurp(ck) == "good");
        spit(ck + ".new", "later");
        CHECK(cc::commit(ck));
        CHECK(slurp(ck) == "later");
        clean(ck);
    }

    if (g_fail) { std::cerr << g_fail << " check(s) failed\n"; return 1; }
    std::cout << "checkpoint commit tests passed\n";
    return 0;
}
