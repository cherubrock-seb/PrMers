// Host test (no GPU) for the twisted-Edwards ECM Prime95 stage-2 handoff
// helpers: result classification, the per-curve pending markers and the
// interruptible Prime95 runner (driven with stub "mprime" scripts).
#include "core/EcmPrime95Handoff.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace core;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; ++g_fail; } } while (0)

static void write_file(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::trunc);
    f << s;
}

static double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static std::string marker_name(uint64_t p, uint64_t b1, uint64_t curve_no) {
    char num[32];
    std::snprintf(num, sizeof num, "%06llu", (unsigned long long)curve_no);
    return "resume_p" + std::to_string(p) + "_ECM_TE_B1_" + std::to_string(b1) + "_c" + num + ".p95.pending";
}

static std::string marker_body(uint64_t p, uint64_t b1, uint64_t curve_idx, const std::string& resume) {
    return "p=" + std::to_string(p) + "\nB1=" + std::to_string(b1) + "\ncurve=" + std::to_string(curve_idx) +
           "\nresume=" + resume + "\nsigma_hex=ab\ncurve_seed=1\nbase_seed=2\n";
}

static void test_classify() {
    auto o = ecmClassifyPrime95Result("NF", "");
    CHECK(o.ok && !o.factor_found && o.error.empty());
    o = ecmClassifyPrime95Result("F", "10159");
    CHECK(o.ok && o.factor_found);
    o = ecmClassifyPrime95Result("F", "000123");
    CHECK(o.ok && o.factor_found);
    // "F" without a usable factor must be an error, never "no factor".
    for (const char* bad : {"", " 123", "123 ", "-5", "+5", "12a", "0x1F", "1e9"}) {
        o = ecmClassifyPrime95Result("F", bad);
        CHECK(!o.ok && !o.error.empty());
    }
    for (const char* st : {"", "nf", "f", "NF ", "X", "FOUND"}) {
        o = ecmClassifyPrime95Result(st, "");
        CHECK(!o.ok && !o.error.empty());
    }
}

static void test_parse_u64() {
    uint64_t v = 7;
    CHECK(ecmParseU64("0", v) && v == 0);
    CHECK(ecmParseU64("4294967295", v) && v == 4294967295ull);
    CHECK(ecmParseU64("4294967296", v) && v == 4294967296ull);
    CHECK(ecmParseU64("18446744073709551615", v) && v == UINT64_MAX);
    CHECK(ecmParseU64("00000000000000000001", v) && v == 1);
    v = 7;
    for (const char* bad : {"", "18446744073709551616", "99999999999999999999", "-1", "+1", " 1", "1 ", "1\r",
                            "0x10", "1.0", "000000000000000000001"}) {
        CHECK(!ecmParseU64(bad, v));
    }
    CHECK(v == 7);
}

static void test_markers(const fs::path& dir) {
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string resume = (dir / "resume_p1279_ECM_TE_B1_5000_c000003.p95").string();
    EcmPrime95Pending e;
    e.p = 1279; e.b1 = 5000; e.curve_idx = 2; e.resume_path = resume;
    e.sigma_hex = ""; e.curve_seed = UINT64_MAX; e.base_seed = 0;
    CHECK(ecmWritePrime95Pending(e));
    CHECK(fs::exists(ecmPrime95PendingMarkerPath(resume)));
    CHECK(!fs::exists(ecmPrime95PendingMarkerPath(resume) + ".new"));
    EcmPrime95Pending r;
    CHECK(ecmReadPrime95Pending(ecmPrime95PendingMarkerPath(resume), r));
    CHECK(r.p == 1279 && r.b1 == 5000 && r.curve_idx == 2 && r.resume_path == resume &&
          r.sigma_hex.empty() && r.curve_seed == UINT64_MAX && r.base_seed == 0);
    CHECK(ecmWritePrime95Pending(e));  // idempotent overwrite

    // Neighbours that must not be picked up for p=1279, B1=5000.
    write_file(dir / marker_name(1279, 50000, 1), marker_body(1279, 50000, 0, "resume_p1279_ECM_TE_B1_50000_c000001.p95"));
    write_file(dir / marker_name(12791, 5000, 1), marker_body(12791, 5000, 0, "resume_p12791_ECM_TE_B1_5000_c000001.p95"));
    write_file(dir / "resume_p1279_ECM_TE_B1_5000_c000001.p95", "x");
    write_file(dir / "resume_p1279_ECM_TE_B1_5000_c000001.p95.pending.new", "partial");
    // A second good marker (curve 1), CRLF line endings.
    {
        std::string b = marker_body(1279, 5000, 0, "resume_p1279_ECM_TE_B1_5000_c000001.p95");
        std::string crlf;
        for (char ch : b) { if (ch == '\n') crlf += '\r'; crlf += ch; }
        write_file(dir / marker_name(1279, 5000, 1), crlf);
    }
    std::vector<std::string> bad;
    auto found = ecmFindPrime95Pending(dir, 1279, 5000, &bad);
    CHECK(found.size() == 2);
    CHECK(bad.empty());
    if (found.size() == 2) {
        CHECK(found[0].curve_idx == 0 && found[1].curve_idx == 2);  // sorted
        CHECK(found[0].resume_path == (dir / "resume_p1279_ECM_TE_B1_5000_c000001.p95").string());
        CHECK(found[1].curve_seed == UINT64_MAX);
    }

    // Garbage and inconsistent markers are reported, never silently used or dropped.
    struct Bad { uint64_t no; std::string body; };
    const Bad bads[] = {
        {4, ""},                                                                            // empty
        {5, "\x01\x02garbage\n"},                                                           // binary junk
        {6, marker_body(1279, 5000, 5, "resume_p1279_ECM_TE_B1_5000_c000006.p95").substr(0, 20)},  // truncated
        {7, marker_body(1279, 5000, 9, "resume_p1279_ECM_TE_B1_5000_c000007.p95")},         // curve != name
        {8, marker_body(1279, 4999, 7, "resume_p1279_ECM_TE_B1_5000_c000008.p95")},         // B1 mismatch
        {9, marker_body(1279, 5000, 8, "../elsewhere/resume_p1279_ECM_TE_B1_5000_c000010.p95")},  // other file
        {10, "p=1279\nB1=5000\ncurve=-1\nresume=resume_p1279_ECM_TE_B1_5000_c000010.p95\ncurve_seed=1\nbase_seed=2\n"},
        {11, "p=1279\nB1=5000\ncurve=10\nresume=resume_p1279_ECM_TE_B1_5000_c000011.p95\ncurve_seed=99999999999999999999\nbase_seed=2\n"},
        {12, "p=1279\nB1=5000\ncurve=11\nresume=resume_p1279_ECM_TE_B1_5000_c000012.p95\n"},  // no seeds
    };
    for (const Bad& b : bads) write_file(dir / marker_name(1279, 5000, b.no), b.body);
    // Curve index past 32 bits (name and contents agree).
    write_file(dir / "resume_p1279_ECM_TE_B1_5000_c4294967296.p95.pending",
               marker_body(1279, 5000, 4294967295ull, "resume_p1279_ECM_TE_B1_5000_c4294967296.p95"));
    // A directory with a marker's name.
    fs::create_directories(dir / marker_name(1279, 5000, 13));
    bad.clear();
    found = ecmFindPrime95Pending(dir, 1279, 5000, &bad);
    CHECK(found.size() == 2);
    CHECK(bad.size() == sizeof(bads) / sizeof(bads[0]) + 2);

    // Removal.
    ecmRemovePrime95Pending(resume);
    CHECK(!fs::exists(ecmPrime95PendingMarkerPath(resume)));
    ecmRemovePrime95Pending(resume);  // already gone: no error

    // A write that cannot happen reports failure and leaves no partial marker.
    const fs::path ro = dir / "ro";
    fs::create_directories(ro);
    chmod(ro.c_str(), 0555);
    EcmPrime95Pending w = e;
    w.resume_path = (ro / "resume_p1279_ECM_TE_B1_5000_c000003.p95").string();
    if (access(ro.c_str(), W_OK) != 0) {  // not when running as root
        CHECK(!ecmWritePrime95Pending(w));
        CHECK(!fs::exists(ecmPrime95PendingMarkerPath(w.resume_path)));
    }
    chmod(ro.c_str(), 0755);
    // The rename target is a directory: the write fails and the temporary file is removed.
    fs::create_directories(ecmPrime95PendingMarkerPath(w.resume_path) + "/x");
    CHECK(!ecmWritePrime95Pending(w));
    CHECK(!fs::exists(ecmPrime95PendingMarkerPath(w.resume_path) + ".new"));
    fs::remove_all(dir);
}

static std::string make_stub(const fs::path& dir, const std::string& body) {
    fs::create_directories(dir);
    const fs::path exe = dir / "mprime";
    write_file(exe, "#!/bin/sh\n" + body + "\n");
    chmod(exe.c_str(), 0755);
    return "cd '" + dir.string() + "' && exec '" + exe.string() + "' -d > '" + (dir / "log").string() + "' 2>&1";
}

static std::atomic<bool> g_sigint{false};
static void on_sigint(int) { g_sigint.store(true); }

static void test_runner(const fs::path& dir) {
    fs::remove_all(dir);
    bool stopped = true;
    auto never = [] { return false; };

    int st = ecmRunShellInterruptible(make_stub(dir, "exit 0"), never, stopped);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0 && !stopped);
    st = ecmRunShellInterruptible(make_stub(dir, "exit 3"), never, stopped);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 3 && !stopped);
    // Missing executable: the shell fails, it is not an interrupt.
    st = ecmRunShellInterruptible("cd '" + dir.string() + "' && exec ./does-not-exist 2>/dev/null", never, stopped);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) != 0 && !stopped);

    // A stop request terminates a long Prime95 run promptly (SIGTERM).
    {
        std::atomic<bool> stop{false};
        std::thread t([&] { std::this_thread::sleep_for(std::chrono::milliseconds(300)); stop = true; });
        const auto t0 = std::chrono::steady_clock::now();
        st = ecmRunShellInterruptible(make_stub(dir, "trap 'exit 0' TERM\nsleep 30 & wait $!"),
                                      [&] { return stop.load(); }, stopped);
        t.join();
        CHECK(stopped);
        CHECK(secs_since(t0) < 5.0);
    }
    // A Prime95 that ignores SIGTERM is killed after the grace period.
    {
        // Ask to stop only once the stub has installed its trap.
        const auto t0 = std::chrono::steady_clock::now();
        st = ecmRunShellInterruptible(make_stub(dir, "trap '' TERM\nsleep 30"),
                                      [&] { return secs_since(t0) > 0.5; }, stopped, 1);
        CHECK(stopped);
        CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
        CHECK(secs_since(t0) < 5.0);
    }
    // SIGINT sent to this process while Prime95 runs reaches our handler
    // (std::system() would have ignored it) and stops the child.
    {
        std::signal(SIGINT, on_sigint);
        g_sigint = false;
        std::thread t([] { std::this_thread::sleep_for(std::chrono::milliseconds(300)); kill(getpid(), SIGINT); });
        const auto t0 = std::chrono::steady_clock::now();
        st = ecmRunShellInterruptible(make_stub(dir, "trap 'exit 0' TERM\nsleep 30 & wait $!"),
                                      [] { return g_sigint.load(); }, stopped);
        t.join();
        CHECK(g_sigint.load());
        CHECK(stopped);
        CHECK(secs_since(t0) < 5.0);
        std::signal(SIGINT, SIG_DFL);
    }
    // The child exits on its own just as the stop arrives (Ctrl-C reaches the
    // whole process group): still reported as stopped.
    {
        std::atomic<bool> stop{false};
        st = ecmRunShellInterruptible(make_stub(dir, "exit 0"), [&] { bool s = stop.load(); stop = true; return s; },
                                      stopped);
        CHECK(stopped);
    }
    fs::remove_all(dir);
}

int main(int argc, char** argv) {
    const fs::path dir = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() / "ecm_p95_handoff_test";
    test_classify();
    test_parse_u64();
    test_markers(dir / "markers");
    test_runner(dir / "runner");
    if (g_fail) {
        std::cerr << g_fail << " check(s) failed\n";
        return 1;
    }
    std::cout << "ecm Prime95 handoff host test passed\n";
    return 0;
}
