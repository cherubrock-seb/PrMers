// Host-only test of include/core/Pm1VTraceCheckpoint.hpp: the pair95 stage-2
// checkpoint header round-trips, short and foreign headers are rejected with
// the field that differs, positions outside the plan are refused, and the
// .new -> file -> .old rotation keeps the previous checkpoint.
#include "core/Pm1VTraceCheckpoint.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

namespace pvc = core::pm1vtrace;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; ++failures; } } while (0)

// In-memory stand-in for the marin File class (read/write only).
struct Buf {
    std::vector<char> d;
    size_t pos = 0;
    bool write(const char* p, size_t n) { d.insert(d.end(), p, p + n); return true; }
    bool read(char* p, size_t n) {
        if (pos + n > d.size()) { pos = d.size(); return false; }
        std::memcpy(p, d.data() + pos, n);
        pos += n;
        return true;
    }
};

static pvc::Pair95Header sample() {
    pvc::Pair95Header h;
    h.exponent = 9941; h.B1 = 1000; h.B2 = 1000000;
    h.D = 210; h.L = 2; h.babies = 48; h.activeBabies = 40; h.passes = 2;
    h.tasks = 58111; h.planHash = 0x0123456789abcdefULL; h.negAdd = 1;
    h.pass = 1; h.taskPos = 17; h.k = 4242; h.terms = 54727; h.primes = 77777;
    h.elapsed = 12.5;
    return h;
}

int main() {
    // Round trip, 64-bit fields included.
    {
        pvc::Pair95Header h = sample();
        h.terms = (1ULL << 40) + 3;   // above 32 bits
        h.tasks = (1ULL << 41);
        Buf b;
        CHECK(pvc::writeHeader(b, h));
        pvc::Pair95Header r;
        int32_t v = 0;
        CHECK(pvc::readHeader(b, r, &v) == pvc::ReadStatus::Ok);
        CHECK(v == pvc::kPair95Version);
        CHECK(r.terms == h.terms && r.tasks == h.tasks && r.k == h.k && r.taskPos == h.taskPos);
        CHECK(r.planHash == h.planHash && r.elapsed == h.elapsed);
        CHECK(pvc::mismatch(r, h).empty());
    }
    // Every truncation of the header is reported as short, never as Ok.
    {
        Buf full;
        pvc::writeHeader(full, sample());
        for (size_t n = 0; n < full.d.size(); ++n) {
            Buf b;
            b.d.assign(full.d.begin(), full.d.begin() + (long)n);
            pvc::Pair95Header r;
            CHECK(pvc::readHeader(b, r) == pvc::ReadStatus::Short);
        }
    }
    // Unknown versions (older v24, garbage) are not read as v25.
    for (int32_t ver : {0, 24, 26, -1, 0x7fffffff}) {
        Buf b;
        pvc::writeHeader(b, sample());
        std::memcpy(b.d.data(), &ver, sizeof(ver));
        pvc::Pair95Header r;
        int32_t v = 0;
        CHECK(pvc::readHeader(b, r, &v) == pvc::ReadStatus::BadVersion);
        CHECK(v == ver);
    }
    // Each job/plan field is checked and named.
    {
        const pvc::Pair95Header run = sample();
        struct Case { const char* name; uint64_t pvc::Pair95Header::*f; };
        const Case cases[] = {
            {"exponent", &pvc::Pair95Header::exponent}, {"B1", &pvc::Pair95Header::B1},
            {"B2", &pvc::Pair95Header::B2}, {"D", &pvc::Pair95Header::D}, {"L", &pvc::Pair95Header::L},
            {"baby count", &pvc::Pair95Header::babies},
            {"active babies per pass", &pvc::Pair95Header::activeBabies},
            {"pass count", &pvc::Pair95Header::passes}, {"task count", &pvc::Pair95Header::tasks},
            {"plan hash", &pvc::Pair95Header::planHash},
            {"negative-baby mode", &pvc::Pair95Header::negAdd},
        };
        for (const Case& c : cases) {
            pvc::Pair95Header file = run;
            file.*(c.f) += 1;
            const std::string m = pvc::mismatch(file, run);
            CHECK(m.rfind(std::string(c.name) + " differs", 0) == 0);
        }
        pvc::Pair95Header file = run;
        file.B2 = 2000000;
        CHECK(pvc::mismatch(file, run) == "B2 differs (checkpoint 2000000, this run 1000000)");
        // Position fields do not make a checkpoint foreign.
        file = run;
        file.pass = 0; file.taskPos = 99; file.k = 1; file.terms = 5; file.primes = 7; file.elapsed = 1.0;
        CHECK(pvc::mismatch(file, run).empty());
    }
    // Positions.
    {
        auto bucket = [](uint64_t pass) -> uint64_t { return pass == 0 ? 54710 : 3401; };
        pvc::Pair95Header h = sample();
        h.pass = 0; h.taskPos = 0; h.terms = 0;
        CHECK(pvc::positionError(h, bucket).empty());
        h.pass = 0; h.taskPos = 54710; h.terms = 54710;
        CHECK(pvc::positionError(h, bucket).empty());              // end of a pass
        h.taskPos = 54711;
        CHECK(!pvc::positionError(h, bucket).empty());
        h.pass = 2; h.taskPos = 0; h.terms = 58111;
        CHECK(pvc::positionError(h, bucket).empty());              // only the GCD left
        h.taskPos = 1;
        CHECK(!pvc::positionError(h, bucket).empty());
        h.pass = 3; h.taskPos = 0;
        CHECK(!pvc::positionError(h, bucket).empty());
        h.pass = 1; h.taskPos = 0; h.terms = 58112;
        CHECK(!pvc::positionError(h, bucket).empty());             // more terms than tasks
        h.pass = ~0ULL; h.terms = 0;
        CHECK(!pvc::positionError(h, bucket).empty());
    }
    // Rotation: .new -> file, previous file -> .old.
    {
        const fs::path dir = fs::temp_directory_path() / ("pm1_vtrace_ckpt_header_test_" + std::to_string(::getpid()));
        fs::create_directories(dir);
        const std::string base = (dir / "c.ckpt").string();
        auto put = [](const std::string& f, const std::string& s) { std::ofstream(f, std::ios::trunc) << s; };
        auto get = [](const std::string& f) { std::ifstream in(f); std::string s; std::getline(in, s); return s; };
        CHECK(!pvc::commitNew(base));                  // nothing to commit
        CHECK(!fs::exists(base));
        put(base + ".new", "one");
        CHECK(pvc::commitNew(base));
        CHECK(get(base) == "one" && !fs::exists(base + ".new") && !fs::exists(base + ".old"));
        put(base + ".new", "two");
        CHECK(pvc::commitNew(base));
        CHECK(get(base) == "two" && get(base + ".old") == "one");
        put(base + ".new", "three");
        CHECK(pvc::commitNew(base));
        CHECK(get(base) == "three" && get(base + ".old") == "two");
        // removeStage2Checkpoints removes every variant with its satellites.
        const std::string cwd = fs::current_path().string();
        fs::current_path(dir);
        for (const std::string& f : pvc::stage2CheckpointFiles(677)) {
            for (const char* sfx : {"", ".old", ".new", ".backend", ".backend.new"}) put(f + sfx, "x");
        }
        put("pm1_m_677.ckpt", "stage1");                  // not a stage-2 file
        put("pm1_s2_vtrace_pair95_m_6771.ckpt", "other"); // another exponent
        pvc::removeStage2Checkpoints(677);
        for (const std::string& f : pvc::stage2CheckpointFiles(677)) {
            for (const char* sfx : {"", ".old", ".new", ".backend", ".backend.new"}) CHECK(!fs::exists(f + sfx));
        }
        CHECK(fs::exists("pm1_m_677.ckpt"));
        CHECK(fs::exists("pm1_s2_vtrace_pair95_m_6771.ckpt"));
        fs::current_path(cwd);
        fs::remove_all(dir);
    }
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "pm1 V-trace checkpoint header test passed\n";
    return 0;
}
