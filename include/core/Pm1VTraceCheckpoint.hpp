// include/core/Pm1VTraceCheckpoint.hpp
//
// Host-only helpers for the checkpoint of the default P-1 stage 2, the V-trace
// stage 2 with the pair95 planner (pm1_s2_vtrace_pair95_m_<p>.ckpt).
//
// File layout (all fields little-endian, as written by the host):
//
//   int32  version (25)
//   uint64 exponent | B1 | B2                       job
//   uint64 D | L | babies | active babies | passes  plan
//   uint64 tasks | plan hash | neg-baby-add flag
//   uint64 pass | task position | k                 position: the next task to run
//   uint64 terms | primes                           counters (for the result line)
//   double elapsed seconds
//   uint32 register count, then per register: uint32 index | register image
//   uint32 CRC (marin File::write_crc32)
//
// Every counter is 64-bit (the P-1 checkpoint convention, see Pm1Checkpoint.hpp).
// The plan fields and the hash of the task list make sure a checkpoint is only
// resumed by a run that executes exactly the same task list in the same passes:
// D, L and the baby window are chosen from the device memory, so the same job
// can get another plan on another GPU.  The backend (Marin or Aevum) is recorded
// in the .backend sidecar, like for the other P-1 checkpoints.
#pragma once

#include "core/CheckpointLocation.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <system_error>
#include <filesystem>
#include <vector>

namespace core {
namespace pm1vtrace {

inline constexpr int kPair95Version = 25;

struct Pair95Header {
    // job
    uint64_t exponent = 0, B1 = 0, B2 = 0;
    // plan
    uint64_t D = 0, L = 0, babies = 0, activeBabies = 0, passes = 0;
    uint64_t tasks = 0, planHash = 0, negAdd = 0;
    // position
    uint64_t pass = 0, taskPos = 0, k = 0;
    uint64_t terms = 0, primes = 0;
    double elapsed = 0.0;
};

// Hash of a sequence of 64-bit words (FNV-1a over splitmix64-mixed words), used
// for the plan hash.  Only a fingerprint: it is compared, never trusted alone.
struct PlanHash {
    uint64_t h = 1469598103934665603ULL;
    void add(uint64_t v) {
        v += 0x9e3779b97f4a7c15ULL;
        v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
        v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
        v ^= v >> 31;
        h ^= v;
        h *= 1099511628211ULL;
    }
};

template <class F>
inline bool writeU64(F& f, uint64_t v) { return f.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
template <class F>
inline bool readU64(F& f, uint64_t& v) { return f.read(reinterpret_cast<char*>(&v), sizeof(v)); }

template <class F>
inline bool writeHeader(F& f, const Pair95Header& h) {
    const int32_t version = kPair95Version;
    if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return false;
    const uint64_t fields[] = {h.exponent, h.B1, h.B2, h.D, h.L, h.babies, h.activeBabies, h.passes,
                               h.tasks, h.planHash, h.negAdd, h.pass, h.taskPos, h.k, h.terms, h.primes};
    for (uint64_t v : fields) if (!writeU64(f, v)) return false;
    return f.write(reinterpret_cast<const char*>(&h.elapsed), sizeof(h.elapsed));
}

enum class ReadStatus { Ok, BadVersion, Short };

template <class F>
inline ReadStatus readHeader(F& f, Pair95Header& h, int32_t* versionOut = nullptr) {
    int32_t version = 0;
    if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return ReadStatus::Short;
    if (versionOut) *versionOut = version;
    if (version != kPair95Version) return ReadStatus::BadVersion;
    uint64_t* fields[] = {&h.exponent, &h.B1, &h.B2, &h.D, &h.L, &h.babies, &h.activeBabies, &h.passes,
                          &h.tasks, &h.planHash, &h.negAdd, &h.pass, &h.taskPos, &h.k, &h.terms, &h.primes};
    for (uint64_t* v : fields) if (!readU64(f, *v)) return ReadStatus::Short;
    if (!f.read(reinterpret_cast<char*>(&h.elapsed), sizeof(h.elapsed))) return ReadStatus::Short;
    return ReadStatus::Ok;
}

// Empty when a checkpoint with header `file` can be resumed by the run `run`
// (whose position fields are ignored); otherwise a one-line reason naming the
// first field that differs.
inline std::string mismatch(const Pair95Header& file, const Pair95Header& run) {
    struct Field { const char* name; uint64_t a, b; };
    const Field fields[] = {
        {"exponent", file.exponent, run.exponent},
        {"B1", file.B1, run.B1},
        {"B2", file.B2, run.B2},
        {"D", file.D, run.D},
        {"L", file.L, run.L},
        {"baby count", file.babies, run.babies},
        {"active babies per pass", file.activeBabies, run.activeBabies},
        {"pass count", file.passes, run.passes},
        {"task count", file.tasks, run.tasks},
        {"plan hash", file.planHash, run.planHash},
        {"negative-baby mode", file.negAdd, run.negAdd},
    };
    for (const Field& fd : fields) {
        if (fd.a != fd.b) {
            return std::string(fd.name) + " differs (checkpoint " + std::to_string(fd.a) +
                   ", this run " + std::to_string(fd.b) + ")";
        }
    }
    return std::string();
}

// Position sanity: the pass and task position must lie inside the plan.  A
// position equal to the end (pass == passes, taskPos == 0) means the loop is
// done and only the GCD is left.  `bucketSize` returns the task count of a pass.
template <class BucketSize>
inline std::string positionError(const Pair95Header& h, BucketSize bucketSize) {
    if (h.terms > h.tasks) return "term count " + std::to_string(h.terms) + " is beyond the " + std::to_string(h.tasks) + " tasks of the plan";
    if (h.pass > h.passes) return "pass " + std::to_string(h.pass) + " is beyond the " + std::to_string(h.passes) + " passes of the plan";
    if (h.pass == h.passes) {
        if (h.taskPos != 0) return "task position " + std::to_string(h.taskPos) + " after the last pass";
        return std::string();
    }
    const uint64_t n = bucketSize(h.pass);
    if (h.taskPos > n) return "task position " + std::to_string(h.taskPos) + " is beyond the " + std::to_string(n) + " tasks of pass " + std::to_string(h.pass);
    return std::string();
}

// Replaces `base` by `base.new` and keeps the previous checkpoint as `base.old`.
// Returns false (leaving `base` as it was) when `base.new` does not exist or a
// rename fails.
inline bool commitNew(const std::string& base) {
    namespace fs = std::filesystem;
    const std::string newf = base + ".new", oldf = base + ".old";
    std::error_code ec;
    if (!fs::exists(newf, ec)) return false;
    if (fs::exists(base, ec)) {
        fs::remove(oldf, ec);
        ec.clear();
        fs::rename(base, oldf, ec);
        if (ec) return false;
    }
    fs::rename(newf, base, ec);
    return !ec;
}

// Base names of the checkpoints of every Marin P-1 stage-2 variant: classic
// BSGS and V-trace (linear, neg-baby, product-tree, pair95).
inline std::vector<std::string> stage2CheckpointFiles(uint64_t exponent) {
    const std::string p = std::to_string(exponent);
    return {
        "pm1_s2_m_" + p + ".ckpt",
        "pm1_s2_vtrace_m_" + p + ".ckpt",
        "pm1_s2_vtrace_negadd_m_" + p + ".ckpt",
        "pm1_s2_vtrace_producttree_m_" + p + ".ckpt",
        "pm1_s2_vtrace_pair95_m_" + p + ".ckpt",
    };
}

// Removes them, under the save path (-f; the working directory when empty), with their .old/.new
// files and their backend sidecars, and any copy in the working directory this run adopted
// (core/CheckpointLocation.hpp).
inline void removeStage2Checkpoints(uint64_t exponent, const std::string& savePath = "") {
    for (const std::string& base : stage2CheckpointFiles(exponent)) {
        CheckpointLocation(savePath, base).removeAll();
    }
}

}  // namespace pm1vtrace
}  // namespace core
