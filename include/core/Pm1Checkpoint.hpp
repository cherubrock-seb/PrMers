// include/core/Pm1Checkpoint.hpp
//
// Host-only helpers for the Marin P-1 stage-1 checkpoint (pm1_m_<p>.ckpt) and
// for the size limits of the exponent E that stage 1 raises the base to.
// Nothing here is used by an OpenCL kernel.
//
// Checkpoint header layout (all fields little-endian, as written by the host):
//
//   version 3 (legacy): int32 version | uint32 p | uint32 counter | double et | ...
//   version 4         : int32 version | uint32 p | uint64 counter | double et | ...
//
// "counter" is the number of bits of the current E chunk (or of E_diff for a
// -b1old extension) that are still to be processed.  With a chunk larger than
// 2^32 bits (-maxe 512 or more, an unchunked -b1old extension, ...) the 32-bit
// field of version 3 truncated it, and a resume restarted at the wrong bit.
// Version 4 stores it in 64 bits.  The remainder of the file is identical in
// both versions, so a version 3 file is read by widening its counter.
//
// An older binary only accepts version 3 and rejects a version 4 file as
// unreadable (it never mis-resumes from one); see the PR notes.
#pragma once

#include <cstdint>
#include <limits>
#include <string>

namespace core {
namespace pm1ckpt {

inline constexpr int kStage1VersionLegacy32 = 3;  // 32-bit counter
inline constexpr int kStage1Version = 4;          // 64-bit counter (written by this build)

inline bool stage1VersionKnown(int version) {
    return version == kStage1VersionLegacy32 || version == kStage1Version;
}

// Reads the counter that follows the exponent field, sized by the file version.
// F is any type with `bool read(char*, size_t)` (the marin File class).
template <class F>
inline bool readCounter(F& f, int version, uint64_t& counter) {
    if (version == kStage1VersionLegacy32) {
        uint32_t c32 = 0;
        if (!f.read(reinterpret_cast<char*>(&c32), sizeof(c32))) return false;
        counter = c32;
        return true;
    }
    if (version == kStage1Version) {
        uint64_t c64 = 0;
        if (!f.read(reinterpret_cast<char*>(&c64), sizeof(c64))) return false;
        counter = c64;
        return true;
    }
    return false;
}

// Reads  version | p | counter | et  and checks the version and the exponent.
// Returns false (leaving the caller to treat the file as unusable) for an
// unknown version, a different exponent or a short read.
template <class F>
inline bool readHeader(F& f, uint32_t expectedP, uint64_t& counter, double& et, int* versionOut = nullptr) {
    int version = 0;
    if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return false;
    if (!stage1VersionKnown(version)) return false;
    uint32_t rp = 0;
    if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return false;
    if (rp != expectedP) return false;
    if (!readCounter(f, version, counter)) return false;
    if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return false;
    if (versionOut) *versionOut = version;
    return true;
}

// Writes the current (version 4) header.  F has `bool write(const char*, size_t)`.
template <class F>
inline bool writeHeader(F& f, uint32_t p, uint64_t counter, double et) {
    const int version = kStage1Version;
    if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return false;
    if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return false;
    if (!f.write(reinterpret_cast<const char*>(&counter), sizeof(counter))) return false;
    if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Size limits of E.
//
// E = lcm(1..B1) * 2p has about 1.4427*B1 + 2p bits.  The Marin stage 1 builds
// it in chunks of at most -maxe bits, so only one chunk has to fit in an mpz
// number; the legacy driver, the -b1old extension (E_diff), -torus, the
// Gaussian-Mersenne P-1, the ultra-low-memory stage 2 (product exponent up to
// B2) and the stage-4 exponent (-tbits) build the whole exponent at once.
// Every one of them walks E with mpz_tstbit/mpz_sizeinbase, whose bit index is
// GMP's mp_bitcnt_t, i.e. unsigned long: 64-bit on Linux/macOS, 32-bit on
// Windows (LLP64).  GMP also caps an mpz at 2^31-1 limbs (about 2^37 bits).
// The bound below is the smaller of 2^36 bits and half the index range, and it
// is a template on the index type so the 32-bit case can be tested on Linux.
// ---------------------------------------------------------------------------

template <class IndexT = unsigned long>
inline constexpr uint64_t maxEBitsFor() {
    constexpr uint64_t idxMax = static_cast<uint64_t>(std::numeric_limits<IndexT>::max());
    constexpr uint64_t gmpCap = 1ULL << 36;
    return (idxMax / 2 + 1) < gmpCap ? (idxMax / 2 + 1) : gmpCap;
}

// Largest B1 accepted: 2^62 so every 64-bit sieve/product expression stays exact,
// and no more than the index type, because the single-piece prime-power helpers
// of buildE() go through unsigned long.
template <class IndexT = unsigned long>
inline constexpr uint64_t maxB1For() {
    constexpr uint64_t idxMax = static_cast<uint64_t>(std::numeric_limits<IndexT>::max());
    return idxMax < (1ULL << 62) ? idxMax : (1ULL << 62);
}

inline constexpr uint64_t maxEBits() { return maxEBitsFor<unsigned long>(); }
inline constexpr uint64_t kMaxB1 = maxB1For<unsigned long>();

inline constexpr double kLog2E = 1.4426950408889634;

// Bits of the exponent built in one piece for bounds (B1old, B1] (B1old = 0: from scratch).
inline double estimateEBits(uint64_t B1, uint64_t B1old, uint64_t exponent) {
    const uint64_t lo = (B1old > 0 && B1old < B1) ? B1old : 0;
    return kLog2E * static_cast<double>(B1 - lo) + 2.0 * static_cast<double>(exponent) + 64.0;
}

// Empty when the request can be run; otherwise a one-line reason.
//   unchunked: the driver builds E (or E_diff) in a single mpz number.
template <class IndexT = unsigned long>
inline std::string limitError(uint64_t B1, uint64_t B1old, uint64_t maxEBitsOpt, uint64_t exponent, bool unchunked) {
    const uint64_t capBits = maxEBitsFor<IndexT>();
    const uint64_t capB1 = maxB1For<IndexT>();
    if (B1 > capB1) {
        return "B1=" + std::to_string(B1) + " is too large (the largest supported B1 is " + std::to_string(capB1) + ")";
    }
    if (maxEBitsOpt > capBits) {
        return "-maxe asks for chunks of " + std::to_string(maxEBitsOpt) + " bits; the largest supported chunk is " +
               std::to_string(capBits) + " bits (" + std::to_string(capBits >> 23) + " MiB)";
    }
    if (unchunked && estimateEBits(B1, B1old, exponent) > static_cast<double>(capBits)) {
        return "B1=" + std::to_string(B1) + (B1old ? " (extending from -b1old " + std::to_string(B1old) + ")" : std::string()) +
               " needs an exponent of about " + std::to_string(static_cast<uint64_t>(estimateEBits(B1, B1old, exponent))) +
               " bits, more than the " + std::to_string(capBits) +
               " bits this P-1 path can build in one piece; use the default chunked Marin stage 1 (without -b1old, -torus or the legacy engine)";
    }
    return std::string();
}

// The stage-4 exponent (-tbits) is also built whole.
template <class IndexT = unsigned long>
inline std::string tbitsError(uint64_t tbits) {
    const uint64_t capBits = maxEBitsFor<IndexT>();
    if (tbits > capBits) {
        return "-tbits " + std::to_string(tbits) + " is too large (the largest supported exponent size is " +
               std::to_string(capBits) + " bits)";
    }
    return std::string();
}


// All P-1 limit checks for one set of options (duck-typed on CliOptions so it can
// be tested with a stand-in): B1/-maxe/whole-E paths, the ultra-low-memory
// stage 2 product exponent (about 1.4427*B2 bits) and -tbits.
template <class IndexT = unsigned long, class O>
inline std::string optionsLimitError(const O& o) {
    if (o.mode != "pm1" && o.mode != "gm-pm1") return std::string();
    const bool gm = (o.mode == "gm-pm1");
    const bool unchunked = gm || !o.marin || o.torus || (o.B1old > 0 && o.B1 > o.B1old);
    std::string err = limitError<IndexT>(o.B1, gm ? 0 : o.B1old, o.max_e_bits, o.exponent, unchunked);
    if (!err.empty()) return err;
    if (!gm && o.pm1_ultralowmem && o.pm1_lowmem && o.B2 > o.B1) {
        err = limitError<IndexT>(o.B2, 0, 0, o.exponent, true);
        if (!err.empty()) return "ultra-low-memory stage 2 (B2): " + err;
    }
    return tbitsError<IndexT>(o.tbits);
}

}  // namespace pm1ckpt
}  // namespace core
