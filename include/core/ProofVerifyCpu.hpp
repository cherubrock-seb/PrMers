// core/ProofVerifyCpu.hpp
//
// Verification of a PRP proof on the CPU with GMP: the check Proof::verify
// does on the GPU, for the proof the CPU fallback (ProofManagerMarin) writes.
#pragma once

#include "core/ProofMarin.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <string>

namespace core {

// Check that `proof` is shaped as the proof of M<expectedE> at power
// `expectedPower`: the header, and well-formed non-zero residues. On false,
// `why` says what is wrong. Cheap: no arithmetic.
bool checkProofShape(const ProofMarin& proof, uint32_t expectedE, uint32_t expectedPower,
                     std::string& why);

// Verify `proof` as the proof of M<expectedE> at power `expectedPower`.
// Returns true when it is a valid proof. On false, `why` says what failed
// (a header that does not match, a malformed residue, or the check itself).
//
// Cost: the check ends with about E / 2^power modular squarings of E-bit
// numbers, after `power` exponentiations by 64-bit hashes; see
// estimateCpuVerifySeconds. Progress is written to `log` (null for none)
// every few seconds once the check runs longer than that.
bool verifyProofCpu(const ProofMarin& proof, uint32_t expectedE, uint32_t expectedPower,
                    std::string& why, std::ostream* log);

// ---- Policy for verifying the proof the CPU fallback writes ----
//
// The CPU check costs about E / 2^power GMP squarings (about 20 s at E = 1M,
// 26 min at 10M, 12.5 h at 100M), so it is not always done:
//   1. when a GPU verifier is available and works, the proof is verified on the
//      GPU, as the normal proof path verifies it;
//   2. otherwise, when the CPU check is estimated to take no more than a cap
//      (kDefaultCpuVerifyMaxSeconds), it is verified on the CPU;
//   3. otherwise it is not verified: the result says so, and the proof is kept.

// Default cap on the estimated time of a CPU verification. The environment
// variable PRMERS_CPU_PROOF_VERIFY_MAX_SECONDS overrides it (0: never verify
// on the CPU).
constexpr double kDefaultCpuVerifyMaxSeconds = 120.0;
constexpr const char* kCpuVerifyMaxSecondsEnv = "PRMERS_CPU_PROOF_VERIFY_MAX_SECONDS";

// The cap in seconds: the environment variable when it is a number >= 0, else
// the default.
double cpuVerifyMaxSeconds();

// Estimated seconds of verifyProofCpu: the squarings of the final step plus
// the modular exponentiations by the 64-bit hashes (about 96 squaring
// equivalents each, two per power), at a squaring time fitted to GMP (about
// 2.4 ms at E = 1M, growing as E^1.17).
double estimateCpuVerifySeconds(uint32_t E, uint32_t power);

// "45 s", "26 min", "12.5 h" (rounded, for the warning text).
std::string formatDuration(double seconds);

// Verifies the proof file with the GPU, as Proof::verify does: true when the
// proof verifies, false when it does not, and throws when the GPU (or its
// backend) is unusable. An empty function means no GPU is available.
using GpuProofVerifier = std::function<bool(const std::filesystem::path&)>;

struct FallbackVerifyResult {
    enum class Status { Verified, Failed, Skipped };
    Status status = Status::Skipped;
    // "GPU" or "CPU" when the proof was checked.
    std::string method;
    // Failed: what failed. Skipped: the warning to show.
    std::string message;
    double estimatedCpuSeconds = 0.0;
};

// Verify the fallback proof `file` of M<E> at `power` as the policy above says.
// Progress and the method used are written to `log` (null for none). A
// verification that fails or is skipped is reported in the result, not thrown.
FallbackVerifyResult verifyFallbackProof(const std::filesystem::path& file, uint32_t E, uint32_t power,
                                         const GpuProofVerifier& gpuVerify, double cpuMaxSeconds,
                                         std::ostream* log);

} // namespace core
