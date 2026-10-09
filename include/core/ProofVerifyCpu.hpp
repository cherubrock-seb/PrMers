// core/ProofVerifyCpu.hpp
//
// Verification of a PRP proof on the CPU with GMP: the check Proof::verify
// does on the GPU, for the proof the CPU fallback (ProofManagerMarin) writes.
#pragma once

#include "core/ProofMarin.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>

namespace core {

// Verify `proof` as the proof of M<expectedE> at power `expectedPower`.
// Returns true when it is a valid proof. On false, `why` says what failed
// (a header that does not match, a malformed residue, or the check itself).
//
// Cost: the check ends with about E / 2^power modular squarings of E-bit
// numbers, after `power` exponentiations by 64-bit hashes. That is of the
// order of the work of generating the proof on the CPU (about 2^power
// exponentiations by 64-bit hashes), so the verification at most doubles the
// time of the fallback. Progress is written to `log` (null for none) every
// few seconds once the check runs longer than that.
bool verifyProofCpu(const ProofMarin& proof, uint32_t expectedE, uint32_t expectedPower,
                    std::string& why, std::ostream* log);

} // namespace core
