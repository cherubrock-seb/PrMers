/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
#include "core/ProofSet.hpp"

namespace core {

// Words
Words::Words() = default;

Words::Words(const std::vector<uint64_t>& v)
  : data_{v} {}

const std::vector<uint64_t>& Words::data() const noexcept {
    return data_;
}

Words Words::fromUint64(const std::vector<uint64_t>& host, uint32_t exponent) {
    (void)exponent;
    return Words(host);
}

// ProofSet
ProofSet::ProofSet(uint32_t exponent, int proofLevel) {
    (void)exponent;
    (void)proofLevel;
}

bool ProofSet::shouldCheckpoint(uint32_t iter) const {
    (void)iter;
    return false;
}

void ProofSet::save(uint32_t iter, const Words& words) {
    (void)iter;
    (void)words;
}

Words ProofSet::fromUint64(const std::vector<uint64_t>& host, uint32_t exponent) {
    return Words::fromUint64(host, exponent);
}

} // namespace core
