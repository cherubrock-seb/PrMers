// core/ProofManagerMarin.cpp
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
#include "core/ProofManagerMarin.hpp"
#include "io/JsonBuilder.hpp"
#include <vector>
#include <iostream>

namespace core {

ProofManagerMarin::ProofManagerMarin(uint32_t exponent, int proofLevel,
                           cl_command_queue queue, uint32_t n,
                           const std::vector<int>& digitWidth,
                           const std::vector<std::string>& knownFactors)
  : proofSet_(exponent, static_cast<uint32_t>(proofLevel), knownFactors)
  , queue_(queue)
  , n_(n)
  , exponent_(exponent)
  , digitWidth_(digitWidth)
{}

void ProofManagerMarin::checkpoint(cl_mem buf, uint32_t iter) {
    if (! proofSet_.shouldCheckpoint(iter)) return;

    // read back the buffer from GPU
    std::vector<uint64_t> host(n_);
    clEnqueueReadBuffer(queue_, buf, CL_TRUE, 0,
                        n_ * sizeof(uint64_t),
                        host.data(), 0, nullptr, nullptr);

    // Get residue from NTT buffer using compactBits
    auto words = io::JsonBuilder::compactBits(host, digitWidth_, exponent_);
    
    // Save in PRPLL-compatible format
    saveProofCheckpointVerified(proofSet_, iter, words);
}

bool ProofManagerMarin::shouldCheckpoint(uint32_t iter) const {
  return proofSet_.shouldCheckpoint(iter);
}

void ProofManagerMarin::checkpointMarin(engine::digit host, uint32_t iter)
{
    if (!proofSet_.shouldCheckpoint(iter)) return;

    digitWidth_.resize(host.get_size());
    std::vector<uint64_t> digits(host.get_size());

    for (size_t i = 0; i < host.get_size(); ++i)
    {
        digits[i] = host.val(i);
        digitWidth_[i] = static_cast<uint8_t>(host.width(i));
    }

    auto words = io::JsonBuilder::compactBits(digits, digitWidth_, exponent_);
    saveProofCheckpointVerified(proofSet_, iter, words);
}


std::filesystem::path ProofManagerMarin::proof() const {
    try {
        // Generate proof from collected checkpoints
        ProofMarin proof = proofSet_.computeProof();
        
        // Create proof file name: {exponent}-{power}.proof
        std::string filename = std::to_string(exponent_) + "-" + 
                              std::to_string(proof.middles.size()) + ".proof";
        std::filesystem::path proofFilePath = std::filesystem::current_path() / filename;
        
        
        // Save the proof file
        proof.save(proofFilePath);
        
        // Check the proof was saved correctly by attempting to load it
        try {
            auto loadedProof = ProofMarin::load(proofFilePath);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Proof file validation failed: " << e.what() << std::endl;
        }
        
        return proofFilePath;
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating proof file: " << e.what() << std::endl;
        throw;
    }
}

} // namespace core