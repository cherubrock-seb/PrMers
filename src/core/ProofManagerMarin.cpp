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
  : proofSet_(exponent, proofLevel, knownFactors)
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
    proofSet_.save(iter, words);
    
    // Verify the checkpoint by loading it back and comparing
    try {
        auto loadedWords = proofSet_.load(iter);
        
        // Compare the saved and loaded data
        if (words.size() != loadedWords.size()) {
            std::cerr << "Warning: Checkpoint validation failed: size mismatch (" 
                      << words.size() << " vs " << loadedWords.size() << ")" << std::endl;
            return;
        }
        
        for (size_t i = 0; i < words.size(); ++i) {
            if (words[i] != loadedWords[i]) {
                std::cerr << "Warning: Checkpoint validation failed: data mismatch at word " 
                          << i << " (0x" << words[i] << " vs 0x" << loadedWords[i] << ")" << std::endl;
                return;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Checkpoint validation failed at iteration " << iter 
                  << ": " << e.what() << std::endl;
    }
}

bool ProofManagerMarin::shouldCheckpoint(uint32_t iter) const {
  return proofSet_.shouldCheckpoint(iter);
}

void ProofManagerMarin::checkpointMarin(std::vector<uint64_t> host, uint32_t iter)
{
    if (!proofSet_.shouldCheckpoint(iter)) return;

    digitWidth_.resize(host.size());
    std::vector<uint64_t> digits(host.size());

    for (size_t i = 0; i < host.size(); ++i)
    {
        uint64_t x = host[i];
        uint32_t v = static_cast<uint32_t>(x);
        uint32_t w = static_cast<uint8_t>(x >> 32);
        digitWidth_[i] = static_cast<uint8_t>(w);
        if (w == 0) digits[i] = 0;
        else if (w >= 32) digits[i] = static_cast<uint64_t>(v);
        else digits[i] = static_cast<uint64_t>(v & ((1u << w) - 1));
    }

    auto words = io::JsonBuilder::compactBits(digits, digitWidth_, exponent_);
    proofSet_.save(iter, words);

    try
    {
        auto loadedWords = proofSet_.load(iter);
        if (words.size() != loadedWords.size())
        {
            std::cerr << "Warning: Checkpoint validation failed: size mismatch (" << words.size() << " vs " << loadedWords.size() << ")" << std::endl;
            return;
        }
        for (size_t i = 0; i < words.size(); ++i)
        {
            if (words[i] != loadedWords[i])
            {
                std::cerr << "Warning: Checkpoint validation failed: data mismatch at word " << i << " (0x" << words[i] << " vs 0x" << loadedWords[i] << ")" << std::endl;
                return;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Warning: Checkpoint validation failed at iteration " << iter << ": " << e.what() << std::endl;
    }
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