// core/ProofManager.cpp
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
#include "core/ProofManager.hpp"
#include "io/JsonBuilder.hpp"
#include <vector>
#include <iostream>

namespace core {

ProofManager::ProofManager(uint32_t exponent, int proofLevel,
                           cl_command_queue queue, uint32_t n,
                           const std::vector<int>& digitWidth,
                           const std::vector<std::string>& knownFactors)
  : proofSet_(exponent, proofLevel, knownFactors)
  , queue_(queue)
  , n_(n)
  , exponent_(exponent)
  , digitWidth_(digitWidth)
{}

void ProofManager::checkpoint(cl_mem buf, uint32_t iter) {
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
void ProofManager::checkpointMarin(std::vector<uint64_t> host, uint32_t iter) {
    if (! proofSet_.shouldCheckpoint(iter)) return;

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

std::filesystem::path ProofManager::proof(const opencl::Context& ctx, opencl::NttEngine& ntt, math::Carry& carry, bool verify) const {
    try {
                    
        // Calculate limbBytes for GPU proof generation
        size_t limbBytes = n_ * sizeof(uint64_t);

        // Create GPU context for proof generation
        core::GpuContext gpu(exponent_, ctx, ntt, carry, digitWidth_, limbBytes);
        
        // Generate proof from collected checkpoints
        Proof proof = proofSet_.computeProof(gpu);
        
        // Create proof file name: {exponent}-{power}.proof
        std::string filename = std::to_string(exponent_) + "-" + 
                               std::to_string(proof.middles.size()) + ".proof";
        std::filesystem::path proofFilePath = std::filesystem::current_path() / filename;
        
        // Save the proof file
        proof.save(proofFilePath);
        
        try {
            // Check the proof was saved correctly by attempting to load it
            auto loadedProof = Proof::load(proofFilePath);
            // Verify generated proof
            if (verify) {
                loadedProof.verify(gpu);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Proof file verification failed: " << e.what() << std::endl;
        }
        
        return proofFilePath;
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating proof file: " << e.what() << std::endl;
        throw;
    }
}

} // namespace core
