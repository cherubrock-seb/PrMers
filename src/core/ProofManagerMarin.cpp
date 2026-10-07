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
#include <stdexcept>
#include <string>

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
        
        // Create proof file name: {exponent}-{power}.proof, in proof/ like the
        // GPU proof writer, so both paths leave the file in the same place.
        std::string filename = std::to_string(exponent_) + "-" + 
                              std::to_string(proof.middles.size()) + ".proof";
        const std::filesystem::path proofDir = std::filesystem::current_path() / "proof";
        std::filesystem::create_directories(proofDir);
        const std::filesystem::path proofFilePath = proofDir / filename;
        const std::filesystem::path tmpPath = proofDir / (filename + ".tmp");

        // Save under a temporary name and check it loads back before it takes
        // the final name: a proof that cannot be read is not reported.
        std::error_code ec;
        try {
            proof.save(tmpPath);
            auto loadedProof = ProofMarin::load(tmpPath);
            if (loadedProof.E != proof.E || loadedProof.B != proof.B ||
                loadedProof.middles != proof.middles) {
                throw std::runtime_error("proof file does not read back as written");
            }
        } catch (const std::exception& e) {
            std::filesystem::remove(tmpPath, ec);
            throw std::runtime_error(std::string("Proof file validation failed: ") + e.what());
        }
        std::filesystem::remove(proofFilePath, ec);
        std::filesystem::rename(tmpPath, proofFilePath);

        return proofFilePath;
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating proof file: " << e.what() << std::endl;
        throw;
    }
}

} // namespace core
