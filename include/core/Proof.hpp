#pragma once

#include <cstdint>
#include <vector>
#include <filesystem>

namespace core {

class GpuContext;

class Proof {
public:
    const uint32_t E;
    const std::vector<uint32_t> B;
    const std::vector<std::vector<uint32_t>> middles;
    const std::vector<std::string> knownFactors;

    Proof(uint32_t exponent, std::vector<uint32_t> finalResidue, std::vector<std::vector<uint32_t>> intermediateResidues, std::vector<std::string> factors = {})
        : E(exponent), B(std::move(finalResidue)), middles(std::move(intermediateResidues)), knownFactors(std::move(factors)) {}

    // File I/O methods for proof files
    void save(const std::filesystem::path& filePath) const;
    static Proof load(const std::filesystem::path& filePath);
    
    // Hash functions for proof generation
    static std::array<uint64_t, 4> hashWords(uint32_t E, const std::vector<uint32_t>& words);
    static std::array<uint64_t, 4> hashWords(uint32_t E, 
                                           const std::array<uint64_t, 4>& hash,
                                           const std::vector<uint32_t>& words);
    static uint64_t res64(const std::vector<uint32_t>& words);

    // Proof verification
    bool verify(const GpuContext& gpu, uint32_t npower) const;
};

} // namespace core
