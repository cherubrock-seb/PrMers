#pragma once

#include <cstdint>
#include <vector>
#include <filesystem>

namespace core {

class Proof {
public:
    const uint32_t E;
    const std::vector<uint32_t> B;
    const std::vector<std::vector<uint32_t>> middles;

    Proof(uint32_t exponent, std::vector<uint32_t> finalResidue, std::vector<std::vector<uint32_t>> intermediateResidues)
        : E(exponent), B(std::move(finalResidue)), middles(std::move(intermediateResidues)) {}

    // File I/O methods for proof files
    void save(const std::filesystem::path& filePath) const;
    static Proof load(const std::filesystem::path& filePath);
    
    // Hash functions for proof generation
    static std::array<uint64_t, 4> hashWords(uint32_t E, const std::vector<uint32_t>& words);
    static std::array<uint64_t, 4> hashWords(uint32_t E, 
                                           const std::array<uint64_t, 4>& hash,
                                           const std::vector<uint32_t>& words);
    static uint64_t res64(const std::vector<uint32_t>& words);
};

} // namespace core
