#pragma once

#include <cstdint>
#include <vector>
#include <filesystem>

namespace core {

class Proof {
public:
    const uint64_t E;
    const std::vector<uint64_t> B;
    const std::vector<std::vector<uint64_t>> middles;

    Proof(uint64_t exponent, std::vector<uint64_t> finalResidue, std::vector<std::vector<uint64_t>> intermediateResidues)
        : E(exponent), B(std::move(finalResidue)), middles(std::move(intermediateResidues)) {}

    // File I/O methods for proof files
    void save(const std::filesystem::path& filePath) const;
    static Proof load(const std::filesystem::path& filePath);
    
    // Hash functions for proof generation
    static std::array<uint64_t, 4> hashWords(uint64_t E, const std::vector<uint64_t>& words);
    static std::array<uint64_t, 4> hashWords(uint64_t E, 
                                           const std::array<uint64_t, 4>& hash,
                                           const std::vector<uint64_t>& words);
    static uint64_t res64(const std::vector<uint64_t>& words);
};

} // namespace core
