#pragma once

#include "core/ProofMarin.hpp"
#include <cstdint>
#include <vector>
#include <filesystem>

namespace core {

class WordsMarin {
public:
    WordsMarin();
    explicit WordsMarin(const std::vector<uint64_t>& v);

    const std::vector<uint64_t>& data() const noexcept;
    std::vector<uint64_t>& data() noexcept;

    static WordsMarin fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);

private:
    std::vector<uint64_t> data_;
};

class ProofSetMarin {
public:
    const uint32_t E;     // exponent
    const uint32_t power; // proof power level
    const std::vector<std::string> knownFactors; // known factors (for cofactor tests)

    ProofSetMarin(uint32_t exponent, uint32_t proofLevel, std::vector<std::string> factors = {});

    bool shouldCheckpoint(uint32_t iter) const;
    void save(uint32_t iter, const std::vector<uint32_t>& words);
    std::vector<uint32_t> load(uint32_t iter) const;

    static WordsMarin fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);
    static uint32_t bestPower(uint32_t E);
    static bool isInPoints(uint32_t E, uint32_t power, uint32_t k);
    static std::filesystem::path proofPath(uint32_t E);
    static double diskUsageGB(uint32_t E, uint32_t power);
    
    // Core proof generation algorithm
    ProofMarin computeProof() const;

private:
    std::vector<uint32_t> points; // checkpoint iteration points
    
    bool isValidTo(uint32_t limitK) const;
    bool fileExists(uint32_t k) const;
};

} // namespace core