#pragma once

#include "core/Proof.hpp"
#include <cstdint>
#include <vector>
#include <filesystem>
#include <gmpxx.h>

namespace core {

class Words {
public:
    Words();
    explicit Words(const std::vector<uint64_t>& v);

    const std::vector<uint64_t>& data() const noexcept;
    std::vector<uint64_t>& data() noexcept;

    static Words fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);

private:
    std::vector<uint64_t> data_;
};

class ProofSet {
public:
    const uint32_t E;     // exponent
    const uint32_t power; // proof power level

    ProofSet(uint32_t exponent, uint32_t proofLevel);

    bool shouldCheckpoint(uint32_t iter) const;
    void save(uint32_t iter, const std::vector<uint32_t>& words);
    std::vector<uint32_t> load(uint32_t iter) const;

    static Words fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);
    static uint32_t bestPower(uint32_t E);
    static bool isInPoints(uint32_t E, uint32_t power, uint32_t k);
    static std::filesystem::path proofPath(uint32_t E);
    static double diskUsageGB(uint32_t E, uint32_t power);
    
    // Core proof generation algorithm
    Proof computeProof() const;

private:
    std::vector<uint32_t> points; // checkpoint iteration points
    
    bool isValidTo(uint32_t limitK) const;
    bool fileExists(uint32_t k) const;
    
    // GMP-based modular arithmetic helpers
    mpz_class convertToGMP(const std::vector<uint32_t>& words) const;
    std::vector<uint32_t> convertFromGMP(const mpz_class& gmp_val) const;
    mpz_class mersenneReduce(const mpz_class& x, uint32_t E) const;
    mpz_class mersennePowMod(const mpz_class& base, uint64_t exp, uint32_t E) const;
};

} // namespace core
