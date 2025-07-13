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

    static Words fromUint64(const std::vector<uint64_t>& host, uint64_t exponent);

private:
    std::vector<uint64_t> data_;
};

class ProofSet {
public:
    const uint64_t E;     // exponent
    const uint64_t power; // proof power level

    ProofSet(uint64_t exponent, uint64_t proofLevel);

    bool shouldCheckpoint(uint64_t iter) const;
    void save(uint64_t iter, const std::vector<uint64_t>& words);
    std::vector<uint64_t> load(uint64_t iter) const;

    static Words fromUint64(const std::vector<uint64_t>& host, uint64_t exponent);
    static uint64_t bestPower(uint64_t E);
    static bool isInPoints(uint64_t E, uint64_t power, uint64_t k);
    static std::filesystem::path proofPath(uint64_t E);
    static double diskUsageGB(uint64_t E, uint64_t power);
    
    // Core proof generation algorithm
    Proof computeProof() const;

private:
    std::vector<uint64_t> points; // checkpoint iteration points
    
    bool isValidTo(uint64_t limitK) const;
    bool fileExists(uint64_t k) const;
    
    // GMP-based modular arithmetic helpers
    mpz_class convertToGMP(const std::vector<uint64_t>& words) const;
    std::vector<uint64_t> convertFromGMP(const mpz_class& gmp_val) const;
    mpz_class mersenneReduce(const mpz_class& x, uint64_t E) const;
    mpz_class mersennePowMod(const mpz_class& base, uint64_t exp, uint64_t E) const;
};

} // namespace core
