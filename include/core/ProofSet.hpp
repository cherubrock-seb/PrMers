#pragma once

#include "core/Proof.hpp"
#include <cstdint>
#include <vector>
#include <filesystem>

// Forward declarations for GPU types
namespace opencl {
    class NttEngine;
    class Context;
}

namespace math {
    class Carry;
}

typedef struct _cl_mem* cl_mem;

namespace core {

// Helper class to group commonly used GPU parameters and operations
class GpuContext {
public:
    uint32_t exponent;
    const opencl::Context& ctx;
    opencl::NttEngine& ntt;
    math::Carry& carry;
    const std::vector<int>& digitWidth;
    size_t limbBytes;
    
    GpuContext(uint32_t exponent_, const opencl::Context& ctx_, opencl::NttEngine& ntt_, 
               math::Carry& carry_, const std::vector<int>& digitWidth_, size_t limbBytes_)
        : exponent(exponent_), ctx(ctx_), ntt(ntt_), carry(carry_), digitWidth(digitWidth_), limbBytes(limbBytes_) {}
    
    // GPU data transfer methods
    std::vector<uint32_t> read(cl_mem buffer) const;
    void write(cl_mem buffer, const std::vector<uint32_t>& data) const;
};

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
    const std::vector<std::string> knownFactors; // known factors (for cofactor tests)

    ProofSet(uint32_t exponent, uint32_t proofLevel, std::vector<std::string> factors = {});

    bool shouldCheckpoint(uint32_t iter) const;
    void save(uint32_t iter, const std::vector<uint32_t>& words);
    std::vector<uint32_t> load(uint32_t iter) const;

    static Words fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);
    static uint32_t bestPower(uint32_t E);
    static bool isInPoints(uint32_t E, uint32_t power, uint32_t k);
    static std::filesystem::path proofPath(uint32_t E);
    static double diskUsageGB(uint32_t E, uint32_t power);
    
    // Core proof generation algorithm
    Proof computeProof(const GpuContext& gpu) const;

private:
    std::vector<uint32_t> points; // checkpoint iteration points
    
    bool isValidTo(uint32_t limitK) const;
    bool fileExists(uint32_t k) const;
};

} // namespace core
