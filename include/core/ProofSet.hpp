#pragma once

#include <cstdint>
#include <vector>

namespace core {

class Words {
public:
    Words();
    explicit Words(const std::vector<uint64_t>& v);

    const std::vector<uint64_t>& data() const noexcept;

    static Words fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);

private:
    std::vector<uint64_t> data_;
};

class ProofSet {
public:
    ProofSet(uint32_t exponent, int proofLevel);

    bool shouldCheckpoint(uint32_t iter) const;
    void save(uint32_t iter, const Words& words);

    static Words fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);
};

} // namespace core
