#include "core/ProofSet.hpp"

namespace core {

// Words
Words::Words() = default;

Words::Words(const std::vector<uint64_t>& v)
  : data_{v} {}

const std::vector<uint64_t>& Words::data() const noexcept {
    return data_;
}

Words Words::fromUint64(const std::vector<uint64_t>& host, uint32_t exponent) {
    (void)exponent;
    return Words(host);
}

// ProofSet
ProofSet::ProofSet(uint32_t exponent, int proofLevel) {
    (void)exponent;
    (void)proofLevel;
}

bool ProofSet::shouldCheckpoint(uint32_t iter) const {
    (void)iter;
    return false;
}

void ProofSet::save(uint32_t iter, const Words& words) {
    (void)iter;
    (void)words;
}

Words ProofSet::fromUint64(const std::vector<uint64_t>& host, uint32_t exponent) {
    return Words::fromUint64(host, exponent);
}

} // namespace core
