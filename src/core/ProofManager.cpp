// core/ProofManager.cpp

#include "core/ProofManager.hpp"
#include <vector>

namespace core {

ProofManager::ProofManager(uint32_t exponent, int proofLevel,
                           cl_command_queue queue, uint32_t n)
  : proofSet_(exponent, proofLevel)
  , queue_(queue)
  , n_(n)
  , exponent_(exponent)
{}

void ProofManager::checkpoint(cl_mem buf, uint32_t iter) {
    if (! proofSet_.shouldCheckpoint(iter)) return;

    // read back the buffer from GPU
    std::vector<uint64_t> host(n_);
    clEnqueueReadBuffer(queue_, buf, CL_TRUE, 0,
                        n_ * sizeof(uint64_t),
                        host.data(), 0, nullptr, nullptr);

    // turn it into your proof word format
    Words partial = ProofSet::fromUint64(host, exponent_);
    proofSet_.save(iter, partial);
}

} // namespace core
