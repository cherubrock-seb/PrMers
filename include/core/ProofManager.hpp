// core/ProofManager.hpp

#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include "core/ProofSet.hpp"
#include "core/ProofCheckpoint.hpp"

// Forward declarations
namespace prmers::ocl {
    class NttEngine;
    class Context;
}

namespace math {
    class Carry;
}

namespace core {

// Thrown when a generated proof fails its own verification. The proof file is
// not kept: callers must not retry at a lower power or fall back to an
// unverified CPU proof, and must not report a proof for the test.
class ProofVerificationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ProofManager {
public:
    ProofManager(uint32_t exponent, int proofLevel,
                 cl_command_queue queue, uint32_t n,
                 const std::vector<int>& digitWidth,
                 const std::vector<std::string>& knownFactors = {},
                 const std::string& savePath = std::string());
    void checkpoint(cl_mem buf, uint32_t iter);  
    void checkpointMarin(std::vector<uint64_t> host, uint32_t iter);
    // Lower the power residues are saved for (the points of a lower power are
    // a subset of those of the original power).
    void setPower(uint32_t newPower) { proofSet_.power = newPower; }
    std::filesystem::path proof(const prmers::ocl::Context& ctx, opencl::NttEngine& ntt, math::Carry& carry, uint32_t proofPower, bool verify=true) const;

    // Where this run keeps its proof files: under the save path (-f).
    const ProofLocation& location() const { return proofSet_.location(); }
    // See ProofSetMarin::adoptLegacyResidues.
    bool adoptLegacyResidues(uint32_t resumeIter, std::string& note) {
        return proofSet_.adoptLegacyResidues(resumeIter, note);
    }
    void releaseLegacyResidues() { proofSet_.releaseLegacyResidues(); }

private:
    ProofSet           proofSet_;
    cl_command_queue   queue_;
    uint32_t           n_;
    uint32_t           exponent_;
    std::vector<int>   digitWidth_;
};

}
