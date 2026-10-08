// core/ProofManagerMarin.hpp

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
#include <string>
#include "core/ProofSetMarin.hpp"
#include "core/ProofCheckpoint.hpp"
#include "marin/engine.h"

namespace core {

class ProofManagerMarin {
public:
    ProofManagerMarin(uint32_t exponent, int proofLevel,
                 cl_command_queue queue, uint32_t n,
                 const std::vector<int>& digitWidth,
                 const std::vector<std::string>& knownFactors = {},
                 const std::string& savePath = std::string());
    void checkpoint(cl_mem buf, uint32_t iter);    
    void checkpointMarin(engine::digit host, uint32_t iter);
    std::filesystem::path proof() const;
    bool shouldCheckpoint(uint32_t iter) const;
    // Power of the proof proof() writes: the one the checkpoints were saved for.
    uint32_t power() const { return proofSet_.power; }

    // Lower the power residues are saved for; see ProofSetMarin::setPower.
    void setPower(uint32_t newPower) { proofSet_.setPower(newPower); }

    // Where this run keeps its proof files: under the save path (-f).
    const ProofLocation& location() const { return proofSet_.location(); }
    // See ProofSetMarin::adoptLegacyResidues.
    bool adoptLegacyResidues(uint32_t resumeIter, std::string& note) {
        return proofSet_.adoptLegacyResidues(resumeIter, note);
    }
    void releaseLegacyResidues() { proofSet_.releaseLegacyResidues(); }

private:
    ProofSetMarin           proofSet_;
    cl_command_queue   queue_;
    uint32_t           n_;
    uint32_t           exponent_;
    std::vector<int>   digitWidth_;
};

}