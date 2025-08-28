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
#include "core/ProofSet.hpp"

// Forward declarations
namespace opencl {
    class NttEngine;
    class Context;
}

namespace math {
    class Carry;
}

namespace core {

class ProofManager {
public:
    ProofManager(uint32_t exponent, int proofLevel,
                 cl_command_queue queue, uint32_t n,
                 const std::vector<int>& digitWidth,
                 const std::vector<std::string>& knownFactors = {});
    void checkpoint(cl_mem buf, uint32_t iter);  
    void checkpointMarin(std::vector<uint64_t> host, uint32_t iter);
    std::filesystem::path proof(const opencl::Context& ctx, opencl::NttEngine& ntt, math::Carry& carry, bool verify=true) const;

private:
    ProofSet           proofSet_;
    cl_command_queue   queue_;
    uint32_t           n_;
    uint32_t           exponent_;
    std::vector<int>   digitWidth_;
};

}
