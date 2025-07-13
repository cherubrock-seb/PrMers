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

namespace core {

class ProofManager {
public:
    ProofManager(uint32_t exponent, int proofLevel,
                 cl_command_queue queue, uint32_t n,
                 const std::vector<int>& digitWidth);
    void checkpoint(cl_mem buf, uint32_t iter);    
    std::filesystem::path proof() const;

private:
    ProofSet           proofSet_;
    cl_command_queue   queue_;
    uint32_t           n_;
    uint32_t           exponent_;
    std::vector<int>   digitWidth_;
};

}
