// core/ProofManager.hpp

#pragma once
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#include <cstdint>
#include "core/ProofSet.hpp"

namespace core {

class ProofManager {
public:
    ProofManager(uint32_t exponent, int proofLevel,
                 cl_command_queue queue, uint32_t n);
    void checkpoint(cl_mem buf, uint32_t iter);

private:
    ProofSet           proofSet_;
    cl_command_queue   queue_;
    uint32_t           n_;
    uint32_t           exponent_;
};

}
