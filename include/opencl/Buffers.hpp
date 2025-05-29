#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include "math/Precompute.hpp"
#include "opencl/Context.hpp"
namespace opencl {

class Buffers {
public:
    Buffers(const opencl::Context& ctx, const math::Precompute& pre);
    ~Buffers();
    
    cl_mem input;             // main data buffer
    cl_mem digitWeightBuf;    // digit weights
    cl_mem digitInvWeightBuf; // inverse digit weights
    cl_mem twiddleBuf;        // forward twiddles
    cl_mem invTwiddleBuf;     // inverse twiddles
    cl_mem wiBuf;             // inverse-NTT root powers
    cl_mem blockCarryBuf;
    cl_mem digitWidthMaskBuf;
    
    static cl_mem createBuffer(const opencl::Context& ctx, cl_mem_flags flags,
                               size_t size, const void* ptr,
                               const std::string& name);
};

} // namespace opencl
