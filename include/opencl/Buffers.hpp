#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
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
    Buffers(const prmers::ocl::Context& ctx, const math::Precompute& pre);
    ~Buffers();
    
    cl_mem input;             // main data buffer
    cl_mem digitWeightBuf;    // digit weights
    cl_mem digitInvWeightBuf; // inverse digit weights
    cl_mem twiddle4Buf;        // forward twiddles
    cl_mem invTwiddle4Buf;     // inverse twiddles
    cl_mem twiddle5Buf;        // forward twiddles
    cl_mem invTwiddle5Buf;     // inverse twiddles
    cl_mem blockCarryBuf;
    cl_mem digitWidthMaskBuf;
    cl_mem Hbuf;
    cl_mem Hq;
    cl_mem Qbuf;
    cl_mem tmp; 
    cl_mem r2,save,bufd,buf3,last_correct_state,last_correct_bufd;
    std::vector<cl_mem> evenPow;
    // Number of 64-bit words in the packed digit-width mask for n digits. The carry kernels
    // prefetch maskPacked[base + 2] with base up to (n - 1) / 64, so at least two guard words
    // must follow the last word that holds mask bits.
    static size_t maskPackedWords(size_t n) {
        const size_t chunks = (n + 63) / 64;
        const size_t padded = (chunks + 3) & ~size_t(3);
        return padded < chunks + 2 ? chunks + 2 : padded;
    }
    static cl_mem createBuffer(const prmers::ocl::Context& ctx, cl_mem_flags flags,
                               size_t size, const void* ptr,
                               const std::string& name);
};

} // namespace opencl
