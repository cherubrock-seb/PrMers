/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
#include "opencl/Buffers.hpp"
#include "opencl/Context.hpp"
#include <iostream>
#include <stdexcept>

namespace opencl {

Buffers::Buffers(const opencl::Context& ctx, const math::Precompute& pre)
  : input(nullptr)
{
    const size_t n = pre.getN();
    const size_t twiddle4Size = (n % 5 == 0) ? 3 * n / 5 : 3 * n;
    const size_t twiddle5Size = 4 * n / 5;

    digitWeightBuf = createBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        n * sizeof(uint64_t),
        pre.digitWeight().data(), "digitWeight");

    digitInvWeightBuf = createBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        n * sizeof(uint64_t),
        pre.digitInvWeight().data(), "digitInvWeight");

    twiddle4Buf = createBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        twiddle4Size * sizeof(uint64_t),
        pre.twiddlesRadix4().data(), "twiddles4");

    invTwiddle4Buf = createBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        twiddle4Size * sizeof(uint64_t),
        pre.invTwiddlesRadix4().data(), "invTwiddles4");
    if(n%5==0){
        twiddle5Buf = createBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            twiddle5Size * sizeof(uint64_t),
            pre.twiddlesRadix5().data(), "twiddles5");

        invTwiddle5Buf = createBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            twiddle5Size * sizeof(uint64_t),
            pre.invTwiddlesRadix5().data(), "invTwiddles5");
    }
    /*else{
        clReleaseMemObject(invTwiddle5Buf);
        clReleaseMemObject(twiddle5Buf);

    }*/

    blockCarryBuf = createBuffer(ctx, CL_MEM_READ_WRITE,
        ctx.getWorkersCarry() * sizeof(uint64_t),
        nullptr, "blockCarry");

    const auto& maskBits = pre.getDigitWidthMask();
    size_t maskN = maskBits.size();
    size_t chunks = (maskN + 63) / 64;

    std::vector<uint64_t> maskPacked(chunks, 0ULL);
    for (size_t i = 0; i < maskN; ++i) {
        if (maskBits[i]) {
            size_t blk = i >> 6;      // i / 64
            size_t bit = i & 63;      // i % 64
            maskPacked[blk] |= (1ULL << bit);
        }
    }

    digitWidthMaskBuf = createBuffer(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        maskPacked.size() * sizeof(uint64_t),
        maskPacked.data(),
        "digitWidthMaskPacked");
}


Buffers::~Buffers() {
    if (input)              clReleaseMemObject(input);
    if (digitWeightBuf)     clReleaseMemObject(digitWeightBuf);
    if (digitInvWeightBuf)  clReleaseMemObject(digitInvWeightBuf);
    if (twiddle4Buf)         clReleaseMemObject(twiddle4Buf);
    if (invTwiddle4Buf)      clReleaseMemObject(invTwiddle4Buf);
    if (twiddle5Buf)         clReleaseMemObject(twiddle5Buf);
    if (invTwiddle5Buf)      clReleaseMemObject(invTwiddle5Buf);
    if (blockCarryBuf)      clReleaseMemObject(blockCarryBuf);
}

cl_mem Buffers::createBuffer(const opencl::Context&  ctx, cl_mem_flags flags,
                             size_t size, const void* ptr,
                             const std::string& name)
{
    cl_int err;
    cl_mem buf = clCreateBuffer(ctx.getContext(), flags, size, const_cast<void*>(ptr), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffer " << name << ": " << err << std::endl;
        throw std::runtime_error("createBuffer " + name);
    }
    return buf;
}

} // namespace opencl
