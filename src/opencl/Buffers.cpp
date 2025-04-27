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
  , digitWeightBuf(createBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        pre.getN()*sizeof(uint64_t),
        pre.digitWeight().data(), "digitWeight"))
  , digitInvWeightBuf(createBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        pre.getN()*sizeof(uint64_t),
        pre.digitInvWeight().data(), "digitInvWeight"))
  , twiddleBuf(createBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        pre.getN()*3*sizeof(uint64_t),
        pre.twiddles().data(), "twiddles"))
  , invTwiddleBuf(createBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        pre.getN()*3*sizeof(uint64_t),
        pre.invTwiddles().data(), "invTwiddles"))
  , wiBuf(createBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        pre.getN()*3*sizeof(uint64_t),
        pre.invTwiddles().data(), "wi"))
  ,blockCarryBuf(createBuffer(ctx, CL_MEM_READ_WRITE,
        ctx.getWorkersCarry()*sizeof(uint64_t),
        nullptr, "blockCarry"))
{}

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
