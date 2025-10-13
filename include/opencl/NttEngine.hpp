#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "math/Precompute.hpp"
#include <cstdint>
#include "opencl/Context.hpp"
#include "opencl/Buffers.hpp"
#include "opencl/Kernels.hpp"
#include "opencl/NttPipeline.hpp"

namespace math { class Carry; }

namespace opencl {

class NttEngine {
public:
    NttEngine(const prmers::ocl::Context& ctx,
              Kernels& kernels,
              Buffers& buffers,
              const math::Precompute& precompute, /*bool pm1,*/ bool debug=false);

    int forward(cl_mem buf_x, uint64_t iter);
    int inverse(cl_mem buf_x, uint64_t iter);
    int forward_simple(cl_mem buf_x, uint64_t iter);
    int inverse_simple(cl_mem buf_x, uint64_t iter);
    int pointwiseMul(cl_mem a, cl_mem b);
    
    void mulInPlace(cl_mem A, cl_mem B, math::Carry& carry, size_t limbBytes);
    void mulInPlace2(/*cl_mem A,*/ cl_mem B, math::Carry& carry, size_t limbBytes);
    void mulInPlace3(cl_mem A, cl_mem B, math::Carry& carry, size_t limbBytes);
    void mulInPlace5(cl_mem A, cl_mem B, math::Carry& carry, size_t limbBytes);
    void squareInPlace(cl_mem A, math::Carry& carry, size_t limbBytes);
    void powInPlace(cl_mem result, cl_mem base, uint64_t exp, math::Carry& carry, size_t limbBytes);
    void copy(cl_mem src, cl_mem dst, size_t bytes);
    void subOne(cl_mem buf);

private:
    size_t ls0_val_, ls2_val_, ls3_val_, ls5_val_;
    size_t ls0_vali_, ls2_vali_, ls5_vali_;
    const prmers::ocl::Context& ctx_;
    std::vector<NttStage> forward_pipeline;
    std::vector<NttStage> inverse_pipeline;    
    std::vector<NttStage> forward_simple_pipeline;
    std::vector<NttStage> inverse_simple_pipeline;
    cl_command_queue      queue_;
    Kernels&              kernels_;
    Buffers&              buffers_;
    const math::Precompute& pre_;
};

} // namespace opencl
