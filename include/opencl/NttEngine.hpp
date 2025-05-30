#pragma once

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
namespace opencl {

class NttEngine {
public:
    NttEngine(const Context& ctx,
              Kernels& kernels,
              Buffers& buffers,
              const math::Precompute& precompute);

    int forward(cl_mem buf_x, uint64_t iter);
    int inverse(cl_mem buf_x, uint64_t iter);
    int forward_simple(cl_mem buf_x, uint64_t iter);
    int inverse_simple(cl_mem buf_x, uint64_t iter);
    int pointwiseMul(cl_mem a, cl_mem b);

private:
    const Context& ctx_;
    //std::vector<Stage> forwardPipeline_;
    //std::vector<Stage> inversePipeline_;
    cl_command_queue      queue_;
    Kernels&              kernels_;
    Buffers&              buffers_;
    const math::Precompute& pre_;
};

} // namespace opencl
