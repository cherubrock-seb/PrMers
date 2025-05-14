// Carry.hpp
#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>
#include "opencl/Context.hpp"
#include "opencl/EventSynchronizer.hpp" 

namespace math {

class Carry {
public:
    Carry(const opencl::Context& ctx, cl_command_queue queue, cl_program program, size_t vectorSize, std::vector<int> digitWidth, cl_mem digitWidthMaskBuf,  opencl::EventSynchronizer& sync);
    void carryGPU(cl_mem buffer, cl_mem blockCarryBuffer, size_t bufferSize);
    void carryGPU_mul_base(cl_mem buffer, cl_mem blockCarryBuffer, size_t bufferSize);
    void handleFinalCarry(std::vector<uint64_t>& x, const std::vector<int>& digitWidth);
    uint64_t digit_adc(const uint64_t lhs, const int digit_width, uint64_t & carry);

private:
    const opencl::Context&    context_;
    cl_command_queue  queue_;
    cl_kernel         carryKernel_;
    cl_kernel         carryKernel2_;
    cl_kernel         carryKernel3_;
    size_t            vectorSize_;
    std::vector<int>  digitWidth_;
    cl_mem                 digitWidthMaskBuf_;
    opencl::EventSynchronizer& sync_;
};

} // namespace math
