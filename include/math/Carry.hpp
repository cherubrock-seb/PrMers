// Carry.hpp
#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>
#include "opencl/Context.hpp"

namespace math {

class Carry {
public:
    Carry(const opencl::Context& ctx, cl_command_queue queue, cl_program program, size_t vectorSize, std::vector<int> digitWidth, cl_mem digitWidthMaskBuf);
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
    std::vector<int>  digitWidth_;
    cl_mem                 digitWidthMaskBuf_;
};

} // namespace math
