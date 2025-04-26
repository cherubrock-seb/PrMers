// Carry.hpp
#pragma once

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
    Carry(const opencl::Context& ctx, cl_command_queue queue, cl_program program, size_t vectorSize, std::vector<int> digitWidth);
    void carryGPU(cl_mem buffer, cl_mem blockCarryBuffer, size_t bufferSize);
    void handleFinalCarry(std::vector<unsigned long>& x, const std::vector<int>& digitWidth);
    uint64_t digit_adc(const uint64_t lhs, const int digit_width, uint64_t & carry);

private:
    const opencl::Context&    context_;
    cl_command_queue  queue_;
    cl_kernel         carryKernel_;
    cl_kernel         carryKernel2_;
    size_t            vectorSize_;
    std::vector<int>  digitWidth_;

};

} // namespace math
