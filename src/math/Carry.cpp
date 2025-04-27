// Carry.cpp
#include "math/Carry.hpp"
#include "opencl/Context.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace math {

Carry::Carry(const opencl::Context& ctx, cl_command_queue queue, cl_program program, size_t vectorSize, std::vector<int> digitWidth)
    : context_(ctx)
    , queue_(queue)
    , vectorSize_(vectorSize)
    , digitWidth_(digitWidth)
{
    cl_int err;
    carryKernel_ = clCreateKernel(program, "kernel_carry", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel_carry");
    }

    carryKernel2_ = clCreateKernel(program, "kernel_carry_2", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel_carry_2");
    }
}

void Carry::carryGPU(cl_mem buffer, cl_mem blockCarryBuffer, size_t bufferSize)
{
    cl_int err;
    size_t workersCarry = context_.getWorkersCarry();
    //std::cout << "Launching kernel CARRY workers=" << workersCarry << std::endl;
    //workersCarry = 2;
    // kernel_carry
    err  = clSetKernelArg(carryKernel_, 0, sizeof(cl_mem), &buffer);
    err |= clSetKernelArg(carryKernel_, 1, sizeof(cl_mem), &blockCarryBuffer);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set kernel_carry args");
    }

    //size_t globalWorkSize = bufferSize / sizeof(cl_ulong4);
    err = clEnqueueNDRangeKernel(queue_, carryKernel_, 1, nullptr, &workersCarry, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << "Failed to enqueue kernel_carry, error code: " << err;
        throw std::runtime_error(oss.str());
    }





   // clFinish(queue_);

    // kernel_carry_2
    err  = clSetKernelArg(carryKernel2_, 0, sizeof(cl_mem), &buffer);
    err |= clSetKernelArg(carryKernel2_, 1, sizeof(cl_mem), &blockCarryBuffer);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set kernel_carry_2 args");
    }

    err = clEnqueueNDRangeKernel(queue_, carryKernel2_, 1, nullptr, &workersCarry, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue kernel_carry_2");
    }

    //clFinish(queue_);
}


uint64_t Carry::digit_adc(const uint64_t lhs, const int digit_width, uint64_t & carry)
{
    const uint64_t s = lhs + carry;
    const uint64_t c =  s < lhs;
    carry = (s >> digit_width) + (c << (64 - digit_width));
    return s & ((uint64_t(1) << digit_width) - 1);
}

void Carry::handleFinalCarry(std::vector<uint64_t>& x, const std::vector<int>& digit_width_cpu)
{
    int n = x.size();
    x[0] += 1;
    uint64_t c = 0;
    
    for (cl_uint k = 0; k < n; ++k) {
        x[k] = digit_adc(x[k], digit_width_cpu[k], c);
    }

    while (c != 0) {
        for (cl_uint k = 0; k < n; ++k) {
            x[k] = digit_adc(x[k], digit_width_cpu[k], c);
            if (c == 0) break;
        }
    }

    x[0] -= 1;
}

} // namespace math
