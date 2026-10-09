#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdint>
#include <string>
#include "opencl/Context.hpp"
#include "math/Precompute.hpp"

namespace prmers::ocl {

class Program {
public:
    Program(const prmers::ocl::Context& context, cl_device_id device,
            const std::string& filePath, const math::Precompute& pre,
            const std::string& buildOptions = "", bool debug = false);

    ~Program();

    cl_program getProgram() const noexcept;

    // The -DMODULUS_P=<p> build option. p can be any 32-bit exponent, so it must stay unsigned: an int would turn
    // p >= 2^31 into a negative number. The UL suffix makes the literal an unsigned 64-bit constant for the kernel
    // compiler regardless of its value.
    static std::string modulusDefine(uint64_t p) { return "-DMODULUS_P=" + std::to_string(p) + "UL"; }

private:
    cl_program program_;
    const prmers::ocl::Context&    context_;

    std::string loadKernelSource(const std::string& filePath) const;
    void checkBuildError(cl_program program, cl_device_id device) const;
};

} // namespace opencl
