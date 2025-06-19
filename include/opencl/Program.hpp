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
#include "opencl/Context.hpp"
#include "math/Precompute.hpp"

namespace opencl {

class Program {
public:
    Program(const opencl::Context& context, cl_device_id device,
            const std::string& filePath, const math::Precompute& pre,
            const std::string& buildOptions = "");

    ~Program();

    cl_program getProgram() const noexcept;

private:
    cl_program program_;
    const opencl::Context&    context_;

    std::string loadKernelSource(const std::string& filePath) const;
    void checkBuildError(cl_program program, cl_device_id device) const;
};

} // namespace opencl
