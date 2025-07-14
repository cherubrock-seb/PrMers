// opencl/Kernels.hpp
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#include <unordered_map>
#include <string>
#include <algorithm>


#pragma once

namespace opencl {

class Kernels {
public:
    Kernels(cl_program program, cl_command_queue queue);
    ~Kernels();

    void createKernel(const std::string& name);
    cl_kernel getKernel(const std::string& name) const;

    void runSquaring(cl_mem buf, size_t n);
    void runSub2(cl_mem buf);
    void runSub1(cl_mem buf);

private:
    cl_program            program_;
    cl_command_queue      queue_;
    std::unordered_map<std::string, cl_kernel> kernels_;
};

} // namespace opencl
