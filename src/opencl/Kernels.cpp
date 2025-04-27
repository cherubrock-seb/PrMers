#include "opencl/Kernels.hpp"
#include <stdexcept>

namespace opencl {

Kernels::Kernels(cl_program program, cl_command_queue queue)
    : program_(program), queue_(queue) {}

Kernels::~Kernels() {
    for (auto& kv : kernels_) {
        clReleaseKernel(kv.second);
    }
}

void Kernels::createKernel(const std::string& name) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program_, name.c_str(), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel: " + name);
    }
    kernels_[name] = kernel;
}

cl_kernel Kernels::getKernel(const std::string& name) const {
    auto it = kernels_.find(name);
    if (it == kernels_.end()) {
        throw std::runtime_error("Kernel not found: " + name);
    }
    return it->second;
}

void Kernels::runSquaring(cl_mem buf, size_t n) {
    cl_kernel k = getKernel("kernel_ntt_radix2_square_radix2");
    clSetKernelArg(k, 0, sizeof(buf), &buf);
    size_t global = n;
    size_t local  = std::min<size_t>(n, 256);
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    //clFinish(queue_);
}

void Kernels::runSub2(cl_mem buf) {
    cl_kernel k = getKernel("kernel_sub2");
    clSetKernelArg(k, 0, sizeof(buf), &buf);
    size_t global = 1, local = 1;
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    //clFinish(queue_);
}

} // namespace opencl
