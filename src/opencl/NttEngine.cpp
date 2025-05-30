#include "opencl/NttEngine.hpp"
#include "opencl/NttPipeline.hpp"
#include "opencl/Buffers.hpp"
#include "opencl/Kernels.hpp"
#include "util/OpenCLError.hpp"
#include <iostream>
#include <algorithm>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace opencl {

NttEngine::NttEngine(const Context& ctx,
                     Kernels& kernels,
                     Buffers& buffers,
                     const math::Precompute& precompute)
    : ctx_(ctx)
    , queue_(ctx_.getQueue())
    , kernels_(kernels)
    , buffers_(buffers)
    , pre_(precompute)
{
    
}

static void executeKernelAndDisplay(cl_command_queue queue,
                                    cl_kernel kernel,
                                    cl_mem buf_x,
                                    size_t workers,
                                    const size_t* localSize,
                                    const std::string& kernelName,
                                    bool profiling,
                                    bool debug)
{
    cl_int err = clEnqueueNDRangeKernel(
        queue, kernel, 1, nullptr, &workers, localSize,
        0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << util::getCLErrorString(err)
                  << " (" << err << ")\n";
    }
}

int NttEngine::forward(cl_mem buf_x, uint64_t /*iter*/) {
    cl_uint n = pre_.getN();
    size_t ls0_val = ctx_.getLocalSize();
    size_t ls2_val = ctx_.getLocalSize2();
    size_t ls3_val = ctx_.getLocalSize3();
    const size_t* ls0 = &ls0_val;
    const size_t* ls2 = &ls2_val;
    const size_t* ls3 = &ls3_val;

    cl_mem buf_w  = buffers_.twiddleBuf;
    cl_mem buf_dw = buffers_.digitWeightBuf;

    auto pipeline = buildForwardPipeline(
        n,
        queue_,
        buf_x,                                    
        kernels_.getKernel("kernel_ntt_radix4_mm_first"),
        kernels_.getKernel("kernel_ntt_radix4_mm_2steps"),
        kernels_.getKernel("kernel_ntt_radix4_mm_3steps"),
        kernels_.getKernel("kernel_ntt_radix4_mm_m4"),
        kernels_.getKernel("kernel_ntt_radix4_mm_m8"),
        kernels_.getKernel("kernel_ntt_radix4_last_m1"),
        kernels_.getKernel("kernel_ntt_radix4_last_m1_n4"),
        kernels_.getKernel("kernel_ntt_radix4_radix2_square_radix2_radix4"),
        kernels_.getKernel("kernel_ntt_radix2_square_radix2"),
        buf_w,
        buf_dw,
        ls0,
        ls2,
        ls3
    );

    int executed = 0;
    for (auto& stage : pipeline) {
        setStageArgs(stage);
        size_t workers;
        if (stage.globalDiv == 0) {
            workers = (n/4) * 2;
        } else {
            workers = (n/4) / stage.globalDiv;
        }
        executeKernelAndDisplay(
            queue_,
            stage.kernel,
            buf_x,
            workers,
            stage.localSize,
            stage.name,
            false,
            true
        );
        ++executed;
    }
    return executed;
}
int NttEngine::inverse(cl_mem buf_x, uint64_t /*iter*/) {
    cl_uint n = pre_.getN();
    size_t ls0_val = ctx_.getLocalSize();
    size_t ls2_val = ctx_.getLocalSize2();
    const size_t* ls0 = &ls0_val;
    const size_t* ls2 = &ls2_val;

    cl_mem buf_wi  = buffers_.invTwiddleBuf;
    cl_mem buf_diw = buffers_.digitInvWeightBuf;

    auto pipeline = buildInversePipeline(
        n,
        queue_,
        buf_x,                                     
        kernels_.getKernel("kernel_inverse_ntt_radix4_m1_n4"),
        kernels_.getKernel("kernel_inverse_ntt_radix4_m1"),
        kernels_.getKernel("kernel_ntt_radix4_inverse_mm_2steps"),
        kernels_.getKernel("kernel_inverse_ntt_radix4_mm"),
        kernels_.getKernel("kernel_inverse_ntt_radix4_mm_last"),
        buf_wi,
        buf_diw,
        ls0,
        ls2
    );

    int executed = 0;
    for (auto& stage : pipeline) {
        setStageArgs(stage);
        size_t workers = (n/4) / stage.globalDiv;
        executeKernelAndDisplay(
            queue_,
            stage.kernel,
            buf_x,
            workers,
            stage.localSize,
            stage.name,
            false,
            true
        );
        ++executed;
    }
    return executed;
}


int NttEngine::pointwiseMul(cl_mem a, cl_mem b)
{
    cl_kernel k = kernels_.getKernel("kernel_pointwise_mul");
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &b);
    size_t n = pre_.getN();
    size_t ls0_val = ctx_.getLocalSize();
    const size_t* ls0 = &ls0_val;
    executeKernelAndDisplay(
        queue_,
        k,
        a,
        n,
        ls0,
        "kernel_pointwise_mul",
        false,
        false);
    return 1;
}

} // namespace opencl
