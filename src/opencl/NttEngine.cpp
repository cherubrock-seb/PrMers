#include "opencl/NttEngine.hpp"
#include "opencl/Buffers.hpp"
#include "opencl/Kernels.hpp"
#include "util/OpenCLError.hpp"
#include <iostream>
#include <algorithm>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
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
    cl_uint n = pre_.getN();
    {
        ls0_val_ = ctx_.getLocalSize();
        ls2_val_ = ctx_.getLocalSize2();
        ls3_val_ = ctx_.getLocalSize3();
        std::cout << "ls0_val_" << ls0_val_ << std::endl;
        const size_t* ls0 = nullptr;
        const size_t* ls2 = &ls2_val_;
        const size_t* ls3 = &ls3_val_;

        cl_mem buf_w  = buffers_.twiddleBuf;
        cl_mem buf_dw = buffers_.digitWeightBuf;

        forward_pipeline = buildForwardPipeline(
            n,
            queue_,
            nullptr,                                    
            kernels_.getKernel("kernel_ntt_radix4_mm_first"),
            kernels_.getKernel("kernel_ntt_radix4_mm_2steps"),
            kernels_.getKernel("kernel_ntt_radix4_mm_m4"),
            kernels_.getKernel("kernel_ntt_radix4_mm_m8"),
            kernels_.getKernel("kernel_ntt_radix4_mm_m16"),
            kernels_.getKernel("kernel_ntt_radix4_mm_m32"),
            kernels_.getKernel("kernel_ntt_radix4_last_m1"),
            kernels_.getKernel("kernel_ntt_radix4_last_m1_n4"),
            kernels_.getKernel("kernel_ntt_radix4_radix2_square_radix2_radix4"),
            kernels_.getKernel("kernel_ntt_radix2_square_radix2"),
            kernels_.getKernel("kernel_ntt_radix4_mm_2steps_first"),
            kernels_.getKernel("kernel_ntt_radix4_square_radix4"),
            buf_w,
            buf_dw,
            ls0,
            ls2,
            ls3
        );
    }

    {
        ls0_vali_ = ctx_.getLocalSize();
        ls2_vali_ = ctx_.getLocalSize2();
        const size_t* ls0 = &ls0_vali_;
        const size_t* ls2 = &ls2_vali_;

        cl_mem buf_wi  = buffers_.invTwiddleBuf;
        cl_mem buf_diw = buffers_.digitInvWeightBuf;
        int lastOutputInv = forward_pipeline.back().outputInverse;
        inverse_pipeline = buildInversePipeline(
            n,
            queue_,
            nullptr,                                     
            kernels_.getKernel("kernel_inverse_ntt_radix4_m1_n4"),
            kernels_.getKernel("kernel_inverse_ntt_radix4_m1"),
            kernels_.getKernel("kernel_ntt_radix4_inverse_mm_2steps"),
            kernels_.getKernel("kernel_inverse_ntt_radix4_mm"),
            kernels_.getKernel("kernel_inverse_ntt_radix4_mm_last"),
            kernels_.getKernel("kernel_ntt_radix4_inverse_mm_2steps_last"),
            buf_wi,
            buf_diw,
            ls0,
            ls2,
            lastOutputInv
        );
    }

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
//    std::cerr << "Kernel " << kernelName << std::endl;

    if (err != CL_SUCCESS) {
        std::cerr << "Kernel " << kernelName << util::getCLErrorString(err)
                  << " (" << err << ")\n";
    }
}

int NttEngine::forward(cl_mem buf_x, uint64_t /*iter*/) {
    cl_uint n = pre_.getN();
    int executed = 0;
    for (auto& stage : forward_pipeline) {
        setStageArgs(stage, buf_x);
        int scale = stage.globalScale;
        size_t base = n;
        size_t workers;
        
        workers = base / static_cast<size_t>(scale);
        
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
    
    int executed = 0;
    for (auto& stage : inverse_pipeline) {
        setStageArgs(stage, buf_x);
        int scale = stage.globalScale;
        size_t base = n;
        size_t workers;
        
        workers = base / static_cast<size_t>(scale);
        
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
