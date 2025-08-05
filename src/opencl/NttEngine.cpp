#include "opencl/NttEngine.hpp"
#include "opencl/Buffers.hpp"
#include "opencl/Kernels.hpp"
#include "util/OpenCLError.hpp"
#include "math/Carry.hpp"
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
                     const math::Precompute& precompute, bool pm1)
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
        ls5_val_ = ctx_.getLocalSize5();
        //std::cout << "ls0_val_" << ls0_val_ << std::endl;
        const size_t* ls0 = nullptr;
        const size_t* ls2 = &ls2_val_;
        const size_t* ls3 = &ls3_val_;
        const size_t* ls5 = &ls5_val_;

        cl_mem buf_dw = buffers_.digitWeightBuf;
        cl_mem buf_w4 = buffers_.twiddle4Buf;
        cl_mem buf_w5 = buffers_.twiddle5Buf;
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
            kernels_.getKernel("kernel_ntt_radix5_mm_first"),
            buf_dw,
            buf_w4,
            buf_w5,
            ls0,
            ls2,
            ls3,
            ls5
        );
    }
    
    {
        ls0_vali_ = ctx_.getLocalSize();
        ls2_vali_ = ctx_.getLocalSize2();
        ls5_vali_ = ctx_.getLocalSize5();
        
        const size_t* ls0 = nullptr;
        const size_t* ls2 = &ls2_vali_;
        const size_t* ls5 = &ls5_vali_;

        cl_mem buf_diw = buffers_.digitInvWeightBuf;
        cl_mem buf_wi4  = buffers_.invTwiddle4Buf;
        cl_mem buf_wi5  = buffers_.invTwiddle5Buf;
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
            kernels_.getKernel("kernel_ntt_inverse_radix5_mm_last"),
            buf_wi4,
            buf_wi5,
            buf_diw,
            ls0,
            ls2,
            lastOutputInv,
            ls5
        );
    }

    //if(pm1){
        std::cout << "\nPipeline simple (PM1 stage2)\n";
        {
            ls0_val_ = ctx_.getLocalSize();
            ls2_val_ = ctx_.getLocalSize2();
            ls3_val_ = ctx_.getLocalSize3();
            ls5_val_ = ctx_.getLocalSize5();
            //std::cout << "ls0_val_" << ls0_val_ << std::endl;
            const size_t* ls0 = nullptr;
            const size_t* ls2 = &ls2_val_;
            const size_t* ls3 = &ls3_val_;
            const size_t* ls5 = &ls5_val_;

            cl_mem buf_dw = buffers_.digitWeightBuf;
            cl_mem buf_w4 = buffers_.twiddle4Buf;
            cl_mem buf_w5 = buffers_.twiddle5Buf;
            forward_simple_pipeline = buildForwardSimplePipeline(
                n,
                queue_,
                nullptr,                                    
                kernels_.getKernel("kernel_ntt_radix4_mm_first"),
                kernels_.getKernel("kernel_ntt_radix4_mm_2steps"),
                kernels_.getKernel("kernel_ntt_radix4_mm_m2"),
                kernels_.getKernel("kernel_ntt_radix4_mm_m4"),
                kernels_.getKernel("kernel_ntt_radix4_mm_m8"),
                kernels_.getKernel("kernel_ntt_radix4_mm_m16"),
                kernels_.getKernel("kernel_ntt_radix4_mm_m32"),
                kernels_.getKernel("kernel_ntt_radix4_last_m1_nosquare"),
                kernels_.getKernel("kernel_ntt_radix4_last_m1_n4_nosquare"),
                kernels_.getKernel("kernel_ntt_radix4_radix2_square_radix2_radix4"),
                kernels_.getKernel("kernel_ntt_radix2_square_radix2"),
                kernels_.getKernel("kernel_ntt_radix4_mm_2steps_first"),
                kernels_.getKernel("kernel_ntt_radix4_square_radix4"),
                kernels_.getKernel("kernel_ntt_radix5_mm_first"),
                kernels_.getKernel("kernel_ntt_radix2"),
                buf_dw,
                buf_w4,
                buf_w5,
                ls0,
                ls2,
                ls3,
                ls5
            );
        }

        {
            ls0_vali_ = ctx_.getLocalSize();
            ls2_vali_ = ctx_.getLocalSize2();
            ls5_vali_ = ctx_.getLocalSize5();
            const size_t* ls0 = nullptr;
            const size_t* ls2 = &ls2_vali_;
            const size_t* ls5 = &ls5_vali_;

            cl_mem buf_diw = buffers_.digitInvWeightBuf;
            cl_mem buf_wi4  = buffers_.invTwiddle4Buf;
            cl_mem buf_wi5  = buffers_.invTwiddle5Buf;
            int lastOutputInv = forward_simple_pipeline.back().outputInverse;
            inverse_simple_pipeline = buildInverseSimplePipeline(
                n,
                queue_,
                nullptr,                                     
                kernels_.getKernel("kernel_inverse_ntt_radix4_m1_n4"),
                kernels_.getKernel("kernel_inverse_ntt_radix4_m1"),
                kernels_.getKernel("kernel_ntt_radix4_inverse_mm_2steps"),
                kernels_.getKernel("kernel_inverse_ntt_radix4_mm"),
                kernels_.getKernel("kernel_inverse_ntt_radix4_mm_last"),
                kernels_.getKernel("kernel_ntt_radix4_inverse_mm_2steps_last"),
                kernels_.getKernel("kernel_ntt_inverse_radix5_mm_last"),
                kernels_.getKernel("kernel_ntt_radix2"),
                buf_wi4,
                buf_wi5,
                buf_diw,
                ls0,
                ls2,
                lastOutputInv,
                ls5
            );
        }
    //}
}

static void executeKernelAndDisplay(cl_command_queue queue,
                                    cl_kernel kernel,
                                    cl_mem buf_x,
                                    size_t workers,
                                    const size_t* localSize,
                                    const std::string& kernelName,
                                    bool profiling,
                                    bool debug,
                                    cl_uint n)
{

    debug = false;
    if (debug) {
        clFinish(queue);
        size_t numElems = n; 
        std::vector<uint64_t> host_x(numElems);

        cl_int err = clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0,
                                  numElems * sizeof(uint64_t),
                                  host_x.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Error reading buf_x: " << n  << util::getCLErrorString(err)
                      << " (" << err << ")\n";
            return;
        }

        std::cout << "=== Contenu de `buf_x` Avant kernel `" << kernelName
        << " workers = " << workers 
                  << "` ===\n";
            std::cout << "[";
            for (int j = 0; static_cast<size_t>(j) < numElems; ++j) {
                std::cout << host_x[j] << ",";
            }
            std::cout << "]\n";
        
        std::cout << "============================================\n";
    }

    //const size_t* actualLocalSize = (localSize && localSize[0] != 0) ? localSize : nullptr;

    cl_int err = clEnqueueNDRangeKernel(
        queue, kernel, 1, nullptr, &workers, localSize,
        0, nullptr, nullptr);
    //std::cerr << "Kernel " << kernelName << " Actual actualLocalSize=" << actualLocalSize << " nullptr = " << nullptr << std::endl;

    if (err != CL_SUCCESS) {
        std::cerr << "Kernel " << kernelName << util::getCLErrorString(err)
                  << " (" << err << ")\n";
    }
    if (debug) {
        clFinish(queue);
        size_t numElems = n;
        std::vector<uint64_t> host_x(numElems);

        err = clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0,
                                  numElems * sizeof(uint64_t),
                                  host_x.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Error reading buf_x: " << n  << util::getCLErrorString(err)
                      << " (" << err << ")\n";
            return;
        }

        std::cout << "=== Contenu de `buf_x` après kernel `" << kernelName
                  << " workers = " << workers << "` ===\n";
        std::cout << "[";
            for (int j = 0; static_cast<size_t>(j) < numElems; ++j) {
                std::cout << host_x[j] << ",";
            }
            std::cout << "]\n";
        std::cout << "============================================\n";
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
            true,
            n
        );
        ++executed;
    }
    return executed;
}


int NttEngine::forward_simple(cl_mem buf_x, uint64_t /*iter*/) {
    cl_uint n = pre_.getN();
    int executed = 0;
    for (auto& stage : forward_simple_pipeline) {
        setStageArgs2(stage, buf_x);
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
            true,
            n
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
            true,
            n
        );
        ++executed;
    }
    return executed;
}

int NttEngine::inverse_simple(cl_mem buf_x, uint64_t /*iter*/) {
    cl_uint n = pre_.getN();
    
    int executed = 0;
    for (auto& stage : inverse_simple_pipeline) {
        setStageArgs2(stage, buf_x);
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
            true,
            n
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
        false,
        n);
    return 1;
}

void NttEngine::squareInPlace(cl_mem A, math::Carry& carry, size_t limbBytes) {
    cl_int err;
    
    cl_mem tmpA = clCreateBuffer(ctx_.getContext(),
                                 CL_MEM_READ_WRITE,
                                 limbBytes,
                                 nullptr,
                                 &err);
    if (err != CL_SUCCESS) std::abort();
    
    clEnqueueCopyBuffer(queue_, A, tmpA,
                        0, 0, limbBytes,
                        0, nullptr, nullptr);
    
    forward_simple(tmpA, 0);
    pointwiseMul(tmpA, tmpA);
    inverse_simple(tmpA, 0);
    carry.carryGPU(tmpA, buffers_.blockCarryBuf, limbBytes);
    
    clEnqueueCopyBuffer(queue_, tmpA, A,
                        0, 0, limbBytes,
                        0, nullptr, nullptr);
    
    clReleaseMemObject(tmpA);
}

void NttEngine::copy(cl_mem src, cl_mem dst, size_t bytes) {
    clEnqueueCopyBuffer(queue_, src, dst, 0, 0, bytes, 0, nullptr, nullptr);
}

void NttEngine::mulInPlace(cl_mem A, cl_mem B, math::Carry& carry, size_t limbBytes) {
    copy(A, buffers_.input, limbBytes);
    cl_int err;
    forward_simple(buffers_.input, 0);
    cl_mem temp = clCreateBuffer(
        ctx_.getContext(),
        CL_MEM_READ_WRITE,
        limbBytes,
        nullptr,
        &err
    );
    copy(buffers_.input, temp, limbBytes);
   
    copy(B, buffers_.input, limbBytes);
    forward_simple(buffers_.input, 0);
    pointwiseMul(buffers_.input, temp);
    inverse_simple(buffers_.input, 0);
    carry.carryGPU(buffers_.input, buffers_.blockCarryBuf, limbBytes);
    copy(buffers_.input, A, limbBytes);
    clReleaseMemObject(temp);
}

void NttEngine::mulInPlace2(cl_mem A, cl_mem B, math::Carry& carry, size_t limbBytes) {
    forward_simple(buffers_.input, 0);
    pointwiseMul(buffers_.input, B);
    inverse_simple(buffers_.input, 0);
    carry.carryGPU(buffers_.input, buffers_.blockCarryBuf, limbBytes);
}

void NttEngine::mulInPlace3(cl_mem A, cl_mem B, math::Carry& carry, size_t limbBytes) {
    //copy(A, buffers_.input, limbBytes);
    cl_int err;
    forward_simple(A, 0);
    cl_mem temp = clCreateBuffer(
        ctx_.getContext(),
        CL_MEM_READ_WRITE,
        limbBytes,
        nullptr,
        &err
    );
    copy(A, temp, limbBytes);
   
    copy(B, buffers_.input, limbBytes);
    forward_simple(buffers_.input, 0);
    pointwiseMul(buffers_.input, temp);
    inverse_simple(buffers_.input, 0);
    carry.carryGPU(buffers_.input, buffers_.blockCarryBuf, limbBytes);
    copy(buffers_.input, A, limbBytes);
    clReleaseMemObject(temp);
}

void NttEngine::mulInPlace5(cl_mem A, cl_mem B, math::Carry& carry, size_t limbBytes) {
    cl_int err;

    cl_mem tmpA = clCreateBuffer(ctx_.getContext(),
                                 CL_MEM_READ_WRITE,
                                 limbBytes,
                                 nullptr,
                                 &err);
    if (err != CL_SUCCESS) std::abort();

    cl_mem tmpB = clCreateBuffer(ctx_.getContext(),
                                 CL_MEM_READ_WRITE,
                                 limbBytes,
                                 nullptr,
                                 &err);
    if (err != CL_SUCCESS) std::abort();

    clEnqueueCopyBuffer(queue_, A, tmpA,
                        0, 0, limbBytes,
                        0, nullptr, nullptr);
    clEnqueueCopyBuffer(queue_, B, tmpB,
                        0, 0, limbBytes,
                        0, nullptr, nullptr);

    forward_simple(tmpA, 0);
    forward_simple(tmpB, 0);
    pointwiseMul(tmpA, tmpB);
    inverse_simple(tmpA, 0);
    carry.carryGPU(tmpA, buffers_.blockCarryBuf, limbBytes);

    clEnqueueCopyBuffer(queue_, tmpA, A,
                        0, 0, limbBytes,
                        0, nullptr, nullptr);

    clReleaseMemObject(tmpA);
    clReleaseMemObject(tmpB);
}

// Modular exponentiation for Mersenne numbers: result = base^exp mod (2^E - 1)
void NttEngine::powInPlace(cl_mem result, cl_mem base, uint64_t exp, math::Carry& carry, size_t limbBytes) {
    if (exp == 0) {
        // Set result to 1 - initialize result buffer with 1
        size_t numWords = limbBytes / sizeof(uint64_t);
        std::vector<uint64_t> one_data(numWords, 0);
        one_data[0] = 1;
        clEnqueueWriteBuffer(queue_, result, CL_TRUE, 0, limbBytes, one_data.data(), 0, nullptr, nullptr);
        return;
    }
    
    if (exp == 1) {
        // Copy base to result (safe even if they're the same buffer)
        if (result != base) {
            copy(base, result, limbBytes);
        }
        return;
    }
    
    // For binary exponentiation, we need to preserve the base value
    // Create temporary buffers for safe computation
    cl_int err;
    
    cl_mem base_copy_buf = clCreateBuffer(ctx_.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create base copy buffer");
    }
    
    cl_mem accumulator_buf = clCreateBuffer(ctx_.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(base_copy_buf);
        throw std::runtime_error("Failed to create accumulator buffer");
    }
    
    // Copy base to temporary buffer to preserve it
    copy(base, base_copy_buf, limbBytes);
    
    // Initialize accumulator with 1
    size_t numWords = limbBytes / sizeof(uint64_t);
    std::vector<uint64_t> one_data(numWords, 0);
    one_data[0] = 1;
    clEnqueueWriteBuffer(queue_, accumulator_buf, CL_TRUE, 0, limbBytes, one_data.data(), 0, nullptr, nullptr);
    
    // Binary exponentiation: result = base^exp mod (2^E - 1)
    while (exp > 0) {
        if (exp & 1) {
            // accumulator = accumulator * base_copy mod (2^E - 1)
            mulInPlace5(accumulator_buf, base_copy_buf, carry, limbBytes);
        }
        
        exp >>= 1;
        if (exp > 0) {
            // base_copy = base_copy * base_copy mod (2^E - 1)
            squareInPlace(base_copy_buf, carry, limbBytes);
        }
    }
    
    // Copy final result to result buffer
    copy(accumulator_buf, result, limbBytes);
    
    // Clean up temporary buffers
    clReleaseMemObject(base_copy_buf);
    clReleaseMemObject(accumulator_buf);
}

void NttEngine::subOne(cl_mem buf) {
    kernels_.runSub1(buf);
}

} // namespace opencl
