// src/opencl/NttEngine.cpp
/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
#include "opencl/NttEngine.hpp"
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
{}

// -----------------------------------------------------------------------------
// Helper function to execute a kernel and (optionally) display its execution time.
// -----------------------------------------------------------------------------
void executeKernelAndDisplay(cl_command_queue queue, cl_kernel kernel,
                             cl_mem buf_x, size_t workers, size_t localSize,
                             const std::string& kernelName, cl_uint nmax,
                             bool profiling,bool debug) {
    cl_event event;
    //std::cout << "Launching kernel: " << kernelName << ", workers=" << workers << ", localSize=" << localSize << std::endl;
    /*if (debug) {
        std::cout << "BEFORE  " << kernelName << std::endl;
        std::vector<uint64_t> locx(8, 0ULL);
        clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, 8 * sizeof(uint64_t), locx.data(), 0, nullptr, nullptr);
        std::cout << "BEFORE " << kernelName << " : x = [ ";
        for (cl_uint i = 0; i < 8; i++) std::cout << locx[i] << " ";
        std::cout << "...]" << std::endl;
    }*/
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &workers, &localSize, 0, nullptr, profiling ? &event : nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error executing kernel '" << kernelName << "': " << util::getCLErrorString(err) << " (" << err << ")" << std::endl;
    }
    
    if (profiling && event) {
        cl_ulong start_time, end_time;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
        double duration = (end_time - start_time) / 1e3;
        std::cout << "[Profile] " << kernelName << " duration: " << duration << " µs" << std::endl;
        clReleaseEvent(event);
    }
    /*
    if (debug) {
        std::cout << "After " << kernelName << std::endl;
        std::vector<uint64_t> locx(8, 0ULL);
        clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, 8 * sizeof(uint64_t), locx.data(), 0, nullptr, nullptr);
        std::cout << "After " << kernelName << " : x = [ ";
        for (cl_uint i = 0; i < 8; i++) std::cout << locx[i] << " ";
        std::cout << "...]" << std::endl;
    }*/
   
}

int NttEngine::forward(cl_mem buf_x, uint64_t /*iter*/) {
    cl_uint n          = pre_.getN();    
    cl_kernel kernel_ntt_mm_3_steps = kernels_.getKernel("kernel_ntt_radix4_mm_3steps");
    cl_kernel kernel_ntt_mm_2_steps = kernels_.getKernel("kernel_ntt_radix4_mm_2steps");
    cl_kernel kernel_radix2_square_radix2 = kernels_.getKernel("kernel_ntt_radix2_square_radix2");
    cl_kernel kernel_ntt_mm_m4 = kernels_.getKernel("kernel_ntt_radix4_mm_m4");
    cl_kernel kernel_ntt_mm_m8 = kernels_.getKernel("kernel_ntt_radix4_mm_m8");
    cl_kernel kernel_ntt_mm_first = kernels_.getKernel("kernel_ntt_radix4_mm_first");
    cl_kernel kernel_ntt_last_m1 = kernels_.getKernel("kernel_ntt_radix4_last_m1");
    cl_kernel kernel_ntt_last_m1_n4 = kernels_.getKernel("kernel_ntt_radix4_last_m1_n4");
    cl_kernel kernel_radix4_radix2_square_radix2_radix4 = kernels_.getKernel("kernel_ntt_radix4_radix2_square_radix2_radix4");
    cl_mem buf_w = buffers_.twiddleBuf;
    cl_mem buf_wi = buffers_.invTwiddleBuf;
    cl_mem buf_digit_weight = buffers_.digitWeightBuf;
    size_t workers = n/4;
    size_t localSize = ctx_.getLocalSize();
    size_t localSize2 = ctx_.getLocalSize2();;
    size_t localSize3 = ctx_.getLocalSize3();;
    bool profiling = false;
    cl_uint maxLocalMem=ctx_.getLocalMemSize();
    bool _even_exponent=ctx_.isEvenExponent();
    int kernelsExecuted = 0;
    

    if(n==4){
        cl_uint m = 1;
        clSetKernelArg(kernel_ntt_last_m1_n4, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_last_m1_n4, 1, sizeof(cl_mem), &buf_w);
        clSetKernelArg(kernel_ntt_last_m1_n4, 2, sizeof(cl_mem), &buf_digit_weight);
        kernelsExecuted++;
        executeKernelAndDisplay(queue_, kernel_ntt_last_m1_n4, buf_x, workers, localSize,
            "kernel_ntt_last_m1_n4 (m=" + std::to_string(m) + ")", n, profiling,true);
    }
    else{
        cl_uint m = n / 4;
        
        clSetKernelArg(kernel_ntt_mm_first, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_mm_first, 1, sizeof(cl_mem), &buf_w);
        clSetKernelArg(kernel_ntt_mm_first, 2, sizeof(cl_mem), &buf_digit_weight);
        kernelsExecuted++;
        executeKernelAndDisplay(queue_, kernel_ntt_mm_first, buf_x, workers/2, localSize,
            "kernel_ntt_radix4_mm_first (m=" + std::to_string(m) + ") workers=" + std::to_string(workers), n, profiling,true);
        
        clSetKernelArg(kernel_ntt_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_mm_2_steps, 1, sizeof(cl_mem), &buf_w);
        cl_uint mm = n/4;

        for (cl_uint m = n / 16; m >= 32; m /= 16) {
            
            clSetKernelArg(kernel_ntt_mm_2_steps, 2, sizeof(cl_uint), &m);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_mm_2_steps, buf_x, workers/4, localSize2,
                "kernel_ntt_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling,true);
            mm = m/4;
        }
       //if(mm==256){
        if(mm==256){
           
            mm = 64;
            m = 64;
            clSetKernelArg(kernel_ntt_mm_3_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm_3_steps, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm_3_steps, 2, sizeof(cl_uint), &mm);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_mm_3_steps, buf_x, workers, localSize,
            "kernel_ntt_mm_3_steps (m=" + std::to_string(m) + ")", n, profiling,true);
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling,true);
        } 
         if(mm==32){
            mm = 8;
            m = 8;
            clSetKernelArg(kernel_ntt_mm_m8, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm_m8, 1, sizeof(cl_mem), &buf_w);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_mm_m8, buf_x, workers/2, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling,true);
            
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 0, sizeof(cl_mem), &buf_x);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_radix4_radix2_square_radix2_radix4, buf_x, (workers*4)/8, localSize3,
                "kernel_radix4_radix2_square_radix2_radix4 (m=" + std::to_string(m) + ")", n, profiling,true);
        } 
        else if(mm==64){
            mm = 16;
            m = 16;
            clSetKernelArg(kernel_ntt_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm_2_steps, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm_2_steps, 2, sizeof(cl_uint), &m);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_mm_2_steps, buf_x, workers/4, localSize2,
            "kernel_ntt_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling,true);
            
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_last_m1 (m=" + std::to_string(m) + ")", n, profiling,true);
        }   
        else if(mm==8){
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 0, sizeof(cl_mem), &buf_x);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_radix4_radix2_square_radix2_radix4, buf_x, (workers*4)/8, localSize,
                "kernel_radix4_radix2_square_radix2_radix4 (m=" + std::to_string(m) + ")", n, profiling,true);
       }
       else if(mm==4){
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling,true);
       }   
       else if(mm==16){
            m = 4;
            mm=4;

            clSetKernelArg(kernel_ntt_mm_m4, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm_m4, 1, sizeof(cl_mem), &buf_w);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_mm_m4, buf_x, workers/2, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling,true);


            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling,true);

       }
       else if(mm==2){
            clSetKernelArg(kernel_radix2_square_radix2, 0, sizeof(cl_mem), &buf_x);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_radix2_square_radix2, buf_x, workers*2, localSize,
                    "kernel_radix2_square_radix2 (m=" + std::to_string(m) + ") workers=" + std::to_string(workers*2), n, profiling,true);
       }   
    }
    return kernelsExecuted;
}


int NttEngine::inverse(cl_mem buf_x, uint64_t /*iter*/) {
    cl_uint n          = pre_.getN();    
    cl_kernel kernel_ntt_inverse_mm_2_steps  = kernels_.getKernel("kernel_ntt_radix4_inverse_mm_2steps");
    cl_kernel kernel_inverse_ntt_mm          = kernels_.getKernel("kernel_inverse_ntt_radix4_mm");
    cl_kernel kernel_inverse_ntt_mm_last     = kernels_.getKernel("kernel_inverse_ntt_radix4_mm_last");
    cl_kernel kernel_inverse_ntt_m1          = kernels_.getKernel("kernel_inverse_ntt_radix4_m1");
    cl_kernel kernel_inverse_ntt_m1_n4       = kernels_.getKernel("kernel_inverse_ntt_radix4_m1_n4");
    cl_mem buf_wi = buffers_.invTwiddleBuf;
    cl_mem buf_digit_invweight = buffers_.digitInvWeightBuf;
    size_t workers = n/4;
    size_t localSize = ctx_.getLocalSize();
    size_t localSize2 = ctx_.getLocalSize2();;
    size_t localSize3 = ctx_.getLocalSize3();;
    bool profiling = false;
    cl_uint maxLocalMem=ctx_.getLocalMemSize();
    bool _even_exponent=ctx_.isEvenExponent();
    cl_uint m = 0;
    int kernelsExecuted = 0;
    if(n==4){
        m = 1;
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 1, sizeof(cl_mem), &buf_wi);
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 2, sizeof(cl_mem), &buf_digit_invweight);
        kernelsExecuted++;
        executeKernelAndDisplay(queue_, kernel_inverse_ntt_m1_n4, buf_x, workers, localSize,
            "kernel_inverse_ntt_m1_n4 (m=" + std::to_string(m) + ")", n, profiling,true);
    }
    else{
       
        if(_even_exponent){

            
            m = 1;
            clSetKernelArg(kernel_inverse_ntt_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_inverse_ntt_m1, 1, sizeof(cl_mem), &buf_wi);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_inverse_ntt_m1, buf_x, workers, localSize,
                "kernel_inverse_ntt_radix4_m1 (m=" + std::to_string(m) + ")", n, profiling,true);


            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 1, sizeof(cl_mem), &buf_wi);
            cl_uint mm = 4;
          
            for (cl_uint m = 4; m < n/16; m *= 16) {
                
                clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 2, sizeof(cl_uint), &m);
                kernelsExecuted++;
                executeKernelAndDisplay(queue_, kernel_ntt_inverse_mm_2_steps, buf_x, workers/4, localSize2,
                    "kernel_ntt_inverse_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling,true);
                mm = m*16;
            }
            if(mm<=n/16 && n>8){
                clSetKernelArg(kernel_inverse_ntt_mm, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_inverse_ntt_mm, 1, sizeof(cl_mem), &buf_wi);
                clSetKernelArg(kernel_inverse_ntt_mm, 2, sizeof(cl_uint), &mm);
                kernelsExecuted++;
                executeKernelAndDisplay(queue_, kernel_inverse_ntt_mm, buf_x, workers/2, localSize,
                        "kernel_inverse_ntt_radix4_mm (m=" + std::to_string(m) + ")", n, profiling,true);
            }

        }
        else{
            m = 8;
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 1, sizeof(cl_mem), &buf_wi);
            cl_uint mm = 8;
            for (cl_uint m = 8; m < n/16; m *= 16) {
                
                clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 2, sizeof(cl_uint), &m);
                kernelsExecuted++;
                executeKernelAndDisplay(queue_, kernel_ntt_inverse_mm_2_steps, buf_x, workers/4, localSize2,
                    "kernel_ntt_inverse_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling,true);
                mm = m*16;
            }
            if(mm<=n/16 && n>8){
                clSetKernelArg(kernel_inverse_ntt_mm, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_inverse_ntt_mm, 1, sizeof(cl_mem), &buf_wi);
                clSetKernelArg(kernel_inverse_ntt_mm, 2, sizeof(cl_uint), &mm);
                kernelsExecuted++;
                executeKernelAndDisplay(queue_, kernel_inverse_ntt_mm, buf_x, workers/2, localSize,
                        "kernel_inverse_ntt_radix4_mm (m=" + std::to_string(m) + ")", n, profiling,true);
            }
        }
        


        m = n/4;
        clSetKernelArg(kernel_inverse_ntt_mm_last, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 1, sizeof(cl_mem), &buf_wi);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 2, sizeof(cl_mem), &buf_digit_invweight);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 3, sizeof(cl_uint), &m);
        kernelsExecuted++;
        executeKernelAndDisplay(queue_, kernel_inverse_ntt_mm_last, buf_x, workers, localSize,
            "kernel_inverse_ntt_radix4_mm_last (m=" + std::to_string(m) + ")", n, profiling,true);
    }
    return kernelsExecuted;


}



int NttEngine::pointwiseMul(cl_mem a, cl_mem b) {
  cl_kernel k = kernels_.getKernel("kernel_pointwise_mul");
  clSetKernelArg(k, 0, sizeof(cl_mem), &a);
  clSetKernelArg(k, 1, sizeof(cl_mem), &b);
  size_t n = pre_.getN();
  size_t wg = ctx_.getLocalSize();
  executeKernelAndDisplay(queue_, k, a, n, wg, "kernel_pointwise_mul", n, false, false);
  return 1;
}

} // namespace opencl
