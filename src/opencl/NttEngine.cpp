// src/opencl/NttEngine.cpp

#include "opencl/NttEngine.hpp"
#include "opencl/Buffers.hpp"
#include "opencl/Kernels.hpp"
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
// Error string helper
// -----------------------------------------------------------------------------
const char* getCLErrorString(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";
        default:                                    return "UNKNOWN ERROR";
    }
}
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
        std::cerr << "Error executing kernel '" << kernelName << "': " << getCLErrorString(err) << " (" << err << ")" << std::endl;
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
    cl_kernel kernel_ntt_mm = kernels_.getKernel("kernel_ntt_radix4_mm");
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
        clSetKernelArg(kernel_ntt_mm_first, 3, sizeof(cl_uint), &m);
        kernelsExecuted++;
        executeKernelAndDisplay(queue_, kernel_ntt_mm_first, buf_x, workers, localSize,
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
            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(cl_uint), &mm);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling,true);
            /*
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(cl_uint), &mm);
            clSetKernelArg(kernel_ntt_mm, 3, 0, NULL); 
            executeKernelAndDisplay(queue_, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling,true);
            mm = 1;
            m = 1;
            clSetKernelArg(kernel_radix2_square_radix2, 0, sizeof(cl_mem), &buf_x);
            executeKernelAndDisplay(queue_, kernel_radix2_square_radix2, buf_x, workers*2, localSize,
                    "kernel_radix2_square_radix2 (m=" + std::to_string(m) + ") workers=" + std::to_string(workers*2), n, profiling,true);
                    */
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 2, sizeof(cl_mem), &buf_wi);
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
            "kernel_ntt_mm_3_steps (m=" + std::to_string(m) + ")", n, profiling,true);
            
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling,true);
        }   
        else if(mm==8){
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 2, sizeof(cl_mem), &buf_wi);
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

            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(cl_uint), &mm);
            kernelsExecuted++;
            executeKernelAndDisplay(queue_, kernel_ntt_mm, buf_x, workers, localSize,
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
                executeKernelAndDisplay(queue_, kernel_inverse_ntt_mm, buf_x, workers, localSize,
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
                executeKernelAndDisplay(queue_, kernel_inverse_ntt_mm, buf_x, workers, localSize,
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

} // namespace opencl
