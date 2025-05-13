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
#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#endif

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "opencl/Program.hpp"
#include "opencl/Context.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <algorithm>

namespace opencl {

Program::Program(const opencl::Context& context, cl_device_id device,
                 const std::string& filePath,const math::Precompute& pre,
                 const std::string& buildOptions
                 )
    : context_(context), 
    program_(nullptr)
{
    std::string source = loadKernelSource(filePath);
    const char* src = source.c_str();
    size_t length = source.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context.getContext(), 1, &src, &length, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL program from source.");
    }

    // fetch all of our tuning parameters from Context
    cl_uint n          = context.getTransformSize();
    cl_uint wg         = context.getWorkGroupCount();
    int     lpd        = context.getLocalCarryPropagationDepth();
    cl_uint wCarry     = context.getWorkersCarry();
    cl_uint wNtt       = n / 4;
    cl_uint w2step     = n / 16;
    cl_uint ls         = static_cast<cl_uint>(context.getLocalSize());
    cl_uint ls2        = static_cast<cl_uint>(context.getLocalSize2());
    cl_uint ls3        = static_cast<cl_uint>(context.getLocalSize3());

    cl_uint div4       = lpd / 4;
    cl_uint div4_min = std::max<cl_uint>(1, (lpd - 4) / 4);
    cl_uint div2       = lpd / 4;
    cl_uint div2_min = std::max<cl_uint>(1, (lpd - 2) / 2);
    
    int     modP       = context.getExponent();
    cl_uint nTrans     = n;

    std::ostringstream ss;
    ss << buildOptions;
    ss 
      << " -DDIGIT_WIDTH_VALUE_1=" << pre.getDigitWidthValue1()
      << " -DDIGIT_WIDTH_VALUE_2=" << pre.getDigitWidthValue2()
      << " -DWG_SIZE="                     << wg
      << " -DLOCAL_PROPAGATION_DEPTH="     << lpd
      << " -DCARRY_WORKER="                << wCarry
      << " -DLOCAL_PROPAGATION_DEPTH_DIV4="      << div4
      << " -DLOCAL_PROPAGATION_DEPTH_DIV4_MIN="  << div4_min
      << " -DLOCAL_PROPAGATION_DEPTH_DIV2="      << div2
      << " -DLOCAL_PROPAGATION_DEPTH_DIV2_MIN="  << div2_min
      << " -DWORKER_NTT="                  << wNtt
      << " -DWORKER_NTT_2_STEPS="          << w2step
      << " -DMODULUS_P="                   << modP
      << " -DTRANSFORM_SIZE_N="            << nTrans
      << " -DLOCAL_SIZE="                  << ls
      << " -DLOCAL_SIZE2="                 << ls2
      << " -DLOCAL_SIZE3="                 << ls3;
    size_t idx6 = 6*2, idx7 = 7*2, idx8 = 8*2;
    

    try {
        const auto& tw   = pre.twiddles();
        const auto& i_tw = pre.invTwiddles();

        if (tw.size()  <= idx8+1 || 
            i_tw.size() <= idx8+1) 
        {
            throw std::out_of_range(
                "pre.twiddles()/pre.invTwiddles() trop court pour idx 6,7,8");
        }

        ss 
        << " -DW12_01_X=" << tw[idx6] 
        << " -DW12_01_Y=" << tw[idx6+1]
        << " -DW15_01_X=" << tw[idx7]
        << " -DW15_01_Y=" << tw[idx7+1]
        << " -DW15_2_X="  << tw[idx8]
        << " -DW15_2_Y="  << tw[idx8+1]

        << " -DWI12_01_X=" << i_tw[idx6] 
        << " -DWI12_01_Y=" << i_tw[idx6+1]
        << " -DWI15_01_X=" << i_tw[idx7]
        << " -DWI15_01_Y=" << i_tw[idx7+1]
        << " -DWI15_2_X="  << i_tw[idx8]
        << " -DWI15_2_Y="  << i_tw[idx8+1]
        << " -DW6="  << i_tw[6]
        << " -DW7="  << i_tw[7]
        << " -DW10="  << i_tw[10];


    }
    catch (const std::out_of_range& e) {
        std::cerr << "[WARNING] Indice hors‐limites pour les twiddles : "
                << e.what() << std::endl;
        //throw;
    }
    catch (const std::exception& e) {
        std::cerr << "[WARNING] Problème lors de la récupération des twiddles : "
                << e.what() << std::endl;
        //throw;
    }
    std::string buildOptions2 = ss.str();
    std::cout << "Building OpenCL program with options: " << buildOptions2 << std::endl;

    
    err = clBuildProgram(program_, 1, &device, buildOptions2.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        checkBuildError(program_, device);
        throw std::runtime_error("Failed to build OpenCL program.");
    }

    std::cout << "OpenCL program built successfully from: " << filePath << std::endl;
    cl_uint numKernels = 0;
    clCreateKernelsInProgram(program_, 0, nullptr, &numKernels);
    std::vector<cl_kernel> kernels(numKernels);
    clCreateKernelsInProgram(program_, numKernels, kernels.data(), nullptr);

/*   std::cout << "Compiled Kernels (" << numKernels << "):" << std::endl;
    for (auto& k : kernels) {
        size_t kernelNameSize;
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &kernelNameSize);
        std::vector<char> kernelName(kernelNameSize);
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, kernelNameSize, kernelName.data(), nullptr);
        std::cout << "  - " << kernelName.data() << std::endl;
        clReleaseKernel(k);
    }*/

}

Program::~Program() {
    if (program_) clReleaseProgram(program_);
}

cl_program Program::getProgram() const noexcept {
    return program_;
}

std::string Program::loadKernelSource(const std::string& filePath) const {
    namespace fs = std::filesystem;

    if (!fs::exists(filePath)) {
        throw std::runtime_error("Kernel source file not found: " + filePath);
    }

    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open kernel source file: " + filePath);
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void Program::checkBuildError(cl_program program, cl_device_id device) const {
    size_t logSize = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::vector<char> log(logSize);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    std::cerr << "OpenCL Build Log:\n" << log.data() << std::endl;
}

} // namespace opencl
