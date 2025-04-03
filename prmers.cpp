#define CL_TARGET_OPENCL_VERSION 200
#ifndef KERNEL_PATH
#define KERNEL_PATH ""
#endif

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
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <csignal>
#include <cstdio> 
#include <cstdarg> 
#include <map>
#include "proof/proof.h"


// Global variables for backup functionality
volatile std::sig_atomic_t g_interrupt_flag = 0; // Flag to indicate SIGINT received
unsigned int backup_interval = 120; // Default backup interval in seconds
std::string save_path = ".";       // Default save/load directory (current directory)
// Vector to store accumulated log messages
static std::vector<std::string> log_messages;

// Signal handler for SIGINT (Ctrl-C)
void signalHandler(int signum) {
    g_interrupt_flag = 1;
}

#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RED "\033[31m"
#define COLOR_RESET "\033[0m"
#define COLOR_MAGENTA "\033[35m"


using namespace std::chrono;

// -----------------------------------------------------------------------------
// Helpers for 64-bit arithmetic modulo p (p = 2^64 - 2^32 + 1)
// -----------------------------------------------------------------------------
static constexpr uint64_t MOD_P = (((1ULL << 32) - 1ULL) << 32) + 1ULL;
bool debug = false;

uint64_t mulModP(uint64_t a, uint64_t b) {
#ifdef _MSC_VER
    uint64_t hi;
    uint64_t lo = _umul128(a, b, &hi);
#else
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t lo = (uint64_t)prod;
#endif

    uint32_t A = (uint32_t)(hi >> 32);
    uint32_t B = (uint32_t)(hi & 0xffffffffULL);
    uint64_t r = lo;
    uint64_t oldr = r;
    r += ((uint64_t)B << 32);
    if (r < oldr)
        r += ((1ULL << 32) - 1ULL);
    uint64_t sub = (uint64_t)A + (uint64_t)B;
    if (r >= sub)
        r -= sub;
    else
        r = r + MOD_P - sub;
    if (r >= MOD_P)
        r -= MOD_P;
    return r;
}

// Add a carry onto the number and return the carry of the first digit_width bits
uint64_t digit_adc(const uint64_t lhs, const int digit_width, uint64_t & carry)
{
    const uint64_t s = lhs + carry;
    const uint64_t c =  s < lhs;
    carry = (s >> digit_width) + (c << (64 - digit_width));
    return s & ((uint64_t(1) << digit_width) - 1);
}

uint64_t powModP(uint64_t base, uint64_t exp) {
    uint64_t result = 1ULL;
    while(exp > 0ULL) {
        if(exp & 1ULL)
            result = mulModP(result, base);
        base = mulModP(base, base);
        exp >>= 1ULL;
    }
    return result;
}

uint64_t invModP(uint64_t x) {
    return powModP(x, MOD_P - 2ULL);
}

// -----------------------------------------------------------------------------
// Compute transform size for the given exponent (forcing power-of-4)
// -----------------------------------------------------------------------------
static size_t transformsize(uint32_t exponent) {
    /*int log_n = 0, w = 0;
    do {
        ++log_n;
        w = exponent / (1 << log_n);
    } while (((w + 1) * 2 + log_n) >= 63);
    if (log_n & 1)
        ++log_n;
    return (size_t)(1ULL << log_n);*/
        int log_n = 0; uint32_t w = 0;
		do
		{
			++log_n;
			w = exponent >> log_n;
		} while ((w + 1) * 2 + log_n >= 63);

		return size_t(1) << log_n;
}

// -----------------------------------------------------------------------------
// Precompute digit weights, inverse weights, and digit widths for a given p.
// -----------------------------------------------------------------------------
void precalc_for_p(uint32_t p,
                   std::vector<uint64_t>& digit_weight,
                   std::vector<uint64_t>& digit_invweight,
                   std::vector<int>& digit_width) {
    size_t n = transformsize(p);
    if(n<4){
        n=4;
    }
    digit_weight.resize(n);
    digit_invweight.resize(n);
    digit_width.resize(n);
    __uint128_t bigPminus1 = ((__uint128_t)MOD_P) - 1ULL;
    __uint128_t tmp = bigPminus1 / 192ULL;
    tmp /= n;
    tmp *= 5ULL;
    uint64_t exponent = (uint64_t)tmp;
    uint64_t nr2 = powModP(7ULL, exponent);
    uint64_t inv_n = invModP((uint64_t)n);
    uint32_t w_val = p / (uint32_t)n;
    digit_weight[0] = 1ULL;
    digit_invweight[0] = inv_n;
    uint32_t o = 0;
    for (size_t j = 0; j <= n; j++) {
        uint64_t qj = (uint64_t)p * j;
        uint64_t qq = (j == 0) ? 0ULL : (qj - 1ULL);
        uint32_t ceil_qj_n = (uint32_t)(qq / n + 1ULL);
        if (j > 0) {
            uint32_t c = ceil_qj_n - o;
            if ((c != w_val) && (c != w_val + 1)) {
                std::cerr << "Error: computed c (" << c << ") differs from w ("
                          << w_val << ") or w+1 (" << w_val + 1 << ") at j = " << j << std::endl;
                exit(1);
            }
            digit_width[j - 1] = static_cast<uint64_t>(c);
            if (j < n) {
                uint32_t r = (uint32_t)(qj % n);
                uint64_t nr2r = (r != 0U) ? powModP(nr2, (uint64_t)(n - r)) : 1ULL;
                digit_weight[j] = nr2r;
                uint64_t nr2r_inv = invModP(nr2r);
                digit_invweight[j] = mulModP(nr2r_inv, inv_n);
            }
        }
        if(j > 0)
            o = ceil_qj_n;
    }
}

// -----------------------------------------------------------------------------
// File reading helper
// -----------------------------------------------------------------------------
std::string readFile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Unable to open file: " + filename);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

// -----------------------------------------------------------------------------
// Usage printing helper (updated with new backup and path options)
// -----------------------------------------------------------------------------
void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <p> [-d <device_id>] [-O <options>] [-c <localCarryPropagationDepth>] [-profile] [-prp|-ll] [-t <backup_interval>] [-f <path>] [-vload2]" << std::endl;
    std::cout << "  <p>       : Minimum exponent to test (required)" << std::endl;
    std::cout << "  -d <device_id>: (Optional) Specify OpenCL device ID (default: 0)" << std::endl;
    std::cout << "  -O <options>  : (Optional) Enable OpenCL optimization flags (e.g., fastmath, mad, unsafe, nans, optdisable)" << std::endl;
    std::cout << "  -c <localCarryPropagationDepth>: (Optional) Set local carry propagation depth (default: 8)." << std::endl;
    std::cout << "  -profile      : (Optional) Enable kernel execution profiling." << std::endl;
    std::cout << "  -prp          : (Optional) Run in PRP mode (default). Set initial value to 3 and perform p iterations without executing kernel_sub2; final result must equal 9." << std::endl;
    std::cout << "  -ll           : (Optional) Run in Lucas-Lehmer mode. (Initial value 4 and p-2 iterations with kernel_sub2 executed.)" << std::endl;
    std::cout << "  -t <backup_interval>: (Optional) Specify backup interval in seconds (default: 120)." << std::endl;
    std::cout << "  -f <path>           : (Optional) Specify path for saving/loading files (default: current directory)." << std::endl;
    std::cout << "  -vload2            : (Optional) Enable data loading in blocks of 2 instead of 4." << std::endl;
    std::cout << "Example: " << progName << " 127 -O fastmath mad -c 16 -profile -ll -t 120 -f /my/backup/path -vload2" << std::endl;
}


// -----------------------------------------------------------------------------
// Build options helper
// -----------------------------------------------------------------------------
std::string getBuildOptions(int argc, char** argv) {
    std::string build_options = "";
    bool found_O = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-O") == 0 && i + 1 < argc) {
            found_O = true;
            for (int j = i + 1; j < argc; ++j) {
                std::string opt = argv[j];
                if (opt == "fastmath") build_options += " -cl-fast-relaxed-math";
                else if (opt == "mad") build_options += " -cl-mad-enable";
                else if (opt == "unsafe") build_options += " -cl-unsafe-math-optimizations";
                else if (opt == "nans") build_options += " -cl-no-signed-zeros";
                else if (opt == "optdisable") build_options += " -cl-opt-disable";
                else if (opt[0] == '-') break;
            }
            break;
        }
    }
    if (found_O && build_options.empty()) {
        std::cerr << "Error: -O specified but no valid options given. Use -h for help." << std::endl;
        exit(1);
    }
    return build_options;
}

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
                             const std::string& kernelName, size_t nmax,
                             bool profiling) {
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &workers, &localSize, 0, nullptr, profiling ? &event : nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error executing kernel '" << kernelName << "': " << getCLErrorString(err) << " (" << err << ")" << std::endl;
    }
    clFinish(queue);
    if (profiling && event) {
        cl_ulong start_time, end_time;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
        double duration = (end_time - start_time) / 1e3;
        std::cout << "[Profile] " << kernelName << " duration: " << duration << " µs" << std::endl;
        clReleaseEvent(event);
    }
    if (debug) {
        std::vector<uint64_t> locx(nmax, 0ULL);
        clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, nmax * sizeof(uint64_t), locx.data(), 0, nullptr, nullptr);
        std::cout << "After " << kernelName << " : x = [ ";
        for (size_t i = 0; i < 10; i++) std::cout << locx[i] << " ";
        std::cout << "...]" << std::endl;
    }
}

void displayProgress(uint32_t iter, uint32_t total_iters, double elapsedTime) {
    double progress = (100.0 * iter) / total_iters;
    double iters_per_sec = (elapsedTime > 0) ? iter / elapsedTime : 0.0;
    double remaining_time = (iters_per_sec > 0) ? (total_iters - iter) / iters_per_sec : 0.0;
    std::string color;
    if (progress < 50.0)
        color = COLOR_RED;
    else if (progress < 90.0)
        color = COLOR_YELLOW;
    else
        color = COLOR_GREEN;
    uint32_t seconds = static_cast<uint32_t>(remaining_time);
    uint32_t days = seconds / (24 * 3600);
    seconds %= (24 * 3600);
    uint32_t hours = seconds / 3600;
    seconds %= 3600;
    uint32_t minutes = seconds / 60;
    seconds %= 60;

    std::cout << "\r" << color
            << "Progress: " << std::fixed << std::setprecision(2) << progress << "% | "
            << "Elapsed: " << elapsedTime << "s | "
            << "Iterations/sec: " << iters_per_sec << " | "
            << "ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s       "
            << COLOR_RESET << std::flush;
}

void displayBackupInfo(uint32_t iter, uint32_t total_iters, double elapsedTime) {
    double progress = (100.0 * iter) / total_iters;
    double iters_per_sec = (elapsedTime > 0) ? iter / elapsedTime : 0.0;
    double remaining_time = (iters_per_sec > 0) ? (total_iters - iter) / iters_per_sec : 0.0;
    std::cout << "\r" << COLOR_MAGENTA
              << "[Backup] Progress: " << std::fixed << std::setprecision(2) << progress << "% | "
              << "Elapsed: " << elapsedTime << "s | "
              << "Iterations/sec: " << iters_per_sec << " | "
              << "ETA: " << (remaining_time > 0 ? remaining_time : 0.0) << "s       "
              << COLOR_RESET << std::flush;
}


void checkAndDisplayProgress(int32_t iter, uint32_t total_iters,
                             time_point<high_resolution_clock>& lastDisplay,
                             const time_point<high_resolution_clock>& start,
                             cl_command_queue queue) {
    auto duration = duration_cast<seconds>(high_resolution_clock::now() - lastDisplay).count();
    if (duration >= 10 || iter == -1) {
        if (iter == -1)
            iter = total_iters;
        double elapsedTime = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9;
        displayProgress(iter, total_iters, elapsedTime);
        lastDisplay = high_resolution_clock::now();
    }
}

// For quick logging (accumulate messages)
static void logmsg(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), fmt, ap);
    va_end(ap);
    log_messages.push_back(std::string(buffer));
}

// Function to display all accumulated log messages
static void flush_log() {
    for (const auto& msg : log_messages) {
        std::cerr << msg;
    }
    log_messages.clear();
}
// Helper function to create an OpenCL buffer with error checking
cl_mem createBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, const std::string& name) {
    cl_int err;
    cl_mem buffer = clCreateBuffer(context, flags, size, host_ptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating buffer '" << name << "': " << getCLErrorString(err) << std::endl;
        exit(1);
    }
    return buffer;
}

// Helper function to create an OpenCL kernel with error checking
cl_kernel createKernel(cl_program program, const std::string& kernelName) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel '" << kernelName << "': " << getCLErrorString(err) << std::endl;
        exit(1);
    }
    return kernel;
}


void executeFusionneNTT_Forward(cl_command_queue queue,cl_kernel kernel_ntt_mm_3_steps,cl_kernel kernel_ntt_mm_2_steps, cl_kernel kernel_radix2_square_radix2,
    cl_kernel kernel_ntt_mm, cl_kernel kernel_ntt_mm_first, cl_kernel kernel_ntt_last_m1,
    cl_kernel kernel_ntt_last_m1_n4,
    cl_mem buf_x, cl_mem buf_w,cl_mem buf_digit_weight, size_t n,
    size_t workers, size_t localSize, bool profiling,
    size_t maxLocalMem, bool _even_exponent) {
    if(n==4){
        size_t m = 1;
        clSetKernelArg(kernel_ntt_last_m1_n4, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_last_m1_n4, 1, sizeof(cl_mem), &buf_w);
        clSetKernelArg(kernel_ntt_last_m1_n4, 2, sizeof(cl_mem), &buf_digit_weight);
        clSetKernelArg(kernel_ntt_last_m1_n4, 3, 0, NULL); 
        executeKernelAndDisplay(queue, kernel_ntt_last_m1_n4, buf_x, workers, localSize,
            "kernel_ntt_last_m1_n4 (m=" + std::to_string(m) + ")", n, profiling);
    }
    else{
        size_t m = n / 4;

        clSetKernelArg(kernel_ntt_mm_first, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_mm_first, 1, sizeof(cl_mem), &buf_w);
        clSetKernelArg(kernel_ntt_mm_first, 2, sizeof(cl_mem), &buf_digit_weight);
        clSetKernelArg(kernel_ntt_mm_first, 3, sizeof(size_t), &m);
        clSetKernelArg(kernel_ntt_mm_first, 4, 0, NULL); 
        executeKernelAndDisplay(queue, kernel_ntt_mm_first, buf_x, workers, localSize,
            "kernel_ntt_radix4_mm_first (m=" + std::to_string(m) + ")", n, profiling);
        
        clSetKernelArg(kernel_ntt_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_mm_2_steps, 1, sizeof(cl_mem), &buf_w);
        size_t mm = n/4;
        for (size_t m = n / 16; m >= 32; m /= 16) {
            
            clSetKernelArg(kernel_ntt_mm_2_steps, 2, sizeof(size_t), &m);
            clSetKernelArg(kernel_ntt_mm_2_steps, 3, 0, NULL); 
            executeKernelAndDisplay(queue, kernel_ntt_mm_2_steps, buf_x, workers/4, localSize/4,
                "kernel_ntt_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling);
            mm = m/4;
        }
       //if(mm==256){
        if(mm==256){
            mm = 64;
            m = 64;
            clSetKernelArg(kernel_ntt_mm_3_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm_3_steps, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm_3_steps, 2, sizeof(size_t), &mm);
            clSetKernelArg(kernel_ntt_mm, 3, 0, NULL); 
            executeKernelAndDisplay(queue, kernel_ntt_mm_3_steps, buf_x, workers, localSize,
            "kernel_ntt_mm_3_steps (m=" + std::to_string(m) + ")", n, profiling);
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_last_m1, 2, 0, NULL);
            executeKernelAndDisplay(queue, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling);
        } 
         if(mm==32){
            mm = 8;
            m = 8;
            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(size_t), &mm);
            clSetKernelArg(kernel_ntt_mm, 3, 0, NULL); 
            executeKernelAndDisplay(queue, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling);
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(size_t), &mm);
            clSetKernelArg(kernel_ntt_mm, 3, 0, NULL); 
            executeKernelAndDisplay(queue, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling);
            mm = 1;
            m = 1;
            clSetKernelArg(kernel_radix2_square_radix2, 0, sizeof(cl_mem), &buf_x);
            executeKernelAndDisplay(queue, kernel_radix2_square_radix2, buf_x, workers*2, localSize,
                    "kernel_radix2_square_radix2 (m=" + std::to_string(m) + ") workers=" + std::to_string(workers*2), n, profiling);
        } 
        else if(mm==64){
            mm = 16;
            m = 16;
            clSetKernelArg(kernel_ntt_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm_2_steps, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm_2_steps, 2, sizeof(size_t), &m);
            clSetKernelArg(kernel_ntt_mm_2_steps, 3, 0, NULL); 
            executeKernelAndDisplay(queue, kernel_ntt_mm_2_steps, buf_x, workers/4, localSize/4,
            "kernel_ntt_mm_3_steps (m=" + std::to_string(m) + ")", n, profiling);
            
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_last_m1, 2, 0, NULL);
            executeKernelAndDisplay(queue, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling);
        }   
        else if(mm==8){
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(size_t), &mm);
            clSetKernelArg(kernel_ntt_mm, 3, 0, NULL); 
            executeKernelAndDisplay(queue, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling);
            mm = 1;
            m =1;
            clSetKernelArg(kernel_radix2_square_radix2, 0, sizeof(cl_mem), &buf_x);
            executeKernelAndDisplay(queue, kernel_radix2_square_radix2, buf_x, workers*2, localSize,
                    "kernel_radix2_square_radix2 (m=" + std::to_string(m) + ") workers=" + std::to_string(workers*2), n, profiling);
       }
       else if(mm==4){
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_last_m1, 2, 0, NULL);
            executeKernelAndDisplay(queue, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling);
       }   
       else if(mm==16){
            m = 4;
            mm=4;

            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(size_t), &mm);
            clSetKernelArg(kernel_ntt_mm, 3, 0, NULL);
  executeKernelAndDisplay(queue, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling);


 clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_last_m1, 2, 0, NULL);
            executeKernelAndDisplay(queue, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling);

       }
       else if(mm==2){
            clSetKernelArg(kernel_radix2_square_radix2, 0, sizeof(cl_mem), &buf_x);
            executeKernelAndDisplay(queue, kernel_radix2_square_radix2, buf_x, workers*2, localSize,
                    "kernel_radix2_square_radix2 (m=" + std::to_string(m) + ") workers=" + std::to_string(workers*2), n, profiling);
       }      
    }
}

void executeFusionneNTT_Inverse(cl_command_queue queue,cl_kernel kernel_ntt_inverse_mm_2_steps,
    cl_kernel kernel_inverse_ntt_mm, cl_kernel kernel_inverse_ntt_mm_last, cl_kernel kernel_inverse_ntt_m1,
    cl_kernel kernel_inverse_ntt_m1_n4,cl_mem buf_x, cl_mem buf_wi, cl_mem buf_digit_invweight, size_t n,
    size_t workers, size_t localSize, bool profiling,
    size_t maxLocalMem, bool _even_exponent) {
    size_t m = 0;
    if(n==4){
        m = 1;
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 1, sizeof(cl_mem), &buf_wi);
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 2, sizeof(cl_mem), &buf_digit_invweight);
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 3, 0, NULL);
        executeKernelAndDisplay(queue, kernel_inverse_ntt_m1_n4, buf_x, workers, localSize,
            "kernel_inverse_ntt_m1_n4 (m=" + std::to_string(m) + ")", n, profiling);
    }
    else{
       
        if(_even_exponent){

            
            m = 1;
            clSetKernelArg(kernel_inverse_ntt_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_inverse_ntt_m1, 1, sizeof(cl_mem), &buf_wi);
            clSetKernelArg(kernel_inverse_ntt_m1, 2, 0, NULL);
            executeKernelAndDisplay(queue, kernel_inverse_ntt_m1, buf_x, workers, localSize,
                "kernel_inverse_ntt_radix4_m1 (m=" + std::to_string(m) + ")", n, profiling);


            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 1, sizeof(cl_mem), &buf_wi);
            size_t mm = 4;
          
            for (size_t m = 4; m < n/16; m *= 16) {
                
                clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 2, sizeof(size_t), &m);
                clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 3, 0, NULL); 
                executeKernelAndDisplay(queue, kernel_ntt_inverse_mm_2_steps, buf_x, workers/4, localSize/4,
                    "kernel_ntt_inverse_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling);
                mm = m*16;
            }
            if(mm<=n/16 && n>8){
                clSetKernelArg(kernel_inverse_ntt_mm, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_inverse_ntt_mm, 1, sizeof(cl_mem), &buf_wi);
                clSetKernelArg(kernel_inverse_ntt_mm, 2, sizeof(size_t), &mm);
                clSetKernelArg(kernel_inverse_ntt_mm, 3, 0, NULL);
                executeKernelAndDisplay(queue, kernel_inverse_ntt_mm, buf_x, workers, localSize,
                        "kernel_inverse_ntt_radix4_mm (m=" + std::to_string(m) + ")", n, profiling);
            }

        }
        else{
            m = 2;
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 1, sizeof(cl_mem), &buf_wi);
            size_t mm = 2;
            for (size_t m = 2; m < n/16; m *= 16) {
                
                clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 2, sizeof(size_t), &m);
                clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 3, 0, NULL); 
                executeKernelAndDisplay(queue, kernel_ntt_inverse_mm_2_steps, buf_x, workers/4, localSize/4,
                    "kernel_ntt_inverse_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling);
                mm = m*16;
            }
            if(mm<=n/16 && n>8){
                clSetKernelArg(kernel_inverse_ntt_mm, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_inverse_ntt_mm, 1, sizeof(cl_mem), &buf_wi);
                clSetKernelArg(kernel_inverse_ntt_mm, 2, sizeof(size_t), &mm);
                clSetKernelArg(kernel_inverse_ntt_mm, 3, 0, NULL);
                executeKernelAndDisplay(queue, kernel_inverse_ntt_mm, buf_x, workers, localSize,
                        "kernel_inverse_ntt_radix4_mm (m=" + std::to_string(m) + ")", n, profiling);
            }
        }
        


        m = n/4;
        clSetKernelArg(kernel_inverse_ntt_mm_last, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 1, sizeof(cl_mem), &buf_wi);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 2, sizeof(cl_mem), &buf_digit_invweight);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 3, sizeof(size_t), &m);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 4, 0, NULL);
        executeKernelAndDisplay(queue, kernel_inverse_ntt_mm_last, buf_x, workers, localSize,
            "kernel_inverse_ntt_radix4_mm_last (m=" + std::to_string(m) + ")", n, profiling);
    }


    
}


void printVector(const std::vector<uint64_t>& vec, const std::string& name) {
    std::cout << name << " = [ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}
void printVector2(const std::vector<uint64_t>& vec, const std::string& name) {
    std::cout << name << " = [ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}
void printVector3(const std::vector<int>& vec, const std::string& name) {
    std::cout << name << " = [ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

void handleFinalCarry(std::vector<uint64_t>& x, const std::vector<int>& digit_width_cpu, size_t n) {
    x[0] += 1;
    uint64_t c = 0;
    
    for (size_t k = 0; k < n; ++k) {
        x[k] = digit_adc(x[k], digit_width_cpu[k], c);
    }

    while (c != 0) {
        for (size_t k = 0; k < n; ++k) {
            x[k] = digit_adc(x[k], digit_width_cpu[k], c);
            if (c == 0) break;
        }
    }

    x[0] -= 1;
}

// -----------------------------------------------------------------------------
// Main function
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Set the signal handler for SIGINT (Ctrl-C)
    std::signal(SIGINT, signalHandler);

    if (argc < 2) {
        std::cerr << "Error: Missing <p_min> argument.\n";
        printUsage(argv[0]);
        return 1;
    }
    uint32_t p = 0;
    int device_id = 0;  // Default device ID
    size_t localCarryPropagationDepth = 4;
    std::string mode = "prp"; 
    bool proof = false;
    bool profiling = false;
    bool has_p = false;
    bool force_carry = false;
    bool vload2 = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-debug") == 0) {
            debug = true;
        }
        else if (std::strcmp(argv[i], "-proof") == 0) {
            proof = true;
        }
        else if (std::strcmp(argv[i], "-profile") == 0) {
            profiling = true;
        }
        else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        else if (std::strcmp(argv[i], "-vload2") == 0) {
            vload2 = true;
        }
        else if (std::strcmp(argv[i], "-d") == 0) {
            if (i + 1 < argc) {
                device_id = std::atoi(argv[i + 1]);
                i++;
            } else {
                std::cerr << "Error: Missing value for -d <device_id>." << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "-c") == 0) {
            if (i + 1 < argc) {
                localCarryPropagationDepth = std::atoi(argv[i + 1]);
                force_carry = true;
                i++;
            } else {
                std::cerr << "Error: Missing value for -c <localCarryPropagationDepth>." << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "-prp") == 0) {
            mode = "prp";
        }
        else if (std::strcmp(argv[i], "-ll") == 0) {
            mode = "ll";
        }
        else if (std::strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                backup_interval = std::atoi(argv[i+1]);
                i++;
            } else {
                std::cerr << "Error: Missing value for -t <backup_interval>." << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "-f") == 0) {
            if (i + 1 < argc) {
                save_path = argv[i+1];
                i++;
            } else {
                std::cerr << "Error: Missing value for -f <path>." << std::endl;
                return 1;
            }
        }
        else if (!has_p) { 
            p = std::atoi(argv[i]);
            has_p = true;
        }
        else {
            std::cerr << "Warning: Ignoring unexpected argument '" << argv[i] << "'" << std::endl;
        }
    }

    if (!has_p) {
        std::cerr << "Error: No exponent provided. You must specify <p> as the first argument." << std::endl;
        return 1;
    }
    if (p < 89) {

        std::map<uint32_t, bool> knownResults = {
            {2, true}, {3, true}, {5, true}, {7, true},
            {13, true}, {17, true}, {19, true},
            {31, true}, {61, true}, {89, true}
        };

        auto it = knownResults.find(p);
        if (it == knownResults.end()) {
            std::cout << "\nKernel execution time: 0.0 seconds" << std::endl;
            std::cout << "Iterations per second: ∞ (simulated)" << std::endl;
            std::cout << "\nM" << p << " is composite." << std::endl;
            return 1;
        } else {
            std::cout << "\nKernel execution time: 0.0 seconds" << std::endl;
            std::cout << "Iterations per second: ∞ (simulated)" << std::endl;
            std::cout << "\nM" << p << " is prime!" << std::endl;
            return 0;
        }
    }

    if(p<13){
        mode = "ll";
    }
    if(mode == "ll")
        proof = false;
   
    if (profiling)
        std::cout << "\n🔍 Kernel profiling is activated. Performance metrics will be displayed.\n" << std::endl;

    std::cout << "PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)" << std::endl;
    std::cout << "Testing exponent: " << p << std::endl;
    std::cout << "Using OpenCL device ID: " << device_id << std::endl;
    std::cout << "Mode selected: " << (mode == "prp" ? "PRP" : "Lucas-Lehmer") << std::endl;
    std::cout << "Backup interval: " << backup_interval << " seconds" << std::endl;
    std::cout << "Save/Load path: " << save_path << std::endl;
    if(vload2)
        std::cout << "Mode vload2 " << save_path << std::endl;

    // -------------------------------------------------------------------------
    // Platform, Device, Context, and Command Queue Setup
    // -------------------------------------------------------------------------
    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if(err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "No OpenCL platform found." << std::endl;
        return 1;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[0];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if(err != CL_SUCCESS || numDevices == 0) {
        std::cerr << "No GPU device found." << std::endl;
        return 1;
    }
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    if (device_id < 0 || static_cast<uint64_t>(device_id) >= numDevices){
        std::cerr << "Invalid device id specified (" << device_id << "). Using device 0 instead." << std::endl;
        device_id = 0;
    }
    cl_device_id device = devices[device_id];
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        return 1;
    }

    // -------------------------------------------------------------------------
    // Read and Build OpenCL Program
    // -------------------------------------------------------------------------
    std::string kernelSource;
    try {
        std::string kernelFile = std::string(KERNEL_PATH) + "prmers.cl";
        if(vload2){
            kernelFile = std::string(KERNEL_PATH) + "prmers_vload2.cl";
        }
        
        kernelSource = readFile(kernelFile);
    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    const char* src = kernelSource.c_str();
    size_t srclen = kernelSource.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src, &srclen, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // Precompute parameters
    std::vector<uint64_t> digit_weight_cpu, digit_invweight_cpu;
    std::vector<int> digit_width_cpu;
    precalc_for_p(p, digit_weight_cpu, digit_invweight_cpu, digit_width_cpu);
    if(debug){
        printVector2(digit_weight_cpu, "digit_weight_cpu");
        printVector2(digit_invweight_cpu, "digit_invweight_cpu");
        printVector3(digit_width_cpu, "digit_width_cpu");
    }
   
    size_t n = transformsize(p);
    if(n<4){
        n=4;
    }
    size_t mm=n/4;
    for (size_t m = n / 16; m >= 32; m /= 16) {
        
        mm =m/16;
    }
    bool _even_exponent = !(mm == 8||mm==2||mm==32)||(n==4);
    if(debug){
        std::cout << "Size n for transform is n=" << n << std::endl;
        if(_even_exponent)
            std::cout << "_even_exponent is True" << std::endl;
        else
            std::cout << "_even_exponent is False" << std::endl;
    }

    // Precompute twiddle factors (forward and inverse)
    std::vector<uint64_t> twiddles(3 * n, 0ULL), inv_twiddles(3 * n, 0ULL);
    if(n >= 4) {
        uint64_t exp = (MOD_P - 1) / n;
        uint64_t root = powModP(7ULL, exp);
        uint64_t invroot = invModP(root);
        for (size_t m = n / 2, s = 1; m >= 1; m /= 2, s *= 2) {
            uint64_t r_s = powModP(root, s);
            uint64_t invr_s = powModP(invroot, s);
            uint64_t w_m = 1ULL, invw_m = 1ULL;
            for (size_t j = 0; j < m; j++) {
                size_t idx = 3 * (m + j);
                twiddles[idx + 0] = w_m;
                uint64_t w_m2 = mulModP(w_m, w_m);
                twiddles[idx + 1] = w_m2;
                twiddles[idx + 2] = mulModP(w_m2, w_m);

                inv_twiddles[idx + 0] = invw_m;
                uint64_t invw_m2 = mulModP(invw_m, invw_m);
                inv_twiddles[idx + 1] = invw_m2;
                inv_twiddles[idx + 2] = mulModP(invw_m2, invw_m);

                w_m = mulModP(w_m, r_s);
                invw_m = mulModP(invw_m, invr_s);
            }
        }
    }

    // Get work-group information from the device
    size_t workitem_size[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, nullptr);
    size_t maxThreads = workitem_size[0];
    size_t maxWork;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWork, nullptr);
    std::cout << "Max CL_DEVICE_MAX_WORK_GROUP_SIZE = " << maxWork << std::endl;
    std::cout << "Max CL_DEVICE_MAX_WORK_ITEM_SIZES = " << maxThreads << std::endl;
    size_t maxLocalMem;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxLocalMem), &maxLocalMem, NULL);
    std::cout << "Max CL_DEVICE_LOCAL_MEM_SIZE = " << maxLocalMem << std::endl;
    
    size_t workers = n;
    size_t workersNtt = n/4;
    size_t workersNtt2step = n/16;
    
    size_t localSize = maxWork;
    
    if(!force_carry){
        // check b^s > n * b^2
        int max_digit_width_cpu = *std::max_element(digit_width_cpu.begin(), digit_width_cpu.end());
        std::cout << "Max max_digit_width for IBDWT = " << max_digit_width_cpu << std::endl;
    
        while (std::pow(max_digit_width_cpu, localCarryPropagationDepth) < std::pow(max_digit_width_cpu,2) * n) {
            localCarryPropagationDepth *= 2;
        }
    }
    std::cout << "\nLaunching OpenCL kernel (p = " << p << "); computation may take a while." << std::endl;
    
    size_t constraint = std::max(n / 4, (size_t)1);
    while (workers % localSize != 0 || constraint % localSize != 0) {
        localSize /= 2;
        if (localSize < 1) { localSize = 1; break; }
    }
    size_t localSizeCarry = localSize;
    size_t workersCarry = 1;
    if(workers % localCarryPropagationDepth ==0){
        workersCarry = workers / localCarryPropagationDepth;
    }
    else{
        int localCarryPropagationDepth = 8;
        while (workers % localCarryPropagationDepth == 0) {
            workersCarry = workers / localCarryPropagationDepth;
            localCarryPropagationDepth *= 2;
        }
    }
    if (workersCarry<=1){
        workersCarry = 1;
        localCarryPropagationDepth = n;
    }
    if(workersCarry<localSizeCarry){
        localSizeCarry = workersCarry;
    }



    std::cout << "Transform size: " <<  n << std::endl;
    std::cout << "Final workers count: " << workers << std::endl;
    std::cout << "Work-groups count: " << localSize << std::endl;
    std::cout << "Work-groups size: " << ((workers < localSize) ? 1 : (workers / localSize)) << std::endl;
    std::cout << "Workers for carry propagation count: " << workersCarry << std::endl;
    std::cout << "Local carry propagation depht: " <<  localCarryPropagationDepth << std::endl;
    std::cout << "Local size carry: " <<  localSizeCarry << std::endl;
    uint32_t proofPower = ProofSet::bestPower(p);
    ProofSet proofSet(p, proofPower);
    if(proof)
        std::cout << "Proof Power : " <<  proofPower << std::endl;

    size_t workGroupSize = ((workers < localSize) ? 1 : (workers / localSize));
    size_t adjustedDepth = localCarryPropagationDepth / 4;
    size_t adjustedDepthMin = (localCarryPropagationDepth - 4) / 4;
    size_t adjustedDepth2 = localCarryPropagationDepth / 4;
    size_t adjustedDepthMin2 = (localCarryPropagationDepth - 2) / 2;
    if (adjustedDepthMin < 1) {
        adjustedDepthMin = 1;
    }
    if (adjustedDepth < 1) {
        adjustedDepth = 1;
    }
    if (adjustedDepthMin2 < 1) {
        adjustedDepthMin = 1;
    }
    if (adjustedDepth2 < 1) {
        adjustedDepth = 1;
    }
    // Append work-group size to build options
    std::string build_options = getBuildOptions(argc, argv);
    build_options += " -DWG_SIZE=" + std::to_string(workGroupSize) + " -DLOCAL_PROPAGATION_DEPTH=" + std::to_string(localCarryPropagationDepth) + " -DCARRY_WORKER=" + std::to_string(workersCarry) + " -DLOCAL_PROPAGATION_DEPTH_DIV4=" + std::to_string(adjustedDepth)+ " -DLOCAL_PROPAGATION_DEPTH_DIV4_MIN=" + std::to_string(adjustedDepthMin) + " -DLOCAL_PROPAGATION_DEPTH_DIV2=" + std::to_string(adjustedDepth2)+ " -DLOCAL_PROPAGATION_DEPTH_DIV2_MIN=" + std::to_string(adjustedDepthMin2) + " -DWORKER_NTT=" + std::to_string(workersNtt) + " -DWORKER_NTT_2_STEPS=" + std::to_string(workersNtt2step) ;
    std::cout << "Building OpenCL program with options: " << build_options << std::endl;
    err = clBuildProgram(program, 1, &device, build_options.empty() ? nullptr : build_options.c_str(), nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "Error during program build:" << std::endl;
        std::cerr << buildLog.data() << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // --------------------
    // Create Buffers using helper functions
    // --------------------
    cl_mem buf_digit_weight = createBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        digit_weight_cpu.size() * sizeof(uint64_t),
        (void*)digit_weight_cpu.data(), "buf_digit_weight");

    cl_mem buf_digit_invweight = createBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        digit_invweight_cpu.size() * sizeof(uint64_t),
        (void*)digit_invweight_cpu.data(), "buf_digit_invweight");

    cl_mem buf_digit_width = createBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        digit_width_cpu.size() * sizeof(int),
        (void*)digit_width_cpu.data(), "buf_digit_width");


    cl_mem buf_twiddles = createBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        twiddles.size() * sizeof(uint64_t),
        (void*)twiddles.data(), "buf_twiddles");

    cl_mem buf_inv_twiddles = createBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        inv_twiddles.size() * sizeof(uint64_t),
        (void*)inv_twiddles.data(), "buf_inv_twiddles");

    // Initialize vector x for state; later used to create buf_x.
    std::vector<uint64_t> x(n, 0ULL);
    // Check for existing state files to resume computation.
    unsigned int resume_iter = 0;
    std::string mers_filename = save_path + "/" + std::to_string(p) + mode + ".mers";
    std::string loop_filename = save_path + "/" + std::to_string(p) + mode + ".loop";
    std::ifstream loopFile(loop_filename);
    if(!debug && loopFile) {
        loopFile >> resume_iter;
        loopFile.close();
        std::cout << "Resuming from iteration " << resume_iter << " based on existing file " << loop_filename << std::endl;
        std::ifstream mersFile(mers_filename, std::ios::binary);
        if(mersFile) {
             mersFile.read(reinterpret_cast<char*>(x.data()), n * sizeof(uint64_t));
             mersFile.close();
             std::cout << "Loaded state from " << mers_filename << std::endl;
        }
    } else {
        // Initialize x based on mode: 3 for PRP, 4 for Lucas-Lehmer (LL)
        if (mode == "prp")
            x[0] = 3ULL;
        else
            x[0] = 4ULL;
    }
    




    std::vector<uint64_t> block_carry_init(workersCarry, 0ULL);
    cl_mem buf_x = createBuffer(context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        n * sizeof(uint64_t),
        x.data(), "buf_x");

    cl_mem buf_block_carry = createBuffer(context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        workersCarry * sizeof(uint64_t),
        block_carry_init.data(), "buf_block_carry");




    // --------------------
    // Create Kernels using helper functions
    // --------------------
    cl_kernel k_forward_ntt_mm     = createKernel(program, "kernel_ntt_radix4_mm");
    cl_kernel k_forward_ntt_mm_first     = createKernel(program, "kernel_ntt_radix4_mm_first");
    cl_kernel k_forward_ntt_last_m1 = createKernel(program, "kernel_ntt_radix4_last_m1");
    cl_kernel k_forward_ntt_last_m1_n4 = createKernel(program, "kernel_ntt_radix4_last_m1_n4");
    cl_kernel k_inverse_ntt_m1     = createKernel(program, "kernel_inverse_ntt_radix4_m1");
    cl_kernel k_inverse_ntt_m1_n4     = createKernel(program, "kernel_inverse_ntt_radix4_m1_n4");
    cl_kernel k_inverse_ntt_mm     = createKernel(program, "kernel_inverse_ntt_radix4_mm");
    cl_kernel k_inverse_ntt_mm_last     = createKernel(program, "kernel_inverse_ntt_radix4_mm_last");
    cl_kernel k_sub2 = createKernel(program, "kernel_sub2");
    cl_kernel k_carry = createKernel(program, "kernel_carry");
    cl_kernel k_carry_2 = createKernel(program, "kernel_carry_2");
    cl_kernel k_forward_ntt_mm_3steps     = createKernel(program, "kernel_ntt_radix4_mm_3steps");
    cl_kernel k_forward_ntt_mm_2steps     = createKernel(program, "kernel_ntt_radix4_mm_2steps");
    cl_kernel k_inverse_ntt_mm_2steps     = createKernel(program, "kernel_ntt_radix4_inverse_mm_2steps");
    cl_kernel k_ntt_radix2_square_radix2     = createKernel(program, "kernel_ntt_radix2_square_radix2");


    // --------------------
    // Set Kernel Arguments
    // --------------------
    cl_int errKernel = CL_SUCCESS;
    // kernel_sub2
    errKernel  = clSetKernelArg(k_sub2, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_sub2, 1, sizeof(cl_mem), &buf_digit_width);
    errKernel |= clSetKernelArg(k_sub2, 2, sizeof(uint64_t), &n);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for kernel_sub2: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }
    // kernel_carry
    errKernel  = clSetKernelArg(k_carry, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_carry, 1, sizeof(cl_mem), &buf_block_carry);
    errKernel |= clSetKernelArg(k_carry, 2, sizeof(cl_mem), &buf_digit_width);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for carry: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }
    // kernel_carry_2
    errKernel  = clSetKernelArg(k_carry_2, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_carry_2, 1, sizeof(cl_mem), &buf_block_carry);
    errKernel |= clSetKernelArg(k_carry_2, 2, sizeof(cl_mem), &buf_digit_width);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for carry 2: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }



    // -------------------------------------------------------------------------
    // Main Computation Loop
    // -------------------------------------------------------------------------
    uint32_t total_iters;
    if (mode == "prp") {
        total_iters = p;  // For PRP mode, execute p iterations.
    } else {
        total_iters = p - 2;  // For Lucas-Lehmer mode, execute p-2 iterations.
    }
    // Adjust loop start if resuming from a saved state.
    auto startTime = high_resolution_clock::now();
    auto lastDisplay = startTime;
    auto last_backup_time = startTime;
    checkAndDisplayProgress(0, total_iters, lastDisplay, startTime, queue);
    
    if(debug)
        std::cout << "Number of iterations to be done = " << total_iters << std::endl;
            

    // Main loop now starts from resume_iter (if any) to total_iters.
    for (uint32_t iter = resume_iter; iter < total_iters; iter++) {
        executeFusionneNTT_Forward(queue,k_forward_ntt_mm_3steps,k_forward_ntt_mm_2steps,k_ntt_radix2_square_radix2,
            k_forward_ntt_mm, k_forward_ntt_mm_first, k_forward_ntt_last_m1,k_forward_ntt_last_m1_n4,
            buf_x, buf_twiddles, buf_digit_weight, n, workers/4, localSize, profiling, maxLocalMem, _even_exponent);

        executeFusionneNTT_Inverse(queue,k_inverse_ntt_mm_2steps,
            k_inverse_ntt_mm, k_inverse_ntt_mm_last, k_inverse_ntt_m1,k_inverse_ntt_m1_n4,
            buf_x, buf_inv_twiddles,buf_digit_invweight, n, workers/4, localSize, profiling, maxLocalMem, _even_exponent);
        executeKernelAndDisplay(queue, k_carry, buf_x, workersCarry, localSizeCarry, "kernel_carry", n, profiling);
        executeKernelAndDisplay(queue, k_carry_2, buf_x, workersCarry, localSizeCarry, "kernel_carry_2", n, profiling);
        // In Lucas-Lehmer mode, execute kernel_sub2; in PRP mode, skip it.
        if (mode == "ll") {
            executeKernelAndDisplay(queue, k_sub2, buf_x, 1, 1, "kernel_sub2", n, profiling);
        }
        checkAndDisplayProgress(iter-resume_iter, total_iters, lastDisplay, startTime, queue);
        
        if (proof && std::find(proofSet.points.begin(), proofSet.points.end(), iter) != proofSet.points.end()) {
            clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, n * sizeof(uint64_t), x.data(), 0, nullptr, nullptr);
            Words partialRes = ProofSet::fromUint64(x, p);
            proofSet.save(iter, partialRes);
            //std::cout << "\nProof generation : Saved partial residue at iteration " << iter << std::endl;
             logmsg("Proof generation : Saved partial residue at iteration %lu\n", iter);
        }

        // Check if backup interval has been reached or if SIGINT (Ctrl-C) was received.

        auto now = high_resolution_clock::now();
        if(duration_cast<seconds>(now - last_backup_time).count() >= backup_interval || g_interrupt_flag) {
            clFinish(queue);
            // Read back buf_x from GPU
            clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, n * sizeof(uint64_t), x.data(), 0, nullptr, nullptr);
            // Save buf_x state to file (binary file)
            std::ofstream mersOut(mers_filename, std::ios::binary);
            if(mersOut) {
                mersOut.write(reinterpret_cast<const char*>(x.data()), n * sizeof(uint64_t));
                mersOut.close();
                std::cout << "\nState saved to " << mers_filename << std::endl;
            } else {
                std::cerr << "\nError saving state to " << mers_filename << std::endl;
            }
            // Save current iteration state to file (text file)
            std::ofstream loopOut(loop_filename);
            if(loopOut) {
                loopOut << (iter + 1);
                loopOut.close();
                std::cout << "Loop iteration saved to " << loop_filename << std::endl;
            } else {
                std::cerr << "Error saving loop state to " << loop_filename << std::endl;
            }
            last_backup_time = now;
            double backup_elapsed = duration_cast<nanoseconds>(now - startTime).count() / 1e9;
            displayBackupInfo(iter, total_iters, backup_elapsed);

            // If interrupted by SIGINT, exit gracefully after saving state.
            // If interrupted, exit early after cleanup.
            if (g_interrupt_flag) {
                std::cout << "Exiting early due to interrupt." << std::endl;
                // Perform cleanup and release resources:
                clReleaseMemObject(buf_x);
                clReleaseMemObject(buf_twiddles);
                clReleaseMemObject(buf_inv_twiddles);
                clReleaseMemObject(buf_digit_weight);
                clReleaseMemObject(buf_digit_invweight);
                clReleaseMemObject(buf_digit_width);
                clReleaseMemObject(buf_block_carry);
                clReleaseKernel(k_forward_ntt_mm);
                clReleaseKernel(k_forward_ntt_mm_3steps);
                clReleaseKernel(k_forward_ntt_mm_2steps);
                clReleaseKernel(k_inverse_ntt_mm_2steps);
                clReleaseKernel(k_forward_ntt_mm_first);
                clReleaseKernel(k_ntt_radix2_square_radix2);
                clReleaseKernel(k_forward_ntt_last_m1);
                clReleaseKernel(k_forward_ntt_last_m1_n4);
                clReleaseKernel(k_inverse_ntt_m1);
                clReleaseKernel(k_inverse_ntt_m1_n4);
                clReleaseKernel(k_inverse_ntt_mm);
                clReleaseKernel(k_inverse_ntt_mm_last);
                clReleaseKernel(k_sub2);
                clReleaseKernel(k_carry);
                clReleaseKernel(k_carry_2);
                clReleaseProgram(program);
                clReleaseCommandQueue(queue);
                clReleaseContext(context);
                return 0;
            }
        }
        
    }
    //
    // After successful completion, remove the loop state file as it is no longer needed.
    std::remove(loop_filename.c_str());

    checkAndDisplayProgress(-1, total_iters, lastDisplay, startTime, queue);
    clFinish(queue);
    auto endTime = high_resolution_clock::now();
    
    // Read back result from buf_x
    clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, n * sizeof(uint64_t), x.data(), 0, nullptr, nullptr);
    handleFinalCarry(x, digit_width_cpu, n);
    std::chrono::duration<double> elapsed = endTime - startTime;
    double elapsedTime = elapsed.count();
    double iters_per_sec = total_iters / elapsedTime;
    std::cout << "\nKernel execution time: " << elapsedTime << " seconds" << std::endl;
    std::cout << "Iterations per second: " << iters_per_sec 
              << " (" << total_iters << " iterations in total)" << std::endl;
    if(proof){
        flush_log();
        Words finalRes = ProofSet::fromUint64(x, p);
        auto [myProof, rExps] = proofSet.computeProof(finalRes);
        std::filesystem::path outDir = save_path; 
        std::filesystem::create_directories(outDir);
        auto proofFile = myProof.fileName(outDir);
        myProof.save(proofFile);
        std::cout << "Proof is saved in a file!\n\n";
        /*if (!myProof.verify()) {
            std::cout << "Proof is invalid!\n";
        } else {
            std::cout << "Proof is valid. M" << p 
                    << (myProof.isProbablePrime() ? " prime?\n" : " composite.\n");
        }*/
    }
    if (mode == "prp") {
        bool resultIs9 = (x[0] == 9) && std::all_of(x.begin() + 1, x.end(), [](uint64_t v){ return v == 0; });
        std::cout << "\nM" << p << " PRP test " << (resultIs9 ? "succeeded (result is 9)." : "failed (result is not 9).") << std::endl;
        return resultIs9 ? 0 : 1;
    } else {
        bool isPrime = std::all_of(x.begin(), x.end(), [](uint64_t v) { return v == 0; });
        std::cout << "\nM" << p << " is " << (isPrime ? "prime!" : "composite.") << std::endl;
        return isPrime ? 0 : 1;
    }


    


    // -------------------------------------------------------------------------
    // Cleanup: Release all OpenCL resources
    // -------------------------------------------------------------------------
    clReleaseMemObject(buf_x);
    clReleaseMemObject(buf_twiddles);
    clReleaseMemObject(buf_inv_twiddles);
    clReleaseMemObject(buf_digit_weight);
    clReleaseMemObject(buf_digit_invweight);
    clReleaseMemObject(buf_digit_width);
    clReleaseMemObject(buf_block_carry);
    clReleaseKernel(k_forward_ntt_mm);
    clReleaseKernel(k_forward_ntt_mm_3steps);
    clReleaseKernel(k_forward_ntt_mm_2steps);
    clReleaseKernel(k_inverse_ntt_mm_2steps);
    clReleaseKernel(k_ntt_radix2_square_radix2);
    clReleaseKernel(k_forward_ntt_mm_first);
    clReleaseKernel(k_forward_ntt_last_m1);
    clReleaseKernel(k_forward_ntt_last_m1_n4);
    clReleaseKernel(k_inverse_ntt_m1);
    clReleaseKernel(k_inverse_ntt_m1_n4);
    clReleaseKernel(k_inverse_ntt_mm);
    clReleaseKernel(k_inverse_ntt_mm_last);
    clReleaseKernel(k_sub2);
    clReleaseKernel(k_carry);
    clReleaseKernel(k_carry_2);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
