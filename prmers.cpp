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
#include <cstdio>    // For std::remove

// Global variables for backup functionality
volatile std::sig_atomic_t g_interrupt_flag = 0; // Flag to indicate SIGINT received
unsigned int backup_interval = 120; // Default backup interval in seconds
std::string save_path = ".";       // Default save/load directory (current directory)

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
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t lo = (uint64_t)prod;
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
    int log_n = 0, w = 0;
    do {
        ++log_n;
        w = exponent / (1 << log_n);
    } while (((w + 1) * 2 + log_n) >= 63);
    if (log_n & 1)
        ++log_n;
    return (size_t)(1ULL << log_n);
}

// -----------------------------------------------------------------------------
// Precompute digit weights, inverse weights, and digit widths for a given p.
// -----------------------------------------------------------------------------
void precalc_for_p(uint32_t p,
                   std::vector<uint64_t>& digit_weight,
                   std::vector<uint64_t>& digit_invweight,
                   std::vector<int>& digit_width) {
    size_t n = transformsize(p);
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
            digit_width[j - 1] = c;
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
    std::cout << "Usage: " << progName << " <p> [-d <device_id>] [-O <options>] [-c <localCarryPropagationDepth>] [-profile] [-prp|-ll] [-t <backup_interval>] [-f <path>]" << std::endl;
    std::cout << "  <p>       : Minimum exponent to test (required)" << std::endl;
    std::cout << "  -d <device_id>: (Optional) Specify OpenCL device ID (default: 0)" << std::endl;
    std::cout << "  -O <options>  : (Optional) Enable OpenCL optimization flags (e.g., fastmath, mad, unsafe, nans, optdisable)" << std::endl;
    std::cout << "  -c <localCarryPropagationDepth>: (Optional) Set local carry propagation depth (default: 8)." << std::endl;
    std::cout << "  -profile      : (Optional) Enable kernel execution profiling." << std::endl;
    std::cout << "  -prp          : (Optional) Run in PRP mode (default). Set initial value to 3 and perform p iterations without executing kernel_sub2; final result must equal 9." << std::endl;
    std::cout << "  -ll           : (Optional) Run in Lucas-Lehmer mode. (Initial value 4 and p-2 iterations with kernel_sub2 executed.)" << std::endl;
    std::cout << "  -t <backup_interval>: (Optional) Specify backup interval in seconds (default: 120)." << std::endl;
    std::cout << "  -f <path>           : (Optional) Specify path for saving/loading files (default: current directory)." << std::endl;
    std::cout << "Example: " << progName << " 127 -O fastmath mad -c 16 -profile -ll -t 120 -f /my/backup/path" << std::endl;
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
        for (size_t i = 0; i < nmax; i++) std::cout << locx[i] << " ";
        std::cout << "]" << std::endl;
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
            << "ETA: " << days << "j " << hours << "h " << minutes << "m " << seconds << "s       "
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
    if (duration >= 5 || iter == -1) {
        if (iter == -1)
            iter = total_iters;
        double elapsedTime = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9;
        displayProgress(iter, total_iters, elapsedTime);
        lastDisplay = high_resolution_clock::now();
    }
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


void executeFusionneNTT_Forward(cl_command_queue queue,
                                cl_kernel kernel_ntt, cl_kernel kernel_ntt_alt,
                                cl_kernel kernel_ntt_last, cl_kernel kernel_ntt_alt_last,
                                cl_mem buf_x, cl_mem buf_w, size_t n,
                                size_t workers, size_t localSize, bool profiling,
                                size_t maxLocalMem) {
    // Loop over m values: from n/4 down to 1.
    // The last iteration occurs when m == 1.
    for (size_t m = n / 4; m >= 1; m /= 4) {
        size_t local_twiddle_size = 3 * m * sizeof(cl_ulong);
        // Determine if this is the last iteration (for forward, last when m == 1)
        bool isLast = (m == 1);
        if (local_twiddle_size > maxLocalMem) {
            // Use the alternative kernel that does not use local memory.
            if (isLast) {
                // Use alternative fused kernel for the last iteration.
                clSetKernelArg(kernel_ntt_alt_last, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_ntt_alt_last, 1, sizeof(cl_mem), &buf_w);
                clSetKernelArg(kernel_ntt_alt_last, 2, sizeof(size_t), &n);
                clSetKernelArg(kernel_ntt_alt_last, 3, sizeof(size_t), &m);
                // Do not allocate local memory.
                clSetKernelArg(kernel_ntt_alt_last, 4, 0, NULL);
                executeKernelAndDisplay(queue, kernel_ntt_alt_last, buf_x, workers, localSize,
                                         "kernel_ntt_radix4_last_alt (m=" + std::to_string(m) + ")", n, profiling);
            } else {
                // Use alternative regular kernel.
                clSetKernelArg(kernel_ntt_alt, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_ntt_alt, 1, sizeof(cl_mem), &buf_w);
                clSetKernelArg(kernel_ntt_alt, 2, sizeof(size_t), &n);
                clSetKernelArg(kernel_ntt_alt, 3, sizeof(size_t), &m);
                clSetKernelArg(kernel_ntt_alt, 4, 0, NULL);
                executeKernelAndDisplay(queue, kernel_ntt_alt, buf_x, workers, localSize,
                                         "kernel_ntt_radix4_alt (m=" + std::to_string(m) + ")", n, profiling);
            }
        } else {
            // Use the main kernel which uses local memory.
            if (isLast) {
                // Use the fused version for the last iteration.
                clSetKernelArg(kernel_ntt_last, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_ntt_last, 1, sizeof(cl_mem), &buf_w);
                clSetKernelArg(kernel_ntt_last, 2, sizeof(size_t), &n);
                clSetKernelArg(kernel_ntt_last, 3, sizeof(size_t), &m);
                clSetKernelArg(kernel_ntt_last, 4, local_twiddle_size, NULL);
                executeKernelAndDisplay(queue, kernel_ntt_last, buf_x, workers, localSize,
                                         "kernel_ntt_radix4_last (m=" + std::to_string(m) + ")", n, profiling);
            } else {
                // Use the regular main kernel.
                clSetKernelArg(kernel_ntt, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_ntt, 1, sizeof(cl_mem), &buf_w);
                clSetKernelArg(kernel_ntt, 2, sizeof(size_t), &n);
                clSetKernelArg(kernel_ntt, 3, sizeof(size_t), &m);
                clSetKernelArg(kernel_ntt, 4, local_twiddle_size, NULL);
                executeKernelAndDisplay(queue, kernel_ntt, buf_x, workers, localSize,
                                         "kernel_ntt_radix4 (m=" + std::to_string(m) + ")", n, profiling);
            }
        }
        // Prevent infinite loop when m == 1 (since m would remain 1 on unsigned division).
        if (m == 1)
            break;
    }
}

void executeFusionneNTT_Inverse(cl_command_queue queue,
                                cl_kernel kernel_ntt, cl_kernel kernel_ntt_alt,
                                cl_mem buf_x, cl_mem buf_w, size_t n,
                                size_t workers, size_t localSize, bool profiling,
                                size_t maxLocalMem) {
    // In the inverse loop, m increases from 1 to n/4.
    // The last iteration is when m == n/4.
    for (size_t m = 1; m <= n / 4; m *= 4) {
        size_t local_twiddle_size = 3 * m * sizeof(cl_ulong);
        bool isLast = (m == n / 4);
        if (local_twiddle_size > maxLocalMem) {
            // Use the alternative inverse kernel.
                clSetKernelArg(kernel_ntt_alt, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_ntt_alt, 1, sizeof(cl_mem), &buf_w);
                clSetKernelArg(kernel_ntt_alt, 2, sizeof(size_t), &n);
                clSetKernelArg(kernel_ntt_alt, 3, sizeof(size_t), &m);
                clSetKernelArg(kernel_ntt_alt, 4, 0, NULL);
                executeKernelAndDisplay(queue, kernel_ntt_alt, buf_x, workers, localSize,
                                         "kernel_inverse_ntt_radix4_alt (m=" + std::to_string(m) + ")", n, profiling);
        } else {
            // Use the main inverse kernel with local memory.
                clSetKernelArg(kernel_ntt, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_ntt, 1, sizeof(cl_mem), &buf_w);
                clSetKernelArg(kernel_ntt, 2, sizeof(size_t), &n);
                clSetKernelArg(kernel_ntt, 3, sizeof(size_t), &m);
                clSetKernelArg(kernel_ntt, 4, local_twiddle_size, NULL);
                executeKernelAndDisplay(queue, kernel_ntt, buf_x, workers, localSize,
                                         "kernel_inverse_ntt_radix4 (m=" + std::to_string(m) + ")", n, profiling);
        }
    }
}



void printVector(const std::vector<int>& vec, const std::string& name) {
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
    size_t localCarryPropagationDepth = 8;
    std::string mode = "prp"; // Default mode is PRP
    // Parse command-line options (including new -t and -f)
    bool profiling = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-debug") == 0)
            debug = true;
        else if (std::strcmp(argv[i], "-profile") == 0) {
            profiling = true;
        }
        else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
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
        else {
            // Assume this argument is the exponent p if it does not match any flag.
            p = std::atoi(argv[i]);
        }
    }
    if(p<13){
        mode = "ll";
    }
   
    if (profiling)
        std::cout << "\n🔍 Kernel profiling is activated. Performance metrics will be displayed.\n" << std::endl;

    std::cout << "PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)" << std::endl;
    std::cout << "Testing exponent: " << p << std::endl;
    std::cout << "Using OpenCL device ID: " << device_id << std::endl;
    std::cout << "Mode selected: " << (mode == "prp" ? "PRP" : "Lucas-Lehmer") << std::endl;
    std::cout << "Backup interval: " << backup_interval << " seconds" << std::endl;
    std::cout << "Save/Load path: " << save_path << std::endl;

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
    if(device_id < 0 || device_id >= static_cast<int>(numDevices)) {
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
        printVector(digit_width_cpu, "digit_width_cpu");
    }
   
    size_t n = transformsize(p);
    if (debug)
        std::cout << "Size n for transform is n=" << n << std::endl;

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
    
    std::cout << "\nLaunching OpenCL kernel (p = " << p << "); computation may take a while." << std::endl;
    size_t workers = n;
    size_t localSize = maxWork;
   
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
    
    size_t workGroupSize = ((workers < localSize) ? 1 : (workers / localSize));
    // Append work-group size to build options
    std::string build_options = getBuildOptions(argc, argv);
    build_options += " -DWG_SIZE=" + std::to_string(workGroupSize) + " -DLOCAL_PROPAGATION_DEPTH=" + std::to_string(localCarryPropagationDepth) + " -DCARRY_WORKER=" + std::to_string(workersCarry) ;
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
    if(loopFile) {
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
    std::vector<uint64_t> block_carry_init_out(workersCarry, 0ULL);
    cl_mem buf_x = createBuffer(context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        n * sizeof(uint64_t),
        x.data(), "buf_x");

    cl_mem buf_block_carry = createBuffer(context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        workersCarry * sizeof(uint64_t),
        block_carry_init.data(), "buf_block_carry");

    cl_mem buf_block_carry_out = createBuffer(context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        workersCarry * sizeof(uint64_t),
        block_carry_init_out.data(), "buf_block_carry_out");

    cl_int zero = 0;
    cl_mem flagBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &zero, &err);
    

    // --------------------
    // Create Kernels using helper functions
    // --------------------
    cl_kernel k_precomp = createKernel(program, "kernel_precomp");
    cl_kernel k_postcomp = createKernel(program, "kernel_postcomp");
    cl_kernel k_forward_ntt         = createKernel(program, "kernel_ntt_radix4");
    cl_kernel k_inverse_ntt         = createKernel(program, "kernel_inverse_ntt_radix4");
    cl_kernel k_forward_ntt_alt     = createKernel(program, "kernel_ntt_radix4_alt");
    cl_kernel k_inverse_ntt_alt     = createKernel(program, "kernel_inverse_ntt_radix4_alt");
    cl_kernel k_forward_ntt_last    = createKernel(program, "kernel_ntt_radix4_last");
    cl_kernel k_forward_ntt_alt_last = createKernel(program, "kernel_ntt_radix4_last_alt");
    cl_kernel k_square = createKernel(program, "kernel_square");
    cl_kernel k_sub2 = createKernel(program, "kernel_sub2");
    cl_kernel k_carry = createKernel(program, "kernel_carry");
    cl_kernel k_carry_2 = createKernel(program, "kernel_carry_2");


    // --------------------
    // Set Kernel Arguments
    // --------------------
    cl_int errKernel = CL_SUCCESS;
    // kernel_precomp
    errKernel  = clSetKernelArg(k_precomp, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_precomp, 1, sizeof(cl_mem), &buf_digit_weight);
    errKernel |= clSetKernelArg(k_precomp, 2, sizeof(uint64_t), &n);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for kernel_precomp: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }
    // kernel_postcomp
    errKernel  = clSetKernelArg(k_postcomp, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_postcomp, 1, sizeof(cl_mem), &buf_digit_invweight);
    errKernel |= clSetKernelArg(k_postcomp, 2, sizeof(uint64_t), &n);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for kernel_postcomp: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }
    // kernel_square
    errKernel  = clSetKernelArg(k_square, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_square, 1, sizeof(uint64_t), &n);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for kernel_square: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }
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
    errKernel |= clSetKernelArg(k_carry, 3, sizeof(uint64_t), &n);
    errKernel |= clSetKernelArg(k_carry, 4, sizeof(cl_mem), &flagBuffer);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for carry: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }
    // kernel_carry_2
    errKernel  = clSetKernelArg(k_carry_2, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_carry_2, 1, sizeof(cl_mem), &buf_block_carry);
    errKernel |= clSetKernelArg(k_carry_2, 2, sizeof(cl_mem), &buf_block_carry_out);
    errKernel |= clSetKernelArg(k_carry_2, 3, sizeof(cl_mem), &buf_digit_width);
    errKernel |= clSetKernelArg(k_carry_2, 4, sizeof(uint64_t), &n);
    errKernel |= clSetKernelArg(k_carry_2, 5, sizeof(cl_mem), &flagBuffer);
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
    

    // Main loop now starts from resume_iter (if any) to total_iters.
    for (uint32_t iter = resume_iter; iter < total_iters; iter++) {
        executeKernelAndDisplay(queue, k_precomp, buf_x, workers, localSize, "kernel_precomp", n, profiling);
        executeFusionneNTT_Forward(queue,
            k_forward_ntt, k_forward_ntt_alt, k_forward_ntt_last, k_forward_ntt_alt_last,
            buf_x, buf_twiddles, n, workers, localSize, profiling, maxLocalMem);
        executeFusionneNTT_Inverse(queue,
            k_inverse_ntt, k_inverse_ntt_alt,
            buf_x, buf_inv_twiddles, n, workers, localSize, profiling, maxLocalMem);
        executeKernelAndDisplay(queue, k_postcomp, buf_x, workers, localSize, "kernel_postcomp", n, profiling);
        executeKernelAndDisplay(queue, k_carry, buf_x, workersCarry, localSizeCarry, "kernel_carry", n, profiling);
        executeKernelAndDisplay(queue, k_carry_2, buf_x, workersCarry, localSizeCarry, "kernel_carry_2", n, profiling);
        // In Lucas-Lehmer mode, execute kernel_sub2; in PRP mode, skip it.
        if (mode == "ll") {
            executeKernelAndDisplay(queue, k_sub2, buf_x, 1, 1, "kernel_sub2", n, profiling);
        }
        checkAndDisplayProgress(iter, total_iters, lastDisplay, startTime, queue);

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
                clReleaseMemObject(buf_block_carry_out);
                clReleaseKernel(k_precomp);
                clReleaseKernel(k_postcomp);
                clReleaseKernel(k_forward_ntt);
                clReleaseKernel(k_inverse_ntt);
                clReleaseKernel(k_forward_ntt_alt);
                clReleaseKernel(k_inverse_ntt_alt);
                clReleaseKernel(k_forward_ntt_last);
                clReleaseKernel(k_forward_ntt_alt_last);
                clReleaseKernel(k_square);
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
    // After successful completion, remove the loop state file as it is no longer needed.
    std::remove(loop_filename.c_str());

    checkAndDisplayProgress(-1, total_iters, lastDisplay, startTime, queue);
    clFinish(queue);
    auto endTime = high_resolution_clock::now();

    // Read back result from buf_x
    clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, n * sizeof(uint64_t), x.data(), 0, nullptr, nullptr);
    handleFinalCarry(x, digit_width_cpu, n);

    if (mode == "prp") {
        // In PRP mode, check that the result is exactly 9 (x[0]==9 and all other digits are 0)
        bool resultIs9 = (x[0] == 9) && std::all_of(x.begin() + 1, x.end(), [](uint64_t v){ return v == 0; });
        std::cout << "\nM" << p << " PRP test " << (resultIs9 ? "succeeded (result is 9)." : "failed (result is not 9).") << std::endl;
    } else {
        bool isPrime = std::all_of(x.begin(), x.end(), [](uint64_t v) { return v == 0; });
        std::cout << "\nM" << p << " is " << (isPrime ? "prime!" : "composite.") << std::endl;
    }
    
    std::chrono::duration<double> elapsed = endTime - startTime;
    double elapsedTime = elapsed.count();
    double iters_per_sec = total_iters / elapsedTime;
    std::cout << "Kernel execution time: " << elapsedTime << " seconds" << std::endl;
    std::cout << "Iterations per second: " << iters_per_sec 
              << " (" << total_iters << " iterations in total)" << std::endl;

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
    clReleaseMemObject(buf_block_carry_out);
    clReleaseKernel(k_precomp);
    clReleaseKernel(k_postcomp);
    clReleaseKernel(k_forward_ntt);
    clReleaseKernel(k_inverse_ntt);
    clReleaseKernel(k_forward_ntt_alt);
    clReleaseKernel(k_inverse_ntt_alt);
    clReleaseKernel(k_forward_ntt_last);
    clReleaseKernel(k_forward_ntt_alt_last);
    clReleaseKernel(k_square);
    clReleaseKernel(k_sub2);
    clReleaseKernel(k_carry);
    clReleaseKernel(k_carry_2);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
