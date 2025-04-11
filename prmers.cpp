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
#include <cstdlib>
#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#endif
#include <regex>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <filesystem>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
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
#include <set>
#include <map>
#include <curl/curl.h>
#ifndef _WIN32
#include <unistd.h>
#include <limits.h>
#endif
#include "proof/proof.h"
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

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



//------------------------------------------------------------------------------
// Helper function to escape a string for JSON output.
// This function adds backslashes before quotes and returns the string enclosed in quotes.
//------------------------------------------------------------------------------
std::string jsonEscape(const std::string &s) {
    std::string escaped;
    for (char c : s) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    return "\"" + escaped + "\"";
}

//------------------------------------------------------------------------------
// Function: generatePrimeNetJson
// Purpose:  Generates a JSON string with all required fields to be sent to PrimeNet.
// Parameters:
//   status          - Status string (e.g., "P" for prime, "C" for composite)
//   exponent        - Exponent being tested (e.g., 197487599)
//   worktype        - Work type string (e.g., "PRP-3")
//   res64           - A hexadecimal string result (64-bit result)
//   res2048         - A long hexadecimal string representing a 2048-bit result
//   residueType     - An integer code describing the residue type (typically 1)
//   gerbiczError    - Error count associated with the Gerbicz test (e.g., 0)
//   fftLength       - FFT length used (e.g., 11534336)
//   proofVersion    - Version number of the proof (e.g., 1)
//   proofPower      - Proof power value (e.g., 10)
//   proofHashSize   - Hash size used in the proof (e.g., 64)
//   proofMd5        - MD5 hash as a string (e.g., "cbb1774df764197be28c4bcb659086fd")
//   programName     - Name of the program (e.g., "prpll")
//   programVersion  - Version string of the program (e.g., "0.15-125-ga1349df-dirty")
//   programPort     - Platform port number (e.g., 8)
//   osName          - Operating system name (e.g., "Linux")
//   osVersion       - OS version string (e.g., "6.8.0-52-generic")
//   osArchitecture  - CPU architecture (e.g., "x86_64")
//   user            - Username (e.g., "cherubrock")
//   aid             - Application ID (e.g., "921AC7D5516755497E546BC5BEFB095C")
//   uid             - Unique ID (e.g., "a25018e172fd5d75")
//   timestamp       - Timestamp string (e.g., "2025-04-10 09:41:06")
// Returns: a std::string containing the JSON result.
//------------------------------------------------------------------------------
std::string generatePrimeNetJson(
    const std::string &status,
    unsigned int exponent,
    const std::string &worktype,
    const std::string &res64,
    const std::string &res2048,
    int residueType,
    int gerbiczError,
    unsigned int fftLength,
    int proofVersion,
    int proofPower,
    int proofHashSize,
    const std::string &proofMd5,
    const std::string &programName,
    const std::string &programVersion,
    unsigned int programPort,
    const std::string &osName,
    const std::string &osVersion,
    const std::string &osArchitecture,
    const std::string &user,
    const std::string &aid,
    const std::string &uid,
    const std::string &timestamp)
{
    std::ostringstream oss;
    oss << "{"; // Start of JSON object

    // Basic result fields.
    oss << "\"status\":" << jsonEscape(status) << ", ";
    oss << "\"exponent\":" << exponent << ", ";
    oss << "\"worktype\":" << jsonEscape(worktype) << ", ";
    oss << "\"res64\":" << jsonEscape(res64) << ", ";
    oss << "\"res2048\":" << jsonEscape(res2048) << ", ";
    oss << "\"residue-type\":" << residueType << ", ";
    
    // Errors object (for example, Gerbicz error count).
    oss << "\"errors\": {\"gerbicz\":" << gerbiczError << "}, ";
    
    oss << "\"fft-length\":" << fftLength << ", ";
    
    // Proof information object.
    oss << "\"proof\": {";
    oss << "\"version\":" << proofVersion << ", ";
    oss << "\"power\":" << proofPower << ", ";
    oss << "\"hashsize\":" << proofHashSize << ", ";
    oss << "\"md5\":" << jsonEscape(proofMd5);
    oss << "}, ";
    
    // Program information including OS details.
    oss << "\"program\": {";
    oss << "\"name\":" << jsonEscape(programName) << ", ";
    oss << "\"version\":" << jsonEscape(programVersion) << ", ";
    oss << "\"port\":" << programPort << ", ";
    oss << "\"os\": {";
    oss << "\"os\":" << jsonEscape(osName) << ", ";
    oss << "\"version\":" << jsonEscape(osVersion) << ", ";
    oss << "\"architecture\":" << jsonEscape(osArchitecture);
    oss << "}";
    oss << "}, ";
    
    // User information and timestamp.
    oss << "\"user\":" << jsonEscape(user) << ", ";
    oss << "\"aid\":" << jsonEscape(aid) << ", ";
    oss << "\"uid\":" << jsonEscape(uid) << ", ";
    oss << "\"timestamp\":" << jsonEscape(timestamp);
    
    oss << "}"; // End of JSON object

    return oss.str();
}



// -----------------------------------------------------------------------------
// Compute transform size for the given exponent (forcing power-of-4)
// -----------------------------------------------------------------------------
static cl_uint transformsize(uint32_t exponent) {
    /*int log_n = 0, w = 0;
    do {
        ++log_n;
        w = exponent / (1 << log_n);
    } while (((w + 1) * 2 + log_n) >= 63);
    if (log_n & 1)
        ++log_n;
    return (cl_uint)(1ULL << log_n);*/
        int log_n = 0; uint32_t w = 0;
		do
		{
			++log_n;
			w = exponent >> log_n;
		} while ((w + 1) * 2 + log_n >= 63);

		return cl_uint(1) << log_n;
}


std::string getExecutableDir() {
    char buffer[1024];

#ifdef __APPLE__
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) != 0)
        throw std::runtime_error("❌ Cannot get executable path (macOS).");

#elif defined(_WIN32)
    if (!GetModuleFileNameA(NULL, buffer, sizeof(buffer)))
        throw std::runtime_error("❌ Cannot get executable path (Windows).");

#else
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer)-1);
    if (len == -1)
        throw std::runtime_error("❌ Cannot get executable path (Linux).");
    buffer[len] = '\0';
#endif

    std::string fullPath(buffer);
    return fullPath.substr(0, fullPath.find_last_of("/\\"));
}

// -----------------------------------------------------------------------------
// Precompute digit weights, inverse weights, and digit widths for a given p.
// -----------------------------------------------------------------------------
void precalc_for_p(uint32_t p,
                   std::vector<uint64_t>& digit_weight,
                   std::vector<uint64_t>& digit_invweight,
                   std::vector<int>& digit_width) {
    cl_uint n = transformsize(p);
    if(n<4){
        n=4;
    }
    digit_weight.resize(n);
    digit_invweight.resize(n);
    digit_width.resize(n);
    #ifdef _MSC_VER
        uint64_t high = 0, low = MOD_P - 1ULL;
        uint64_t tmp1 = _udiv128(high, low, 192ULL, &low);
        uint64_t tmp2 = tmp1 / n;
        uint64_t exponent = tmp2 * 5ULL;
    #else
        __uint128_t bigPminus1 = (__uint128_t)MOD_P - 1ULL;
        __uint128_t tmp = bigPminus1 / 192ULL;
        tmp /= n;
        uint64_t exponent = (uint64_t)(tmp * 5ULL);
    #endif

    uint64_t nr2 = powModP(7ULL, exponent);
    uint64_t inv_n = invModP((uint64_t)n);
    uint32_t w_val = p / (uint32_t)n;
    digit_weight[0] = 1ULL;
    digit_invweight[0] = inv_n;
    uint32_t o = 0;
    for (cl_uint j = 0; j <= n; j++) {
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
    std::cout << "Usage: " << progName << " <p> [-d <device_id>] [-O <options>] [-c <localCarryPropagationDepth>]" << std::endl;
    std::cout << "              [-profile] [-prp|-ll] [-t <backup_interval>] [-f <path>]" << std::endl;
    std::cout << "              [-l1 <value>] [-l2 <value>] [-l3 <value>] [--noask] [-user <username>]" << std::endl;
    std::cout << "              [-worktodo <path>]" << std::endl;
    std::cout << std::endl;
    std::cout << "  <p>       : Exponent to test (required unless -worktodo is used)" << std::endl;
    std::cout << "  -d <device_id>   : (Optional) Specify OpenCL device ID (default: 0)" << std::endl;
    std::cout << "  -O <options>     : (Optional) Enable OpenCL optimization flags (e.g., fastmath, mad, unsafe, nans, optdisable)" << std::endl;
    std::cout << "  -c <depth>       : (Optional) Set local carry propagation depth (default: 8)" << std::endl;
    std::cout << "  -profile         : (Optional) Enable kernel execution profiling" << std::endl;
    std::cout << "  -prp             : (Optional) Run in PRP mode (default). Uses initial value 3; final result must equal 9" << std::endl;
    std::cout << "  -ll              : (Optional) Run in Lucas-Lehmer mode. Uses initial value 4 and p-2 iterations" << std::endl;
    std::cout << "  -t <seconds>     : (Optional) Specify backup interval in seconds (default: 120)" << std::endl;
    std::cout << "  -f <path>        : (Optional) Specify path for saving/loading checkpoint files (default: current directory)" << std::endl;
    std::cout << "  -l1 <value>      : (Optional) Force local size for classic NTT kernel" << std::endl;
    std::cout << "  -l2 <value>      : (Optional) Force local size for 2-step radix-16 NTT kernel" << std::endl;
    std::cout << "  -l3 <value>      : (Optional) Force local size for mixed radix NTT kernel" << std::endl;
    std::cout << "  --noask          : (Optional) Automatically send results to PrimeNet without prompting" << std::endl;
    std::cout << "  -user <username> : (Optional) PrimeNet username to auto-fill during result submission" << std::endl;
    std::cout << "  -worktodo <path> : (Optional) Load exponent from specified worktodo.txt (default: ./worktodo.txt)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:\n  " << progName << " 127 -O fastmath mad -c 16 -profile -ll -t 120 -f /my/backup/path \\\n"
              << "            -l1 256 -l2 128 -l3 64 --noask -user myaccountname -worktodo ./mydir/worktodo.txt" << std::endl;
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
                             const std::string& kernelName, cl_uint nmax,
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
        for (cl_uint i = 0; i < 10; i++) std::cout << locx[i] << " ";
        std::cout << "...]" << std::endl;
    }
}

void displayProgress(uint32_t iter, uint32_t total_iters, double elapsedTime, uint32_t expo) {
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
            << "Exponent: " << expo << " | "
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
                             cl_command_queue queue,int32_t expo) {
    auto duration = duration_cast<seconds>(high_resolution_clock::now() - lastDisplay).count();
    if (duration >= 10 || iter == -1) {
        if (iter == -1)
            iter = total_iters;
        double elapsedTime = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9;
        displayProgress(iter, total_iters, elapsedTime, expo);
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
cl_mem createBuffer(cl_context context, cl_mem_flags flags, cl_uint size, void* host_ptr, const std::string& name) {
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
    cl_mem buf_x, cl_mem buf_w,cl_mem buf_digit_weight, cl_uint n,
    size_t workers, size_t localSize,size_t localSize2,size_t localSize3, bool profiling,
    cl_uint maxLocalMem, bool _even_exponent, cl_kernel kernel_radix4_radix2_square_radix2_radix4, cl_mem buf_wi) {
    if(n==4){
        cl_uint m = 1;
        clSetKernelArg(kernel_ntt_last_m1_n4, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_last_m1_n4, 1, sizeof(cl_mem), &buf_w);
        clSetKernelArg(kernel_ntt_last_m1_n4, 2, sizeof(cl_mem), &buf_digit_weight);
        executeKernelAndDisplay(queue, kernel_ntt_last_m1_n4, buf_x, workers, localSize,
            "kernel_ntt_last_m1_n4 (m=" + std::to_string(m) + ")", n, profiling);
    }
    else{
        cl_uint m = n / 4;

        clSetKernelArg(kernel_ntt_mm_first, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_mm_first, 1, sizeof(cl_mem), &buf_w);
        clSetKernelArg(kernel_ntt_mm_first, 2, sizeof(cl_mem), &buf_digit_weight);
        clSetKernelArg(kernel_ntt_mm_first, 3, sizeof(cl_uint), &m);
        executeKernelAndDisplay(queue, kernel_ntt_mm_first, buf_x, workers, localSize,
            "kernel_ntt_radix4_mm_first (m=" + std::to_string(m) + ")", n, profiling);
        
        clSetKernelArg(kernel_ntt_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt_mm_2_steps, 1, sizeof(cl_mem), &buf_w);
        cl_uint mm = n/4;

        for (cl_uint m = n / 16; m >= 32; m /= 16) {
            
            clSetKernelArg(kernel_ntt_mm_2_steps, 2, sizeof(cl_uint), &m);
            executeKernelAndDisplay(queue, kernel_ntt_mm_2_steps, buf_x, workers/4, localSize2,
                "kernel_ntt_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling);
            mm = m/4;
        }
       //if(mm==256){
        if(mm==256){
            mm = 64;
            m = 64;
            clSetKernelArg(kernel_ntt_mm_3_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm_3_steps, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm_3_steps, 2, sizeof(cl_uint), &mm);
            executeKernelAndDisplay(queue, kernel_ntt_mm_3_steps, buf_x, workers, localSize,
            "kernel_ntt_mm_3_steps (m=" + std::to_string(m) + ")", n, profiling);
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            executeKernelAndDisplay(queue, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling);
        } 
         if(mm==32){

            mm = 8;
            m = 8;
            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(cl_uint), &mm);
            executeKernelAndDisplay(queue, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling);
            /*
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(cl_uint), &mm);
            clSetKernelArg(kernel_ntt_mm, 3, 0, NULL); 
            executeKernelAndDisplay(queue, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling);
            mm = 1;
            m = 1;
            clSetKernelArg(kernel_radix2_square_radix2, 0, sizeof(cl_mem), &buf_x);
            executeKernelAndDisplay(queue, kernel_radix2_square_radix2, buf_x, workers*2, localSize,
                    "kernel_radix2_square_radix2 (m=" + std::to_string(m) + ") workers=" + std::to_string(workers*2), n, profiling);
                    */
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 2, sizeof(cl_mem), &buf_wi);
            executeKernelAndDisplay(queue, kernel_radix4_radix2_square_radix2_radix4, buf_x, (workers*4)/8, localSize3,
                "kernel_radix4_radix2_square_radix2_radix4 (m=" + std::to_string(m) + ")", n, profiling);
        } 
        else if(mm==64){
            mm = 16;
            m = 16;
            clSetKernelArg(kernel_ntt_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm_2_steps, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm_2_steps, 2, sizeof(cl_uint), &m);
            executeKernelAndDisplay(queue, kernel_ntt_mm_2_steps, buf_x, workers/4, localSize2,
            "kernel_ntt_mm_3_steps (m=" + std::to_string(m) + ")", n, profiling);
            
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            executeKernelAndDisplay(queue, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling);
        }   
        else if(mm==8){
            mm = 2;
            m = 2;
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_radix4_radix2_square_radix2_radix4, 2, sizeof(cl_mem), &buf_wi);
            executeKernelAndDisplay(queue, kernel_radix4_radix2_square_radix2_radix4, buf_x, (workers*4)/8, localSize,
                "kernel_radix4_radix2_square_radix2_radix4 (m=" + std::to_string(m) + ")", n, profiling);
       }
       else if(mm==4){
            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
            executeKernelAndDisplay(queue, kernel_ntt_last_m1, buf_x, workers, localSize,
            "kernel_ntt_radix4_last_m1 (m=" + std::to_string(m) + ")", n, profiling);
       }   
       else if(mm==16){
            m = 4;
            mm=4;

            clSetKernelArg(kernel_ntt_mm, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_mm, 1, sizeof(cl_mem), &buf_w);
            clSetKernelArg(kernel_ntt_mm, 2, sizeof(cl_uint), &mm);
            executeKernelAndDisplay(queue, kernel_ntt_mm, buf_x, workers, localSize,
            "kernel_ntt_mm (m=" + std::to_string(m) + ")", n, profiling);


            clSetKernelArg(kernel_ntt_last_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_last_m1, 1, sizeof(cl_mem), &buf_w);
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
#ifdef _WIN32
#include <windows.h>
#endif

bool isLaunchedFromTerminal() {
#ifdef _WIN32
    return GetConsoleWindow() != nullptr;
#else
    return isatty(fileno(stdin)); // POSIX: Linux/macOS
#endif
}

int askExponentInteractively() {
#ifdef _WIN32
    char buffer[32];
    MessageBoxA(
        nullptr,
        "PrMers: GPU-accelerated Mersenne primality tester\n\n"
        "You'll now be asked which exponent you'd like to test.",
        "PrMers - Select Exponent",
        MB_OK | MB_ICONINFORMATION
    );
    std::cout << "Enter the exponent to test (e.g. 21701): ";
    std::cin.getline(buffer, sizeof(buffer));
    return std::atoi(buffer);
#else
    std::cout << "============================================\n";
    std::cout << " PrMers: GPU-accelerated Mersenne primality test\n";
    std::cout << " Powered by OpenCL | NTT | LL | PRP | IBDWT\n";
    std::cout << "============================================\n\n";

    std::string input;
    std::cout << "Enter the exponent to test (e.g. 21701): ";
    std::getline(std::cin, input);
    try {
        return std::stoi(input);
    } catch (...) {
        std::cerr << "Invalid input. Aborting." << std::endl;
        std::exit(1);
    }
#endif
}


void executeFusionneNTT_Inverse(cl_command_queue queue,cl_kernel kernel_ntt_inverse_mm_2_steps,
    cl_kernel kernel_inverse_ntt_mm, cl_kernel kernel_inverse_ntt_mm_last, cl_kernel kernel_inverse_ntt_m1,
    cl_kernel kernel_inverse_ntt_m1_n4,cl_mem buf_x, cl_mem buf_wi, cl_mem buf_digit_invweight, cl_uint n,
    size_t workers, size_t localSize,size_t localSize2, bool profiling,
    cl_uint maxLocalMem, bool _even_exponent) {
    cl_uint m = 0;
    if(n==4){
        m = 1;
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 1, sizeof(cl_mem), &buf_wi);
        clSetKernelArg(kernel_inverse_ntt_m1_n4, 2, sizeof(cl_mem), &buf_digit_invweight);
        executeKernelAndDisplay(queue, kernel_inverse_ntt_m1_n4, buf_x, workers, localSize,
            "kernel_inverse_ntt_m1_n4 (m=" + std::to_string(m) + ")", n, profiling);
    }
    else{
       
        if(_even_exponent){

            
            m = 1;
            clSetKernelArg(kernel_inverse_ntt_m1, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_inverse_ntt_m1, 1, sizeof(cl_mem), &buf_wi);
            executeKernelAndDisplay(queue, kernel_inverse_ntt_m1, buf_x, workers, localSize,
                "kernel_inverse_ntt_radix4_m1 (m=" + std::to_string(m) + ")", n, profiling);


            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 1, sizeof(cl_mem), &buf_wi);
            cl_uint mm = 4;
          
            for (cl_uint m = 4; m < n/16; m *= 16) {
                
                clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 2, sizeof(cl_uint), &m);
                executeKernelAndDisplay(queue, kernel_ntt_inverse_mm_2_steps, buf_x, workers/4, localSize2,
                    "kernel_ntt_inverse_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling);
                mm = m*16;
            }
            if(mm<=n/16 && n>8){
                clSetKernelArg(kernel_inverse_ntt_mm, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_inverse_ntt_mm, 1, sizeof(cl_mem), &buf_wi);
                clSetKernelArg(kernel_inverse_ntt_mm, 2, sizeof(cl_uint), &mm);
                executeKernelAndDisplay(queue, kernel_inverse_ntt_mm, buf_x, workers, localSize,
                        "kernel_inverse_ntt_radix4_mm (m=" + std::to_string(m) + ")", n, profiling);
            }

        }
        else{
            m = 8;
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 0, sizeof(cl_mem), &buf_x);
            clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 1, sizeof(cl_mem), &buf_wi);
            cl_uint mm = 8;
            for (cl_uint m = 8; m < n/16; m *= 16) {
                
                clSetKernelArg(kernel_ntt_inverse_mm_2_steps, 2, sizeof(cl_uint), &m);
                executeKernelAndDisplay(queue, kernel_ntt_inverse_mm_2_steps, buf_x, workers/4, localSize2,
                    "kernel_ntt_inverse_mm_2_steps (m=" + std::to_string(m) + ")", n, profiling);
                mm = m*16;
            }
            if(mm<=n/16 && n>8){
                clSetKernelArg(kernel_inverse_ntt_mm, 0, sizeof(cl_mem), &buf_x);
                clSetKernelArg(kernel_inverse_ntt_mm, 1, sizeof(cl_mem), &buf_wi);
                clSetKernelArg(kernel_inverse_ntt_mm, 2, sizeof(cl_uint), &mm);
                executeKernelAndDisplay(queue, kernel_inverse_ntt_mm, buf_x, workers, localSize,
                        "kernel_inverse_ntt_radix4_mm (m=" + std::to_string(m) + ")", n, profiling);
            }
        }
        


        m = n/4;
        clSetKernelArg(kernel_inverse_ntt_mm_last, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 1, sizeof(cl_mem), &buf_wi);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 2, sizeof(cl_mem), &buf_digit_invweight);
        clSetKernelArg(kernel_inverse_ntt_mm_last, 3, sizeof(cl_uint), &m);
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

void handleFinalCarry(std::vector<uint64_t>& x, const std::vector<int>& digit_width_cpu, cl_uint n) {
    x[0] += 1;
    uint64_t c = 0;
    
    for (cl_uint k = 0; k < n; ++k) {
        x[k] = digit_adc(x[k], digit_width_cpu[k], c);
    }

    while (c != 0) {
        for (cl_uint k = 0; k < n; ++k) {
            x[k] = digit_adc(x[k], digit_width_cpu[k], c);
            if (c == 0) break;
        }
    }

    x[0] -= 1;
}


#include <string>

std::string promptHiddenPassword() {
    std::string pwd;
#if defined(_WIN32)
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE); DWORD mode; GetConsoleMode(hStdin, &mode);
    SetConsoleMode(hStdin, mode & ~(ENABLE_ECHO_INPUT));
    std::cout << "Enter your PrimeNet password: ";
    std::getline(std::cin, pwd);
    SetConsoleMode(hStdin, mode); std::cout << std::endl;
#else
    std::cout << "Enter your PrimeNet password: ";
    int ret1 = system("stty -echo");
    (void)ret1; // ignore warning
    std::getline(std::cin, pwd);
    int ret2 = system("stty echo");
    (void)ret2; // ignore warning

#endif
    return pwd;
}


namespace fs = std::filesystem;

// Helper to rename file after successful upload
void markJsonAsSent(const std::string& path) {
    std::string newPath = path + ".sent";
    try {
        fs::rename(path, newPath);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Couldn't rename file to mark as sent: " << e.what() << std::endl;
    }
}

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

std::string extractUID(const std::string& html) {
    std::string key = "name=\"was_logged_in_as\" value=\"";
    auto pos = html.find(key);
    if (pos == std::string::npos) return "";
    pos += key.length();
    auto end = html.find("\"", pos);
    if (end == std::string::npos) return "";
    return html.substr(pos, end - pos);
}

bool sendManualResultWithLogin(const std::string& jsonResult, const std::string& username, const std::string& password) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "❌ Failed to initialize libcurl.\n";
        return false;
    }

    FILE* trace = fopen("curl_trace.txt", "w");
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_STDERR, trace);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_COOKIEFILE, "cookies.txt");
    curl_easy_setopt(curl, CURLOPT_COOKIEJAR, "cookies.txt");
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");

    std::string loginResponse;
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.mersenne.org/");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &loginResponse);

    std::ostringstream loginData;
    char* escapedUser = curl_easy_escape(curl, username.c_str(), 0);
    char* escapedPass = curl_easy_escape(curl, password.c_str(), 0);
    loginData << "user_login=" << escapedUser << "&user_password=" << escapedPass;
    curl_free(escapedUser);
    curl_free(escapedPass);

    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, loginData.str().c_str());

    std::cerr << "[TRACE] Sending login with user: " << username << std::endl;
    CURLcode loginRes = curl_easy_perform(curl);
    std::cerr << "[TRACE] Login response size: " << loginResponse.size() << " bytes\n";

    if (loginRes != CURLE_OK || loginResponse.find("logged in") == std::string::npos) {
        std::cerr << "❌ Login failed or session not recognized.\n";
        curl_easy_cleanup(curl);
        fclose(trace);
        return false;
    }

    std::string htmlFormPage;
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.mersenne.org/manual_result/");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, nullptr);     // clear POST data
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);              // force GET
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &htmlFormPage);


    CURLcode pageRes = curl_easy_perform(curl);
    if (pageRes != CURLE_OK) {
        std::cerr << "❌ Failed to get manual_result page: " << curl_easy_strerror(pageRes) << "\n";
        curl_easy_cleanup(curl);
        fclose(trace);
        return false;
    }

    std::string uid = extractUID(htmlFormPage);
    std::cerr << "[TRACE] Extracted was_logged_in_as UID: " << uid << "\n";
    if (uid.empty()) {
        std::cerr << "❌ Could not find was_logged_in_as value in form page.\n";
        curl_easy_cleanup(curl);
        fclose(trace);
        return false;
    }

    curl_mime* form = curl_mime_init(curl);

    curl_mimepart* field = curl_mime_addpart(form);
    curl_mime_name(field, "data_file");
    curl_mime_data(field, "", CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "was_logged_in_as");
    curl_mime_data(field, uid.c_str(), CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "data");
    curl_mime_data(field, jsonResult.c_str(), CURL_ZERO_TERMINATED);

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.mersenne.org/manual_result/");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, nullptr);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    std::cerr << "[TRACE] Sending manual result with was_logged_in_as = " << uid << std::endl;

    CURLcode res = curl_easy_perform(curl);
    fflush(trace);
    fclose(trace);
    curl_mime_free(form);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "❌ Failed to send result: " << curl_easy_strerror(res) << "\n";
        return false;
    }

    std::cerr << "[TRACE] Server response size: " << response.size() << " bytes\n";
    std::cout << "✅ Server response:...\n";
    std::string startTag = "<h2>Manually check in your results</h2>";
    std::string endTagA = "<a href=\"/manual_result/\">Submit more results</a>";
    std::string endTagB = "Aborting processing.</div>";

    auto startPos = response.find(startTag);
    size_t endPos = std::string::npos;
    std::string chosenEndTag;
    bool usedFallback = false;

    if (startPos != std::string::npos) {
        endPos = response.find(endTagA, startPos);
        chosenEndTag = endTagA;

        if (endPos == std::string::npos) {
            endPos = response.find(endTagB, startPos);
            chosenEndTag = endTagB;
        }

        if (endPos == std::string::npos && response.size() > startPos + 1000) {
            endPos = startPos + 1000;
            usedFallback = true;
        }

        if (endPos != std::string::npos) {
            std::string htmlChunk = response.substr(startPos, endPos - startPos + (usedFallback ? 0 : chosenEndTag.length()));

            std::string readable;
            bool insideTag = false;
            for (char c : htmlChunk) {
                if (c == '<') {
                    insideTag = true;
                    continue;
                }
                if (c == '>') {
                    insideTag = false;
                    readable += ' ';
                    continue;
                }
                if (!insideTag) readable += c;
            }

            std::regex spaceRegex("\\s+");
            readable = std::regex_replace(readable, spaceRegex, " ");

            std::cout << "\n📝 Parsed PrimeNet Result Summary:\n" << readable << "\n";
        } else {
            std::cout << "⚠️ Could not find an end marker. Raw response:\n\n" << response << "\n";
        }
    } else {
        std::cout << "⚠️ Could not find start of results section. Raw response:\n\n" << response << "\n";
    }

    return true;
}



void promptToSendResult(const std::string& jsonPath, std::string& user) {
    std::string response;
    std::cout << "\n✅ JSON result written to: " << jsonPath << std::endl;
    std::cout << "Do you want to send the result to PrimeNet (https://www.mersenne.org) ? (y/n): ";
    std::getline(std::cin, response);
    if (response.empty() || (response[0] != 'y' && response[0] != 'Y')) {
        std::cout << "Result not sent." << std::endl;
        return;
    }

    if (user.empty()) {
        std::cout << "Enter your PrimeNet username (Don't have an account? Create one at https://www.mersenne.org)\nEnter your PrimeNet username : ";
        std::getline(std::cin, user);
    }

    std::string password = promptHiddenPassword();

    std::ifstream fin(jsonPath);
    if (!fin) {
        std::cerr << "Cannot open file: " << jsonPath << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << fin.rdbuf();
    std::string jsonContent = buffer.str();
    fin.close();

    bool success = sendManualResultWithLogin(jsonContent, user, password);

    while (!success) {
        std::cout << "❌ Failed to send result to PrimeNet." << std::endl;
        std::cout << "Do you want to retry? (y/n): ";
        std::getline(std::cin, response);
        if (response.empty() || (response[0] != 'y' && response[0] != 'Y')) {
            std::cout << "Result not sent." << std::endl;
            break;
        }

        std::cout << "Re-enter your PrimeNet username (leave empty to reuse '" << user << "'): ";
        std::string newUser;
        std::getline(std::cin, newUser);
        if (!newUser.empty()) user = newUser;
        password = promptHiddenPassword();

        std::ifstream fin2(jsonPath);
        std::stringstream buffer2;
        buffer2 << fin2.rdbuf();
        jsonContent = buffer2.str();
        fin2.close();

        success = sendManualResultWithLogin(jsonContent, user, password);
    }

    if (success) {
        std::cout << "✅ Result successfully sent to PrimeNet." << std::endl;
        std::ofstream out(jsonPath + ".sent");
        out.put('\n');
        out.close();
    }

    fs::path dir = fs::path(jsonPath).parent_path();
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            auto p = entry.path();
            std::string filename = p.filename().string();
            if (filename.size() >= 12 && filename.compare(filename.size() - 12, 12, "_result.json") == 0
                && !fs::exists(p.string() + ".sent")) {
                if (p == jsonPath) continue;
                promptToSendResult(p.string(), user);
            }
        }
    }
}

int extractExponentFromWorktodo(const std::string& path = "worktodo.txt") {
    std::ifstream infile(path);
    if (!infile) return -1;

    std::string line;
    std::regex prp_pattern(R"(PRP=.*?,.*?,.*?,(\d+),.*)");
    std::smatch match;
    while (std::getline(infile, line)) {
        if (std::regex_match(line, match, prp_pattern)) {
            return std::stoi(match[1].str());
        }
    }
    return -1;
}

// -----------------------------------------------------------------------------
// Main function
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string exponentStr;
    std::string worktodo_path = "worktodo.txt";
    std::vector<const char*> new_argv;
    bool has_p = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-h" || arg == "--h" || arg == "-help" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "-worktodo") == 0) {
            worktodo_path = argv[i + 1];
            break;
        }
    }

    if (argc < 2 || (argc == 2 && argv[1][0] == '-')) {
        int exp = extractExponentFromWorktodo(worktodo_path);
        if (exp > 0) {
            exponentStr = std::to_string(exp);
            new_argv.push_back(argv[0]);
            new_argv.push_back(exponentStr.c_str());
            for (int i = 1; i < argc; ++i) new_argv.push_back(argv[i]);
            argc = static_cast<int>(new_argv.size());
            argv = const_cast<char**>(new_argv.data());
        } else {
            int exp = askExponentInteractively();
            exponentStr = std::to_string(exp);
            new_argv.push_back(argv[0]);
            new_argv.push_back(exponentStr.c_str());
            argc = 2;
            argv = const_cast<char**>(new_argv.data());
        }
    }

    uint32_t pp = static_cast<uint32_t>(std::stoi(argv[1]));
    std::cout << "🧮 Testing exponent: " << pp << std::endl;
    if (argc < 2) {
        std::cerr << "Error: Missing <p_min> argument.\n";
        printUsage(argv[0]);
        return 1;
    }
    uint32_t p = 0;
    int device_id = 0;  // Default device ID
    cl_uint localCarryPropagationDepth = 4;
    std::string mode = "prp"; 
    bool proof = false;
    bool profiling = false;
    bool force_carry = false;
    int max_local_size1 = 0;
    int max_local_size2 = 0;
    int max_local_size3 = 0;
    bool noAsk = false;

    std::string user;
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
        else if (std::strcmp(argv[i], "-d") == 0) {
            if (i + 1 < argc) {
                device_id = std::atoi(argv[i + 1]);
                i++;
            } else {
                std::cerr << "Error: Missing value for -d <device_id>." << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--noask") == 0 || std::strcmp(argv[i], "-noask") == 0) {
            noAsk = true;
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
        else if (std::strcmp(argv[i], "-l1") == 0) {
            if (i + 1 < argc) {
                max_local_size1 = std::atoi(argv[i + 1]);
                i++;
            } else {
                std::cerr << "Error: Missing value for -l1 <max_local_size1>." << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "-l2") == 0) {
            if (i + 1 < argc) {
                max_local_size2 = std::atoi(argv[i + 1]);
                i++;
            } else {
                std::cerr << "Error: Missing value for -l2 <max_local_size2>." << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "-l3") == 0) {
            if (i + 1 < argc) {
                max_local_size3 = std::atoi(argv[i + 1]);
                i++;
            } else {
                std::cerr << "Error: Missing value for -l3 <max_local_size3>." << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "-user") == 0) {
            if (i + 1 < argc) {
                user = argv[i + 1];
                i++;
            } else {
                std::cerr << "Error: Missing value for -user <name>." << std::endl;
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
    if(mode == "ll"){
        proof = false;
        noAsk = true;
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

    char version_str[128];
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version_str), version_str, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL device version." << std::endl;
        return 1;
    }

    unsigned int major = 1, minor = 2;  // fallback au cas où
    sscanf(version_str, "OpenCL %u.%u", &major, &minor);
    std::cout << "OpenCL device version detected: " << major << "." << minor << std::endl;


    cl_command_queue queue;


    #ifdef __APPLE__
        queue = clCreateCommandQueue(context, device, 0, &err);
    #else
        if (major >= 2) {
            queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        } else {
            queue = clCreateCommandQueue(context, device, 0, &err);
        }
    #endif

    
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        return 1;
    }



    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        return 1;
    }

    // -------------------------------------------------------------------------
    // Read and Build OpenCL Program
    // -------------------------------------------------------------------------
    std::string kernelSource;
    try {
        std::string execDir = getExecutableDir();

        std::string kernelFile = execDir + "/prmers.cl";
        if (!std::filesystem::exists(kernelFile)) {
            kernelFile = execDir + "/kernels/prmers.cl";
            if (!std::filesystem::exists(kernelFile)) {
                throw std::runtime_error("Kernel file 'prmers.cl' not found in current or kernels/ directory.");
            }
        }

        kernelSource = readFile(kernelFile);
    } catch (const std::exception &e) {
        std::cerr << "❌ Error loading OpenCL kernel: " << e.what() << std::endl;
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
   
    cl_uint n = transformsize(p);
    if(n<4){
        n=4;
    }
    cl_uint mm=n/4;
    for (cl_uint m = n / 16; m >= 32; m /= 16) {
        
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
        for (cl_uint m = n / 2, s = 1; m >= 1; m /= 2, s *= 2) {
            uint64_t r_s = powModP(root, s);
            uint64_t invr_s = powModP(invroot, s);
            uint64_t w_m = 1ULL, invw_m = 1ULL;
            for (cl_uint j = 0; j < m; j++) {
                cl_uint idx = 3 * (m + j);
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
    cl_uint workitem_size[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, nullptr);
    cl_uint maxThreads = workitem_size[0];
    size_t maxWork;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWork, nullptr);
    std::cout << "Max CL_DEVICE_MAX_WORK_GROUP_SIZE = " << maxWork << std::endl;
    std::cout << "Max CL_DEVICE_MAX_WORK_ITEM_SIZES = " << maxThreads << std::endl;
    cl_uint maxLocalMem;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxLocalMem), &maxLocalMem, NULL);
    std::cout << "Max CL_DEVICE_LOCAL_MEM_SIZE = " << maxLocalMem << std::endl;
    
    size_t workers = static_cast<size_t>(n);;
    size_t workersNtt = static_cast<size_t>(n)/4;
    size_t workersNtt2step = static_cast<size_t>(n)/16;
    size_t maxWorkMax = 256;
    if (maxWork > maxWorkMax){
        maxWork = maxWorkMax;
    }
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
    
    cl_uint constraint = std::max(n / 16, (cl_uint)1);
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

    
    size_t localSize2 = localSize;
    size_t localSize3 = localSize ;  
    
    /*if (max_local_size1 == 0){
        max_local_size2 = (maxWork) <1 ? 1 : (maxWork) ;  
    }
    if (max_local_size2 == 0){
        max_local_size2 = (maxWork/4) <1 ? 1 : (maxWork/4) ; 
    }
    if (max_local_size3 == 0){
        max_local_size3 = (maxWork/2) <1 ? 1 : (maxWork/2) ; 
    }*/

    if (max_local_size1 > 0) {
        localSize = max_local_size1;
    }
    if (max_local_size2 > 0) {
        localSize2 = max_local_size2;
    }
    if (max_local_size3 > 0) {
        localSize3 = max_local_size3;
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

    cl_uint workGroupSize = ((workers < localSize) ? 1 : (workers / localSize));
    cl_uint adjustedDepth = localCarryPropagationDepth / 4;
    cl_uint adjustedDepthMin = (localCarryPropagationDepth - 4) / 4;
    cl_uint adjustedDepth2 = localCarryPropagationDepth / 4;
    cl_uint adjustedDepthMin2 = (localCarryPropagationDepth - 2) / 2;
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
    build_options += " -DWG_SIZE=" + std::to_string(workGroupSize) + " -DLOCAL_PROPAGATION_DEPTH=" + std::to_string(localCarryPropagationDepth) + " -DCARRY_WORKER=" + std::to_string(workersCarry) + " -DLOCAL_PROPAGATION_DEPTH_DIV4=" + std::to_string(adjustedDepth)+ " -DLOCAL_PROPAGATION_DEPTH_DIV4_MIN=" + std::to_string(adjustedDepthMin) + " -DLOCAL_PROPAGATION_DEPTH_DIV2=" + std::to_string(adjustedDepth2)+ " -DLOCAL_PROPAGATION_DEPTH_DIV2_MIN=" + std::to_string(adjustedDepthMin2) + " -DWORKER_NTT=" + std::to_string(workersNtt) + " -DWORKER_NTT_2_STEPS=" + std::to_string(workersNtt2step) + " -DMODULUS_P=" + std::to_string(p) + " -DTRANSFORM_SIZE_N=" + std::to_string(n) + " -DLOCAL_SIZE=" + std::to_string(localSize)+ " -DLOCAL_SIZE2=" + std::to_string(localSize2)+ " -DLOCAL_SIZE3=" + std::to_string(localSize3);
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
    cl_kernel k_ntt_radix4_radix2_square_radix2_radix4     = createKernel(program, "kernel_ntt_radix4_radix2_square_radix2_radix4");

    // --------------------
    // Set Kernel Arguments
    // --------------------
    cl_int errKernel = CL_SUCCESS;
    // kernel_sub2
    errKernel  = clSetKernelArg(k_sub2, 0, sizeof(cl_mem), &buf_x);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for kernel_sub2: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }
    // kernel_carry
    errKernel  = clSetKernelArg(k_carry, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_carry, 1, sizeof(cl_mem), &buf_block_carry);
    if (errKernel != CL_SUCCESS) {
        std::cerr << "Error setting arguments for carry: " << getCLErrorString(errKernel) << std::endl;
        exit(1);
    }
    // kernel_carry_2
    errKernel  = clSetKernelArg(k_carry_2, 0, sizeof(cl_mem), &buf_x);
    errKernel |= clSetKernelArg(k_carry_2, 1, sizeof(cl_mem), &buf_block_carry);
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
    checkAndDisplayProgress(0, total_iters, lastDisplay, startTime, queue, p);
    
    if(debug)
        std::cout << "Number of iterations to be done = " << total_iters << std::endl;
            

    // Main loop now starts from resume_iter (if any) to total_iters.
    for (uint32_t iter = resume_iter; iter < total_iters; iter++) {
        executeFusionneNTT_Forward(queue,k_forward_ntt_mm_3steps,k_forward_ntt_mm_2steps,k_ntt_radix2_square_radix2,
            k_forward_ntt_mm, k_forward_ntt_mm_first, k_forward_ntt_last_m1,k_forward_ntt_last_m1_n4,
            buf_x, buf_twiddles, buf_digit_weight, n, workers/4, localSize, localSize2, localSize3, profiling, maxLocalMem, _even_exponent, k_ntt_radix4_radix2_square_radix2_radix4, buf_inv_twiddles);

        executeFusionneNTT_Inverse(queue,k_inverse_ntt_mm_2steps,
            k_inverse_ntt_mm, k_inverse_ntt_mm_last, k_inverse_ntt_m1,k_inverse_ntt_m1_n4,
            buf_x, buf_inv_twiddles,buf_digit_invweight, n, workers/4, localSize, localSize2, profiling, maxLocalMem, _even_exponent);
        executeKernelAndDisplay(queue, k_carry, buf_x, workersCarry, localSizeCarry, "kernel_carry", n, profiling);
        executeKernelAndDisplay(queue, k_carry_2, buf_x, workersCarry, localSizeCarry, "kernel_carry_2", n, profiling);
        // In Lucas-Lehmer mode, execute kernel_sub2; in PRP mode, skip it.
        if (mode == "ll") {
            executeKernelAndDisplay(queue, k_sub2, buf_x, 1, 1, "kernel_sub2", n, profiling);
        }
        checkAndDisplayProgress(iter-resume_iter, total_iters, lastDisplay, startTime, queue, p);
        
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
                clReleaseMemObject(buf_block_carry);
                clReleaseKernel(k_forward_ntt_mm);
                clReleaseKernel(k_forward_ntt_mm_3steps);
                clReleaseKernel(k_forward_ntt_mm_2steps);
                clReleaseKernel(k_inverse_ntt_mm_2steps);
                clReleaseKernel(k_forward_ntt_mm_first);
                clReleaseKernel(k_ntt_radix2_square_radix2);
                clReleaseKernel(k_ntt_radix4_radix2_square_radix2_radix4);
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

    checkAndDisplayProgress(-1, total_iters, lastDisplay, startTime, queue, p);
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
    std::filesystem::path proofFile;  // Déclaré en dehors

    if (proof) {
        flush_log();
        Words finalRes = ProofSet::fromUint64(x, p);
        auto [myProof, rExps] = proofSet.computeProof(finalRes);
        std::filesystem::path outDir = save_path; 
        std::filesystem::create_directories(outDir);
        proofFile = myProof.fileName(outDir);  // Affectation ici
        myProof.save(proofFile);
        std::cout << "Proof is saved in a file!\n\n";
    }
    try {
        std::ostringstream res64oss;
        for (int i = 0; i < 64 / 64; ++i) {
            res64oss << std::hex << std::setw(16) << std::setfill('0') << x[i];
        }

        std::ostringstream res2048oss;
        for (int i = 0; i < std::min((int)n, 2048 / 64); ++i) {
            res2048oss << std::hex << std::setw(16) << std::setfill('0') << x[i];
        }

        std::time_t now = std::time(nullptr);
        char timestampBuf[32];
        std::strftime(timestampBuf, sizeof(timestampBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        std::string jsonResult = generatePrimeNetJson(
            (mode == "ll") ? (std::all_of(x.begin(), x.end(), [](uint64_t v) { return v == 0; }) ? "P" : "C")
                        : ((x[0] == 9 && std::all_of(x.begin() + 1, x.end(), [](uint64_t v) { return v == 0; })) ? "P" : "C"),
            p,
            (mode == "prp") ? "PRP-3" : "LL",
            res64oss.str(),
            res2048oss.str(),
            1, // residueType
            0, // gerbiczError
            n,
            proof ? 1 : 0,
            proof ? proofPower : 0,
            proof ? 64 : 0,
            proof ? proof_util::fileMD5(proofFile) : "",
            "prmers",
            "0.1",
            device_id,
    #ifdef _WIN32
            "Windows",
    #elif __APPLE__
            "macOS",
    #else
            "Linux",
    #endif
            "",
    #ifdef __x86_64__
            "x86_64",
    #elif __aarch64__
            "arm64",
    #else
            "unknown",
    #endif
            user.empty() ? "cherubrock":user,
            "AID-PRMERS-1234",
            "UID-PRMERS-5678",
            timestampBuf
        );

        std::ofstream jsonOut(save_path + "/" + std::to_string(p) + "_" + mode + "_result.json");
        jsonOut << jsonResult;
        jsonOut.close();
        std::cout << "\n✅ JSON result written to: " << save_path + "/" + std::to_string(p) + "_" + mode + "_result.json" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "❌ Error while generating the final JSON: " << e.what() << std::endl;
    }



    std::string resultLine;
    std::string jsonFile = save_path + "/" + std::to_string(p) + "_" + mode + "_result.json";

    if (mode == "prp") {
        bool resultIs9 = (x[0] == 9) && std::all_of(x.begin() + 1, x.end(), [](uint64_t v){ return v == 0; });
        std::cout << "\nM" << p << " PRP test " << (resultIs9 ? "succeeded (result is 9)." : "failed (result is not 9).") << std::endl;

        // Build the PrimeNet result line for PRP
        std::ostringstream oss;
        oss << "PRP=" << p << ","
            << (proof ? "1" : "0") << ","
            << user << ","
            << (resultIs9 ? "P" : "F") << ",0x"
            << std::hex << std::setw(16) << std::setfill('0') << x[0]; // res64
        resultLine = oss.str();

        
        if(!noAsk){
            promptToSendResult(jsonFile, user);
            std::cout << "\nPress Enter to exit...";
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return resultIs9 ? 0 : 1;
        }
    } else {
        bool isPrime = std::all_of(x.begin(), x.end(), [](uint64_t v) { return v == 0; });
        std::cout << "\nM" << p << " is " << (isPrime ? "prime!" : "composite.") << std::endl;

        // Build the PrimeNet result line for LL
        std::ostringstream oss;
        oss << "LL=" << p << ","
            << user << ","
            << (isPrime ? "P" : "F") << ",0x"
            << std::hex << std::setw(16) << std::setfill('0') << x[0]; // res64
        resultLine = oss.str();

        promptToSendResult(jsonFile, user);

        std::cout << "\nPress Enter to exit...";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
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
    clReleaseMemObject(buf_block_carry);
    clReleaseKernel(k_forward_ntt_mm);
    clReleaseKernel(k_forward_ntt_mm_3steps);
    clReleaseKernel(k_forward_ntt_mm_2steps);
    clReleaseKernel(k_inverse_ntt_mm_2steps);
    clReleaseKernel(k_ntt_radix2_square_radix2);
    clReleaseKernel(k_ntt_radix4_radix2_square_radix2_radix4);
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
