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


#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RED "\033[31m"
#define COLOR_RESET "\033[0m"


using namespace std::chrono;

// 1) Helpers for 64-bit arithmetic modulo p
// p = 2^64 - 2^32 + 1
static constexpr uint64_t MOD_P = (((1ULL << 32) - 1ULL) << 32) + 1ULL;
bool debug = false;

// Multiply a and b modulo p using __uint128_t arithmetic.
uint64_t mulModP(uint64_t a, uint64_t b)
{
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t lo = (uint64_t)prod;

    // Decompose hi into two 32-bit parts.
    uint32_t A = (uint32_t)(hi >> 32);
    uint32_t B = (uint32_t)(hi & 0xffffffffULL);

    // Compute r = lo + (B << 32) - (A + B), with overflow correction.
    uint64_t r = lo;
    uint64_t oldr = r;
    r += ((uint64_t)B << 32);
    if(r < oldr) {
        r += ((1ULL << 32) - 1ULL);
    }
    uint64_t sub = (uint64_t)A + (uint64_t)B;
    if(r >= sub)
        r -= sub;
    else
        r = r + MOD_P - sub;

    // Final reduction.
    if(r >= MOD_P)
        r -= MOD_P;
    return r;
}

// Modular exponentiation: computes base^exp mod p.
uint64_t powModP(uint64_t base, uint64_t exp)
{
    uint64_t result = 1ULL;
    while(exp > 0ULL) {
        if(exp & 1ULL)
            result = mulModP(result, base);
        base = mulModP(base, base);
        exp >>= 1ULL;
    }
    return result;
}

// Compute the modular inverse using Fermat's little theorem.
uint64_t invModP(uint64_t x)
{
    // x^(p-2) mod p
    return powModP(x, MOD_P - 2ULL);
}

// 2) Compute transform size for the given exponent.
// This function increases log_n until the condition ((w+1)*2 + log_n) < 63 is met,
// then returns n = 2^log_n.
static size_t transformsize(uint32_t exponent)
{
    int log_n = 0, w = 0;
    do {
        ++log_n;
        w = exponent / (1 << log_n);
    } while (((w + 1) * 2 + log_n) >= 63);
    return (size_t)(1ULL << log_n);
}

// 3) Precompute digit weights, inverse weights, and digit widths for a given p.
void precalc_for_p(uint32_t p,
                   std::vector<uint64_t>& digit_weight,
                   std::vector<uint64_t>& digit_invweight,
                   std::vector<int>& digit_width)
{
    // Compute transform size n.
    size_t n = transformsize(p);
    digit_weight.resize(n);
    digit_invweight.resize(n);
    digit_width.resize(n);

    // Calculate exponent for nr2: exponent = (((p-1)/192)/n) * 5.
    __uint128_t bigPminus1 = ((__uint128_t)MOD_P) - 1ULL;
    __uint128_t tmp = bigPminus1 / 192ULL;
    tmp /= n;
    tmp *= 5ULL;
    uint64_t exponent = (uint64_t)tmp;  // Assumed to be < 2^64.

    // Compute nr2 = 7^exponent mod p.
    uint64_t nr2 = powModP(7ULL, exponent);
    // Compute the modular inverse of n.
    uint64_t inv_n = invModP((uint64_t)n);
    // Let w_val = floor(p / n).
    uint32_t w_val = p / (uint32_t)n;

    // Initialize weight values.
    digit_weight[0] = 1ULL;
    digit_invweight[0] = inv_n;

    uint32_t o = 0;  // Offset, updated at each iteration.
    for (size_t j = 0; j <= n; j++) {
        uint64_t qj = (uint64_t)p * j;
        // Compute ceil(qj/n) as floor((qj - 1)/n) + 1.
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

std::string readFile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Unable to open file: " + filename);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <p_min> [-d <device_id>]" << std::endl;
    std::cout << "  <p_min>       : Minimum exponent to test (required)" << std::endl;
    std::cout << "  -d <device_id>: (Optional) Specify OpenCL device ID (default: 0)" << std::endl;
    std::cout << "  -h            : Display this help message" << std::endl;
}

void executeKernelAndDisplay(cl_command_queue queue, cl_kernel kernel, 
                             cl_mem buf_x, std::vector<uint64_t>& x, size_t workers,  size_t localSize,
                             const std::string& kernelName, size_t nmax) 
{

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &workers, &localSize, 0, nullptr, nullptr);

    if (debug) {
        
        clFinish(queue);  // Assure la complétion du kernel
        clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, workers * sizeof(uint64_t), x.data(), 0, nullptr, nullptr);

        // Affichage limité à nmax éléments
        std::cout << "After " << kernelName << " : x = [ ";
        for (size_t i = 0; i < nmax; i++) std::cout << x[i] << " ";
        std::cout << "]" << std::endl;
    }
}


// Display progress function
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

    std::cout << "\r" << color  
              << "Progress: " << std::fixed << std::setprecision(2) << progress << "% | "
              << "Elapsed: " << elapsedTime << "s | "
              << "Iterations/sec: " << iters_per_sec << " | "
              << "ETA: " << (remaining_time > 0 ? remaining_time : 0.0) << "s       "
              << COLOR_RESET  
              << std::flush;
}

// Function that checks progress and updates display every 2 seconds
void checkAndDisplayProgress(uint32_t iter, uint32_t total_iters, 
                             time_point<high_resolution_clock>& lastDisplay, 
                             const time_point<high_resolution_clock>& start, 
                             cl_command_queue queue) {
    auto duration = duration_cast<seconds>(high_resolution_clock::now() - lastDisplay).count();

    if (duration >= 2) {  
        double elapsedTime = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9;
        displayProgress(iter, total_iters, elapsedTime);
        lastDisplay = high_resolution_clock::now();
    }
    clFinish(queue);
}




int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Error: Missing <p_min> argument.\n";
        printUsage(argv[0]);
        return 1;
    }

    uint32_t p = 0;

    int device_id = 0;  // Default device ID

    // Parse arguments
    for(int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-debug") == 0) {
            debug = true;
        }
        else if(std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        else if(std::strcmp(argv[i], "-d") == 0) {
            if(i + 1 < argc) {
                device_id = std::atoi(argv[i + 1]);
                i++; // Skip next argument (device_id)
            } else {
                std::cerr << "Error: Missing value for -d <device_id>." << std::endl;
                return 1;
            }
        }
        else {
            p = std::atoi(argv[i]);
        }
    }

    std::cout << "PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)" << std::endl;
    std::cout << "Testing exponent: " << p << std::endl;
    std::cout << "Using OpenCL device ID: " << device_id << std::endl;

    cl_int err;
    // 1. Get an OpenCL platform.
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if(err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "No OpenCL platform found." << std::endl;
        return 1;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[0];

    // 2. Get GPU devices from the selected platform.
    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if(err != CL_SUCCESS || numDevices == 0) {
        std::cerr << "No GPU device found." << std::endl;
        return 1;
    }
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);

    // 3. Validate the device_id argument and select the corresponding device.
    if(device_id < 0 || device_id >= static_cast<int>(numDevices)) {
        std::cerr << "Invalid device id specified (" << device_id << "). Using device 0 instead." << std::endl;
        device_id = 0;
    }
    cl_device_id device = devices[device_id];

    // 4. Create an OpenCL context.
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }
    // Crée une file de commandes pour le device sélectionné
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    // Pour OpenCL 1.x, tu peux utiliser :
    // cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        return 1;
    }

    
    // 5. Load the kernel source from "prmers.cl".
    std::string kernelSource;
    try {
        kernelSource = readFile("prmers.cl");
    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    const char* src = kernelSource.c_str();
    size_t srclen = kernelSource.size();
    
    // 6. Create the OpenCL program.
    cl_program program = clCreateProgramWithSource(context, 1, &src, &srclen, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    
    // 7. Build the program.
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
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

    // 8. Precompute values on the CPU.
    std::vector<uint64_t> digit_weight_cpu, digit_invweight_cpu;
    std::vector<int> digit_width_cpu;
    precalc_for_p(p, digit_weight_cpu, digit_invweight_cpu, digit_width_cpu);
    size_t n = transformsize(p);
    if (debug) {
        std::cout << "Size n for transform is n=" << n << std::endl;
    }
    // 9. Create OpenCL buffers.
    cl_uint p_min = p;
    cl_uint candidate_count = 1;
    std::vector<cl_uint> results(candidate_count, 0);
    
    cl_mem buffer_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(cl_uint) * candidate_count, results.data(), &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create results buffer." << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    
    cl_mem buf_digit_weight = clCreateBuffer(context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            digit_weight_cpu.size() * sizeof(uint64_t),
                            digit_weight_cpu.data(), &err);

    cl_mem buf_digit_invweight = clCreateBuffer(context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            digit_invweight_cpu.size() * sizeof(uint64_t),
                            digit_invweight_cpu.data(), &err);

    cl_mem buf_digit_width = clCreateBuffer(context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            digit_width_cpu.size() * sizeof(int),
                            digit_width_cpu.data(), &err);
    std::vector<uint64_t> x(n, 0ULL);
    x[0] = 4ULL;
    cl_mem buf_x           = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                              n*sizeof(uint64_t), x.data(), &err);
    // 10. Création et compilation du program 
    //    Puis création des kernels individuels
    cl_kernel k_precomp     = clCreateKernel(program, "kernel_precomp",     &err);
    cl_kernel k_forwardNTT  = clCreateKernel(program, "kernel_forward_ntt", &err);
    cl_kernel k_square      = clCreateKernel(program, "kernel_square",      &err);
    cl_kernel k_inverseNTT  = clCreateKernel(program, "kernel_inverse_ntt", &err);
    cl_kernel k_postcomp    = clCreateKernel(program, "kernel_postcomp",    &err);
    cl_kernel k_carry       = clCreateKernel(program, "kernel_carry",       &err);
    cl_kernel k_sub2        = clCreateKernel(program, "kernel_sub2",        &err);

    // 11. On fixe les arguments de chaque kernel. 
    //    Exemple pour k_precomp:
    clSetKernelArg(k_precomp, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_precomp, 1, sizeof(cl_mem), &buf_digit_weight);
    clSetKernelArg(k_precomp, 2, sizeof(uint64_t), &n);

    clSetKernelArg(k_forwardNTT, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_forwardNTT, 1, sizeof(uint64_t), &n);

    clSetKernelArg(k_square, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_square, 1, sizeof(uint64_t), &n);

    clSetKernelArg(k_inverseNTT, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_inverseNTT, 1, sizeof(uint64_t), &n);

    clSetKernelArg(k_postcomp, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_postcomp, 1, sizeof(cl_mem), &buf_digit_invweight);
    clSetKernelArg(k_postcomp, 2, sizeof(uint64_t), &n);

    clSetKernelArg(k_carry, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_carry, 1, sizeof(cl_mem), &buf_digit_width);
    clSetKernelArg(k_carry, 2, sizeof(uint64_t), &n);

    clSetKernelArg(k_sub2, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_sub2, 1, sizeof(cl_mem), &buf_digit_width);
    clSetKernelArg(k_sub2, 2, sizeof(uint64_t), &n);





    std::cout << "\nLaunching OpenCL kernel (p = " << p 
              << "); computation may take a while depending on the exponent." << std::endl;

    size_t globalSize = n;
    size_t nmax = std::min(n, (size_t)16);
    uint32_t total_iters = p - 2;


    auto start = high_resolution_clock::now();
    auto lastDisplay = start;

    size_t one = 1;
    size_t maxWork;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWork, nullptr);

    std::cout << "Max global workers possible: " << maxWork << std::endl;

    // Case 1: If `n` is smaller than `maxWork`, use `maxWork`
    size_t workers;
    if (n <= maxWork) {
        workers = maxWork;
    } 
    // Case 2: If `n` is greater than `maxWork`
    else {
        // If `n` is exactly divisible by `maxWork`, use `maxWork`
        if (n % maxWork == 0) {
            workers = maxWork;
        } 
        // Otherwise, find the largest possible `workers` value such that `n` is divisible by `workers`
        else {
            for (workers = maxWork; workers > 0; workers--) {
                if (n % workers == 0) {
                    break; // Stop when we find a valid multiple
                }
            }
        }
    }


    std::cout << "Final workers count: " << workers << std::endl;
    size_t maxLocalSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLocalSize, nullptr);
    std::cout << "Max local work size: " << maxLocalSize << std::endl;

    size_t localSize;

    // Choose the best local size
    if (n <= maxLocalSize) {
        localSize = n;  // Use full `n` if it's small
    } else {
        if (n % maxLocalSize == 0) {
            localSize = maxLocalSize;  // Perfect multiple
        } else {
            // Find largest divisor of `n` that fits in `maxLocalSize`
            for (localSize = maxLocalSize; localSize > 0; localSize--) {
                if (n % localSize == 0) {
                    break;  // Found the best fit
                }
            }
        }
    }

    std::cout << "Final local workers: " << localSize << std::endl;
    
    for (uint32_t iter = 0; iter < total_iters; iter++) {
        executeKernelAndDisplay(queue, k_precomp, buf_x, x, workers, one, "k_precomp", nmax);
        checkAndDisplayProgress(iter, total_iters, lastDisplay, start, queue);

        executeKernelAndDisplay(queue, k_forwardNTT, buf_x, x, workers, localSize, "k_forwardNTT", nmax);
        checkAndDisplayProgress(iter, total_iters, lastDisplay, start, queue);


        executeKernelAndDisplay(queue, k_square, buf_x, x, workers, one, "k_square", nmax);
        checkAndDisplayProgress(iter, total_iters, lastDisplay, start, queue);


        executeKernelAndDisplay(queue, k_inverseNTT, buf_x, x, workers, localSize, "k_inverseNTT", nmax);
        checkAndDisplayProgress(iter, total_iters, lastDisplay, start, queue);

        executeKernelAndDisplay(queue, k_postcomp, buf_x, x, workers, one, "k_postcomp", nmax);
        checkAndDisplayProgress(iter, total_iters, lastDisplay, start, queue);

        executeKernelAndDisplay(queue, k_carry, buf_x, x, one, one,  "k_carry", nmax);
        checkAndDisplayProgress(iter, total_iters, lastDisplay, start, queue);

        executeKernelAndDisplay(queue, k_sub2, buf_x, x, one, one,  "k_sub2", nmax);
        checkAndDisplayProgress(iter, total_iters, lastDisplay, start, queue);


    }
    std::cout << "End loop ";
    clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, n * sizeof(uint64_t), x.data(), 0, nullptr, nullptr);
    bool isPrime = std::all_of(x.begin(), x.end(), [](uint64_t v) { return v == 0; });

    std::cout << "\nM" << p << " is " << (isPrime ? "prime !" : "composite.") << std::endl;

    std::chrono::duration<double> elapsed = end - start;
    double elapsedTime = elapsed.count();
    double iters_per_sec = total_iters / elapsedTime;

    std::cout << "Kernel execution time: " << elapsedTime << " seconds" << std::endl;
    std::cout << "Iterations per second: " << iters_per_sec 
              << " (" << total_iters << " iterations in total)" << std::endl;

    // 14. Release OpenCL resources.
    // Libérer les buffers mémoire GPU
    clReleaseMemObject(buf_x);
    clReleaseMemObject(buffer_results);
    clReleaseMemObject(buf_digit_weight);
    clReleaseMemObject(buf_digit_invweight);
    clReleaseMemObject(buf_digit_width);
    // Libérer les kernels un par un
    clReleaseKernel(k_precomp);
    clReleaseKernel(k_forwardNTT);
    clReleaseKernel(k_square);
    clReleaseKernel(k_inverseNTT);
    clReleaseKernel(k_postcomp);
    clReleaseKernel(k_carry);
    clReleaseKernel(k_sub2);
    // Libérer le programme et la file de commandes
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
