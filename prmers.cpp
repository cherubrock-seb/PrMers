#define CL_TARGET_OPENCL_VERSION 200
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

uint64_t mulModP(uint64_t a, uint64_t b)
{
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t lo = (uint64_t)prod;
    uint32_t A = (uint32_t)(hi >> 32);
    uint32_t B = (uint32_t)(hi & 0xffffffffULL);
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
    if(r >= MOD_P)
        r -= MOD_P;
    return r;
}

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

uint64_t invModP(uint64_t x)
{
    return powModP(x, MOD_P - 2ULL);
}

// 2) Compute transform size for the given exponent : force to be a power of 4 (radix 4)
static size_t transformsize(uint32_t exponent)
{
    int log_n = 0, w = 0;
    do {
        ++log_n;
        w = exponent / (1 << log_n);
    } while (((w + 1) * 2 + log_n) >= 63);
    if (log_n & 1)
        ++log_n;
    return (size_t)(1ULL << log_n);
}



// 3) Precompute digit weights, inverse weights, and digit widths for a given p.
void precalc_for_p(uint32_t p,
                   std::vector<uint64_t>& digit_weight,
                   std::vector<uint64_t>& digit_invweight,
                   std::vector<int>& digit_width)
{
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

std::string readFile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Unable to open file: " + filename);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <p> [-d <device_id>] [-O <options>] [-profile]" << std::endl;
    std::cout << "  <p>       : Minimum exponent to test (required)" << std::endl;
    std::cout << "  -d <device_id>: (Optional) Specify OpenCL device ID (default: 0)" << std::endl;
    std::cout << "  -O <options>  : (Optional) Enable OpenCL optimization flags (can combine multiple)." << std::endl;
    std::cout << "                  Available options: " << std::endl;
    std::cout << "                    fastmath    => Enables -cl-fast-relaxed-math" << std::endl;
    std::cout << "                    mad         => Enables -cl-mad-enable" << std::endl;
    std::cout << "                    unsafe      => Enables -cl-unsafe-math-optimizations" << std::endl;
    std::cout << "                    nans        => Disables NaN/Inf checks (-cl-no-signed-zeros)" << std::endl;
    std::cout << "                    optdisable  => Disables compiler optimizations (-cl-opt-disable)" << std::endl;
    std::cout << "  -profile      : (Optional) Enable kernel execution profiling to measure execution times." << std::endl;
    std::cout << "  Example: " << progName << " 127 -O fastmath mad -profile" << std::endl;
}

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
                else if (opt[0] == '-') break;  // Arrêter si un autre flag est rencontré
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
void executeKernelAndDisplay(cl_command_queue queue, cl_kernel kernel, 
                             cl_mem buf_x, size_t workers, size_t localSize,
                             const std::string& kernelName, size_t nmax,
                             bool profiling) 
{
    cl_event event;
    cl_int err;
    
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &workers, &localSize, 0, nullptr, profiling ? &event : nullptr);
    clFinish(queue);

    if (profiling) {
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
    std::cout << "\r" << color  
              << "Progress: " << std::fixed << std::setprecision(2) << progress << "% | "
              << "Elapsed: " << elapsedTime << "s | "
              << "Iterations/sec: " << iters_per_sec << " | "
              << "ETA: " << (remaining_time > 0 ? remaining_time : 0.0) << "s       "
              << COLOR_RESET  
              << std::flush;
}

void checkAndDisplayProgress(uint32_t iter, uint32_t total_iters, 
                             time_point<high_resolution_clock>& lastDisplay, 
                             const time_point<high_resolution_clock>& start, 
                             cl_command_queue queue) {
    auto duration = duration_cast<seconds>(high_resolution_clock::now() - lastDisplay).count();
    if (duration >= 5 || iter==-1) {  
        if(iter==-1){
            iter = total_iters;
        }
        double elapsedTime = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9;
        displayProgress(iter, total_iters, elapsedTime);
        lastDisplay = high_resolution_clock::now();
    }
}

void executeFusionneNTT_Forward(cl_command_queue queue, cl_kernel kernel_ntt, 
                        cl_mem buf_x, cl_mem buf_w, size_t n, size_t workers,  
                        size_t localSize, bool profiling) {
    for (size_t m = n / 4; m >= 1; m /= 4) {
        clSetKernelArg(kernel_ntt, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt, 1, sizeof(cl_mem), &buf_w);
        clSetKernelArg(kernel_ntt, 2, sizeof(size_t), &n);
        clSetKernelArg(kernel_ntt, 3, sizeof(size_t), &m);

        executeKernelAndDisplay(queue, kernel_ntt, buf_x, workers, localSize, 
                                "kernel_ntt_radix4 (m=" + std::to_string(m) + ")", n, profiling);
    }
}

void executeFusionneNTT_Inverse(cl_command_queue queue, cl_kernel kernel_ntt, 
                        cl_mem buf_x, cl_mem buf_w, size_t n, size_t workers,  
                        size_t localSize, bool profiling) {
    for (size_t m = 1; m <= n / 4; m *= 4) {
        clSetKernelArg(kernel_ntt, 0, sizeof(cl_mem), &buf_x);
        clSetKernelArg(kernel_ntt, 1, sizeof(cl_mem), &buf_w);
        clSetKernelArg(kernel_ntt, 2, sizeof(size_t), &n);
        clSetKernelArg(kernel_ntt, 3, sizeof(size_t), &m);

        executeKernelAndDisplay(queue, kernel_ntt, buf_x, workers, localSize, 
                                "kernel_inverse_ntt_radix4 (m=" + std::to_string(m) + ")", n, profiling);
    }
}


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Error: Missing <p_min> argument.\n";
        printUsage(argv[0]);
        return 1;
    }
    uint32_t p = 0;
    int device_id = 0;  // Default device ID
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
                i++;
            } else {
                std::cerr << "Error: Missing value for -d <device_id>." << std::endl;
                return 1;
            }
        }
        else {
            p = std::atoi(argv[i]);
        }
    }
    bool profiling = false;  // Par défaut, désactivé

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-profile") == 0) {
            profiling = true;
        }
    }
    if (profiling) {
        std::cout << "\n🔍 Kernel profiling is activated. Performance metrics will be displayed.\n" << std::endl;
    }

    std::cout << "PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)" << std::endl;
    std::cout << "Testing exponent: " << p << std::endl;
    std::cout << "Using OpenCL device ID: " << device_id << std::endl;
    
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
    
    cl_program program = clCreateProgramWithSource(context, 1, &src, &srclen, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    std::string build_options = getBuildOptions(argc, argv);
    if (!build_options.empty()) {
        std::cout << "Building OpenCL program with options: " << build_options << std::endl;
    }
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
    
    std::vector<uint64_t> digit_weight_cpu, digit_invweight_cpu;
    std::vector<int> digit_width_cpu;
    precalc_for_p(p, digit_weight_cpu, digit_invweight_cpu, digit_width_cpu);
    size_t n = transformsize(p);
    if (debug) {
        std::cout << "Size n for transform is n=" << n << std::endl;
    }
    
    // Precompute twiddle factors for radix-4 (forward and inverse).
    std::vector<uint64_t> twiddles(3 * n, 0ULL), inv_twiddles(3 * n, 0ULL);
    if(n >= 4) {  // Only needed if n>=4
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
    
    // Create OpenCL buffers.
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
    
    cl_mem buf_twiddles = clCreateBuffer(context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            twiddles.size() * sizeof(uint64_t),
                            twiddles.data(), &err);
    
    cl_mem buf_inv_twiddles = clCreateBuffer(context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            inv_twiddles.size() * sizeof(uint64_t),
                            inv_twiddles.data(), &err);
    
    
    std::vector<uint64_t> x(n, 0ULL);
    x[0] = 4ULL;
    cl_mem buf_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                  n * sizeof(uint64_t), x.data(), &err);
    
    // Create kernels.
    cl_kernel k_precomp      = clCreateKernel(program, "kernel_precomp", &err);
    cl_kernel k_postcomp     = clCreateKernel(program, "kernel_postcomp", &err);
    cl_kernel k_forward_ntt  = clCreateKernel(program, "kernel_ntt_radix4", &err);
    cl_kernel k_inverse_ntt  = clCreateKernel(program, "kernel_inverse_ntt_radix4", &err);
    cl_kernel k_square       = clCreateKernel(program, "kernel_square", &err);
    cl_kernel k_carry = clCreateKernel(program, "kernel_carry", &err);
    cl_kernel k_sub2  = clCreateKernel(program, "kernel_sub2", &err);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernels." << std::endl;
        return 1;
    }

    clSetKernelArg(k_precomp, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_precomp, 1, sizeof(cl_mem), &buf_digit_weight);
    clSetKernelArg(k_precomp, 2, sizeof(uint64_t), &n);

    clSetKernelArg(k_postcomp, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_postcomp, 1, sizeof(cl_mem), &buf_digit_invweight);
    clSetKernelArg(k_postcomp, 2, sizeof(uint64_t), &n);

    clSetKernelArg(k_square, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_square, 1, sizeof(uint64_t), &n);

        // Set arguments for k_carry and k_sub2 (unchanged).
    clSetKernelArg(k_carry, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_carry, 1, sizeof(cl_mem), &buf_digit_width);
    clSetKernelArg(k_carry, 2, sizeof(uint64_t), &n);
    
    clSetKernelArg(k_sub2, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(k_sub2, 1, sizeof(cl_mem), &buf_digit_width);
    clSetKernelArg(k_sub2, 2, sizeof(uint64_t), &n);
    

    size_t maxThreads;
    size_t workitem_size[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%lu / %lu / %lu \n", workitem_size[0], workitem_size[1], workitem_size[2]);
    maxThreads = workitem_size[0];
    size_t maxWork;
   
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWork, nullptr);
    std::cout << "Max CL_DEVICE_MAX_WORK_GROUP_SIZE = " << maxWork << std::endl;
    std::cout << "Max CL_DEVICE_MAX_WORK_ITEM_SIZES = " << maxThreads << std::endl;
    std::cout << "\nLaunching OpenCL kernel (p = " << p 
              << "); computation may take a while depending on the exponent." << std::endl;
         
     
    size_t workers = n;
    size_t localSize = maxWork;


    // Ajuster localSize pour être un diviseur de (n/4) et ≤ workers
    size_t constraint = std::max(n / 4, (size_t)1);  
    while (workers % localSize != 0 || constraint % localSize != 0) {
        localSize /= 2; 
        if (localSize < 1) {
            localSize = 1;
            break;
        }
    }


    std::cout << "Final workers count: " << workers << std::endl;
    std::cout << "Work-groups count: " << localSize << std::endl;
    std::cout << "Work-groups size: " << ((workers < localSize) ? 1 : (workers / localSize)) << std::endl;

    size_t sizeCarryBuffer =  maxWork;
    if(n<workers){
        sizeCarryBuffer = n;
    }
    std::vector<uint64_t> carry_array(sizeCarryBuffer, 0ULL);
    cl_mem buf_carry_array= clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                            sizeCarryBuffer * sizeof(uint64_t), carry_array.data(), &err);
    clSetKernelArg(k_carry, 3, sizeof(cl_mem), &buf_carry_array);
        // Création du buffer flag
    cl_int zero = 0;
    cl_mem flagBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &zero, &err);
    clSetKernelArg(k_carry, 4, sizeof(cl_mem), &flagBuffer);

    uint32_t total_iters = p - 2;
    
    auto startTime = high_resolution_clock::now();
    auto lastDisplay = startTime;
    checkAndDisplayProgress(0, total_iters, lastDisplay, startTime, queue);
    
    for (uint32_t iter = 0; iter < total_iters; iter++) {
        executeKernelAndDisplay(queue, k_precomp, buf_x, workers, localSize, "kernel_precomp", n, profiling);
        executeFusionneNTT_Forward(queue, k_forward_ntt, buf_x, buf_twiddles, n, workers, localSize, profiling);
        executeKernelAndDisplay(queue, k_square, buf_x, workers, localSize, "kernel_square", n, profiling);
        executeFusionneNTT_Inverse(queue, k_inverse_ntt, buf_x, buf_inv_twiddles, n, workers, localSize, profiling);
        executeKernelAndDisplay(queue, k_postcomp, buf_x, workers, localSize, "kernel_postcomp", n, profiling);
        executeKernelAndDisplay(queue, k_carry, buf_x, sizeCarryBuffer, sizeCarryBuffer, "kernel_carry", n, profiling);
        executeKernelAndDisplay(queue, k_sub2, buf_x, 1, 1, "kernel_sub2", n, profiling);
        checkAndDisplayProgress(iter, total_iters, lastDisplay, startTime, queue);
    }
    checkAndDisplayProgress(-1, total_iters, lastDisplay, startTime, queue);
    
    clFinish(queue);
    auto endTime = high_resolution_clock::now();
    
    clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, n * sizeof(uint64_t), x.data(), 0, nullptr, nullptr);
    bool isPrime = std::all_of(x.begin(), x.end(), [](uint64_t v) { return v == 0; });
    
    std::cout << "\nM" << p << " is " << (isPrime ? "prime !" : "composite.") << std::endl;
    
    std::chrono::duration<double> elapsed = endTime - startTime;
    double elapsedTime = elapsed.count();
    double iters_per_sec = total_iters / elapsedTime;
    
    std::cout << "Kernel execution time: " << elapsedTime << " seconds" << std::endl;
    std::cout << "Iterations per second: " << iters_per_sec 
              << " (" << total_iters << " iterations in total)" << std::endl;
    
    // Release OpenCL resources.
    clReleaseMemObject(buf_x);
    clReleaseMemObject(buf_twiddles);
    clReleaseMemObject(buf_inv_twiddles);
    clReleaseMemObject(buf_carry_array);
    clReleaseMemObject(buf_digit_weight);
    clReleaseMemObject(buf_digit_invweight);
    clReleaseMemObject(buf_digit_width);
    clReleaseKernel(k_precomp);
    clReleaseKernel(k_postcomp);
    clReleaseKernel(k_forward_ntt);
    clReleaseKernel(k_inverse_ntt);
    clReleaseKernel(k_square);
    clReleaseProgram(program);
    clReleaseKernel(k_carry);
    clReleaseKernel(k_sub2);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
