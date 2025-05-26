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
#include "opencl/Context.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#if defined(__APPLE__)
#  include <OpenCL/cl_ext.h>
#else
#  ifdef __has_include
#    if __has_include(<CL/cl_ext.h>)
#      include <CL/cl_ext.h>
#    endif
#  endif
#endif
#ifndef CL_DEVICE_UUID_KHR
#  define CL_DEVICE_UUID_KHR 0x106A
#endif
namespace opencl {

Context::Context(int deviceIndex, std::size_t enqueueMax, bool cl_queue_throttle_active)
    : platform_(nullptr), device_(nullptr),
      context_(nullptr), queue_(nullptr),
      queueSize_(0),
      maxWorkGroupSize_(0),
      localMemSize_(0),
      localSize_(0), localSize2_(0), localSize3_(0),
      localSizeCarry_(0), workersCarry_(2), localCarryPropagationDepth_(8),
      evenExponent_(true)
{
    pickPlatformAndDevice(deviceIndex);
    createContext();
    createQueue(enqueueMax, cl_queue_throttle_active);
    queryDeviceCapabilities(); 
}


Context::~Context() {
    if (queue_)   clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

void Context::pickPlatformAndDevice(int globalIndex) {
    cl_uint numPlat = 0;
    if (clGetPlatformIDs(0, nullptr, &numPlat) != CL_SUCCESS || numPlat == 0)
        throw std::runtime_error("No OpenCL platform found");
    std::vector<cl_platform_id> platforms(numPlat);
    clGetPlatformIDs(numPlat, platforms.data(), nullptr);

    struct DevInfo { cl_platform_id plat; cl_device_id dev; };
    std::vector<DevInfo> allDevices;
    for (auto &plat : platforms) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) != CL_SUCCESS) 
            continue;
        std::vector<cl_device_id> devs(ndev);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, ndev, devs.data(), nullptr);
        for (auto &d : devs)
            allDevices.push_back({plat, d});
    }
    if (allDevices.empty())
        throw std::runtime_error("No OpenCL GPU device found");

    if (globalIndex < 0 || size_t(globalIndex) >= allDevices.size()) {
        std::cerr << "Warning: invalid device index " 
                  << globalIndex << ", using device 0\n";
        globalIndex = 0;
    }
    platform_ = allDevices[globalIndex].plat;
    device_   = allDevices[globalIndex].dev;
}

void Context::listAllOpenCLDevices() {
    std::cout << "\nUsage: prmers [options] -d <device_index>\n\n";
    std::cout << "Select a GPU device by index from the list below:\n\n";
     cl_uint numPlat = 0;
    clGetPlatformIDs(0, nullptr, &numPlat);
    std::vector<cl_platform_id> plats(numPlat);
    clGetPlatformIDs(numPlat, plats.data(), nullptr);

    std::printf(
        "Idx | Platform            | Vendor           | Device Name                      | Driver Version         | Device Version        | CU  | Clock(MHz) | Mem(MiB) | UUID\n"
        "----+---------------------+------------------+----------------------------------+------------------------+-----------------------+-----+------------+----------+--------------------------------\n"
    );

    int globalIdx = 0;
    for (cl_uint pi = 0; pi < numPlat; ++pi) {
        size_t sz = 0;
        std::string platName, platVendor;
        clGetPlatformInfo(plats[pi], CL_PLATFORM_NAME, 0, nullptr, &sz);
        platName.resize(sz);
        clGetPlatformInfo(plats[pi], CL_PLATFORM_NAME, sz, &platName[0], nullptr);
        if (!platName.empty() && platName.back()=='\0') platName.pop_back();

        clGetPlatformInfo(plats[pi], CL_PLATFORM_VENDOR, 0, nullptr, &sz);
        platVendor.resize(sz);
        clGetPlatformInfo(plats[pi], CL_PLATFORM_VENDOR, sz, &platVendor[0], nullptr);
        if (!platVendor.empty() && platVendor.back()=='\0') platVendor.pop_back();

        cl_uint ndev = 0;
        if (clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) != CL_SUCCESS || ndev == 0)
            continue;
        std::vector<cl_device_id> devs(ndev);
        clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_GPU, ndev, devs.data(), nullptr);

        for (cl_uint di = 0; di < ndev; ++di) {
            cl_device_id dev = devs[di];

            char name[256]={0}, vendor[128]={0}, drvVer[128]={0}, devVer[128]={0}, extStr[1024]={0};
            cl_uint cu=0, clock=0;
            cl_ulong gmem=0;

            clGetDeviceInfo(dev, CL_DEVICE_NAME,                sizeof(name),    name,    nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_VENDOR,              sizeof(vendor),  vendor,  nullptr);
            clGetDeviceInfo(dev, CL_DRIVER_VERSION,             sizeof(drvVer),  drvVer,  nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_VERSION,             sizeof(devVer),  devVer,  nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof(cu),      &cu,     nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock),   &clock,  nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE,     sizeof(gmem),    &gmem,   nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS,          sizeof(extStr),  extStr,  nullptr);

            bool hasUUID = false;
            cl_uchar uuid[16] = {0};
            std::string exts(extStr);
            if (exts.find("cl_khr_device_uuid") != std::string::npos) {
                if (clGetDeviceInfo(dev, CL_DEVICE_UUID_KHR, sizeof(uuid), uuid, nullptr) == CL_SUCCESS)
                    hasUUID = true;
            }

            std::printf(
                " %2d | %-19s | %-16s | %-32s | %-22s | %-21s | %3u | %10u | %8llu | ",
                globalIdx,
                platName.c_str(),
                vendor,
                name,
                drvVer,
                devVer,
                cu,
                clock,
                (unsigned long long)(gmem / (1024 * 1024))
            );
            if (hasUUID) {
                for (int b = 0; b < 16; ++b) std::printf("%02X", uuid[b]);
            } else {
                std::printf("N/A");
            }
            std::printf("\n");

            ++globalIdx;
        }
    }
}



void Context::createContext() {
    cl_int err = CL_SUCCESS;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("Failed to create OpenCL context");
}

void Context::createQueue(std::size_t enqueueMax, bool cl_queue_throttle_active) {
    cl_int err = CL_SUCCESS;
    unsigned ver = queryCLVersion();

#if defined(__APPLE__)
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
#else
    if (ver >= 200)
        queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
    else
        queue_ = clCreateCommandQueue(context_, device_, 0, &err);
#endif

    if (err != CL_SUCCESS)
        throw std::runtime_error("Failed to create command queue");

    if (enqueueMax > 0) {
        queueSize_ = enqueueMax;
    }
    else {
        auto vendor = queryDeviceString(CL_DEVICE_VENDOR);
        std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::toupper);
        std::cout << "GPU Vendor: " << vendor << std::endl;
        if (vendor.find("NVIDIA") == std::string::npos) {
            #if defined(CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE) && defined(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE)
                size_t preferredSize = 0, maxSize = 0;
                clGetDeviceInfo(device_,
                                CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE,
                                sizeof(preferredSize),
                                &preferredSize,
                                nullptr);
                clGetDeviceInfo(device_,
                                CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
                                sizeof(maxSize),
                                &maxSize,
                                nullptr);
                std::cout  
                << "Device on‐device queue preferred=" << preferredSize
                << "  max=" << maxSize << "\n";
                queueSize_ = preferredSize;
                //queueSize_ = 16 * 1024;
                
            #else
                queueSize_ = 16 * 1024;
            #endif
            if (queueSize_ == 0)
                queueSize_ = 16 * 1024;
        }
        //queueSize_ = 16 * 1024; //temporary
    }
    if(queueSize_ == 18446744073709551615){
        queueSize_=0;
    }
    std::cout << "Queue size = " << queueSize_ << std::endl;
}

void Context::queryDeviceCapabilities() {
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(maxWorkGroupSize_), &maxWorkGroupSize_, nullptr);

    std::size_t sizes[3] = {0};
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    sizeof(sizes), sizes, nullptr);
    maxWorkItemSizes_.assign(sizes, sizes + 3);

    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(localMemSize_), &localMemSize_, nullptr);

    std::cout << "Max CL_DEVICE_MAX_WORK_GROUP_SIZE = " << maxWorkGroupSize_ << std::endl;
    std::cout << "Max CL_DEVICE_MAX_WORK_ITEM_SIZES = "
              << maxWorkItemSizes_[0] << ", "
              << maxWorkItemSizes_[1] << ", "
              << maxWorkItemSizes_[2] << std::endl;
    std::cout << "Max CL_DEVICE_LOCAL_MEM_SIZE = " << localMemSize_ << " bytes" << std::endl;
}

void Context::computeOptimalSizes(std::size_t n,
                                  const std::vector<int>& digit_width_cpu,
                                  int p,
                                  bool debug)
{
    transformSize_ = static_cast<cl_uint>(n);
    if (n < 4) n = 4;

    cl_uint mm = n / 4;
    for (cl_uint m = n / 16; m >= 32; m /= 16)
        mm = m / 16;

    evenExponent_ = !(mm == 8 || mm == 2 || mm == 32) || (n == 4);
    exponent_ = p;

    if (debug) {
        std::cout << "n=" << n
                  << " evenExponent=" << evenExponent_
                  << std::endl;
    }

    size_t maxWork = maxWorkGroupSize_;
    if (maxWork > 256) maxWork = 256;
    std::size_t workers = n;

    localSize_ = maxWork;

    cl_uint constraint = std::max<cl_uint>(n / 16, 1u);
    while (workers % localSize_ != 0 || constraint % localSize_ != 0) {
        localSize_ /= 2;
        if (localSize_ < 1) { localSize_ = 1; break; }
    }

    // Calcul dynamique de localCarryPropagationDepth_
    localCarryPropagationDepth_ = 1;
    int maxdw = *std::max_element(digit_width_cpu.begin(), digit_width_cpu.end());
    if (debug) std::cout << "max digit width = " << maxdw << std::endl;
    while (std::pow(maxdw, localCarryPropagationDepth_) < std::pow(maxdw, 2) * n) {
        localCarryPropagationDepth_ *= 2;
    }

    if (workers % localCarryPropagationDepth_ == 0) {
        workersCarry_ = workers / localCarryPropagationDepth_;
    } else {
        int trial = 8;
        std::size_t wc = 1;
        while (workers % trial == 0) {
            wc = workers / trial;
            trial *= 2;
        }
        workersCarry_ = (wc > 1 ? wc : 1);
        if (workersCarry_ == 1)
            localCarryPropagationDepth_ = n;
    }

    localSizeCarry_ = std::min(workersCarry_, localSize_);
    localSize2_ = localSize_;
    localSize3_ = localSize_;
    workGroupCount_ = (transformSize_ < localSize_) ? 1u : transformSize_ / static_cast<cl_uint>(localSize_);

    if (debug) {
        std::cout << "final localSize=" << localSize_
                  << " carryDepth=" << localCarryPropagationDepth_
                  << " workersCarry=" << workersCarry_
                  << " localSizeCarry=" << localSizeCarry_
                  << std::endl;
    }
}




// Getters
std::size_t Context::getMaxWorkGroupSize() const noexcept { return maxWorkGroupSize_; }
const std::vector<std::size_t>& Context::getMaxWorkItemSizes() const noexcept { return maxWorkItemSizes_; }
cl_ulong Context::getLocalMemSize() const noexcept { return localMemSize_; }

std::size_t Context::getLocalSize() const noexcept { return localSize_; }
std::size_t Context::getLocalSize2() const noexcept { return localSize2_; }
std::size_t Context::getLocalSize3() const noexcept { return localSize3_; }
std::size_t Context::getLocalSizeCarry() const noexcept { return localSizeCarry_; }
std::size_t Context::getWorkersCarry() const noexcept { return workersCarry_; }
int Context::getLocalCarryPropagationDepth() const noexcept { return localCarryPropagationDepth_; }
int Context::getExponent() const noexcept { return exponent_; }

cl_context Context::getContext() const noexcept { return context_; }
cl_device_id Context::getDevice() const noexcept { return device_; }
cl_command_queue Context::getQueue() const noexcept { return queue_; }
std::size_t Context::getQueueSize() const noexcept { return queueSize_; }
cl_uint Context::getTransformSize() const noexcept { return transformSize_; }

bool Context::isEvenExponent() const noexcept {
    return evenExponent_;
}

cl_uint Context::getWorkGroupCount() const noexcept {
    return workGroupCount_;
}


unsigned Context::queryCLVersion() const {
    char buf[128] = {0};
    if (clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(buf), buf, nullptr) != CL_SUCCESS)
        return 110;
    unsigned major = 1, minor = 1;
    if (std::sscanf(buf, "OpenCL %u.%u", &major, &minor) == 2)
        return major * 100 + minor;
    return major * 100 + minor;
}

std::string Context::queryDeviceString(cl_device_info info) const {
    size_t sz = 0;
    clGetDeviceInfo(device_, info, 0, nullptr, &sz);
    std::string s(sz, '\0');
    clGetDeviceInfo(device_, info, sz, &s[0], nullptr);
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

} // namespace opencl
