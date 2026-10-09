#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <stdexcept>
namespace prmers::ocl {

class Context {
public:
    Context(int deviceIndex = 0, std::size_t enqueueMax = 0, bool cl_queue_throttle_active = false, bool debug = false/*, bool marin = false*/);
    ~Context();

    cl_context        getContext()  const noexcept;
    cl_device_id      getDevice()   const noexcept;
    cl_command_queue  getQueue()    const noexcept;
    std::size_t       getQueueSize() const noexcept;

    std::size_t getMaxWorkGroupSize() const noexcept;
    const std::vector<std::size_t>& getMaxWorkItemSizes() const noexcept;
    cl_ulong getLocalMemSize() const noexcept;

    std::size_t getLocalSize() const noexcept;
    std::size_t getLocalSize2() const noexcept;
    std::size_t getLocalSize3() const noexcept;
    std::size_t getLocalSize4() const noexcept;
    std::size_t getLocalSize5() const noexcept;
    std::size_t getLocalSizeCarry() const noexcept;
    std::size_t getWorkersCarry() const noexcept;
    cl_uint getTransformSize() const noexcept;
    int getLocalCarryPropagationDepth() const noexcept;
    uint64_t getExponent() const noexcept;
    bool isEvenExponent() const noexcept;
    cl_uint getWorkGroupCount() const noexcept;
    void computeOptimalSizes(std::size_t n, const std::vector<int>& digit_width_cpu, uint64_t p, bool debug = false, int localMaxSize = 0, int localMaxSize5 = 0);
    static void listAllOpenCLDevices();

    // Smallest power of two d with maxdw^d >= maxdw^2 * n. When the digits are at most 1 bit wide
    // that never holds (the doubling would not terminate), and a carry can ripple through every
    // digit, so the whole transform length is used.
    static std::size_t carryPropagationDepth(int maxdw, std::size_t n) {
        if (maxdw < 2) return n < 1 ? std::size_t(1) : n;
        std::size_t depth = 1;
        while (std::pow(maxdw, depth) < std::pow(maxdw, 2) * static_cast<double>(n)) depth *= 2;
        return depth;
    }
private:
    cl_platform_id    platform_;
    cl_device_id      device_;
    cl_context        context_;
    cl_command_queue  queue_;
    cl_uint           transformSize_;
    cl_uint workGroupCount_;
    std::size_t       queueSize_;

    std::size_t maxWorkGroupSize_;
    std::vector<std::size_t> maxWorkItemSizes_;
    cl_ulong localMemSize_;

    std::size_t localSize_;
    std::size_t localSize2_;
    std::size_t localSize3_;
    std::size_t localSize4_;
    std::size_t localSizeCarry_;
    std::size_t localSize5_;
    std::size_t workersCarry_;
    std::size_t localCarryPropagationDepth_{1};
    uint64_t exponent_;
    bool evenExponent_;
    bool debug_;

    void pickPlatformAndDevice(int deviceIndex);
    void createContext();
    void createQueue(std::size_t enqueueMax, bool cl_queue_throttle_active);
    void queryDeviceCapabilities();
    unsigned queryCLVersion() const;
    std::string queryDeviceString(cl_device_info) const;
    
};
}