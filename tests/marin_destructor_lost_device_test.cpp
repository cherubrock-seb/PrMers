// Marin engine destructors must not throw.  ~engine_gpu_flat and ~engine_gpu release the device through
// clFinish and clRelease*, which throw on an error; a destructor is noexcept, so on a reset or lost GPU (where
// the engine is destroyed while the first OpenCL error unwinds and every later call fails again) the process
// hit std::terminate instead of reporting the lost GPU.
//
// The OpenCL calls of the release path are defined here: they forward to the OpenCL library until a case
// "loses" the device, then return CL_INVALID_COMMAND_QUEUE.  Before the fix each case aborts the process.
// usage: marin-destructor-lost-device-test [device-index] [p]
#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "marin/engine_gpu.h"

namespace {
bool g_lost = false;
int g_lost_calls = 0;
int failures = 0;

template <class F>
F real(const char* name) {
    return reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
}

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}
}  // namespace

extern "C" {
CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue q) {
    if (g_lost) { ++g_lost_calls; return CL_INVALID_COMMAND_QUEUE; }
    static auto f = real<cl_int (*)(cl_command_queue)>("clFinish");
    return f(q);
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel k) {
    if (g_lost) { ++g_lost_calls; return CL_INVALID_COMMAND_QUEUE; }
    static auto f = real<cl_int (*)(cl_kernel)>("clReleaseKernel");
    return f(k);
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem m) {
    if (g_lost) { ++g_lost_calls; return CL_INVALID_COMMAND_QUEUE; }
    static auto f = real<cl_int (*)(cl_mem)>("clReleaseMemObject");
    return f(m);
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program p) {
    if (g_lost) { ++g_lost_calls; return CL_INVALID_COMMAND_QUEUE; }
    static auto f = real<cl_int (*)(cl_program)>("clReleaseProgram");
    return f(p);
}
}

namespace {
// An engine is destroyed while an OpenCL error unwinds: the caller must get that error, not std::terminate.
template <class Engine>
void unwind_case(const char* name, uint32_t p, size_t device) {
    g_lost_calls = 0;
    bool caught = false;
    try {
        std::unique_ptr<engine> eng(new Engine(p, 3, device, false));
        g_lost = true;
        throw util::gpulost::GpuLostError(std::string(util::gpulost::kLostPrefix) + ": simulated");
    } catch (const util::gpulost::GpuLostError& e) {
        caught = util::gpulost::is_lost_message(e.what());
    }
    g_lost = false;
    expect(caught, std::string(name) + ": the lost-GPU error did not reach the caller");
    expect(g_lost_calls > 0, std::string(name) + ": the release did not reach a failing OpenCL call");
    std::cout << "ok   " << name << ": destroyed during unwinding, lost-GPU error reported (" << g_lost_calls
              << " failing calls)\n";
}

// Plain scope exit on a lost device: no exception in flight, the destructor still must not throw.
template <class Engine>
void scope_case(const char* name, uint32_t p, size_t device) {
    g_lost_calls = 0;
    {
        Engine eng(p, 3, device, false);
        g_lost = true;
    }
    g_lost = false;
    expect(g_lost_calls > 0, std::string(name) + ": the release did not reach a failing OpenCL call");
    std::cout << "ok   " << name << ": destroyed on a lost device without throwing\n";
}
}  // namespace

int main(int argc, char** argv) {
    const size_t device = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 0;
    const uint32_t p = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 21701;
    {
        const ocl::platform platform;
        if (platform.get_device_count() <= device) { std::cerr << "no OpenCL device " << device << "\n"; return 2; }
    }

    unwind_case<engine_gpu>("engine_gpu", p, device);
    unwind_case<engine_gpu_flat>("engine_gpu_flat", p, device);
    scope_case<engine_gpu>("engine_gpu", p, device);
    scope_case<engine_gpu_flat>("engine_gpu_flat", p, device);

    // The explicit low-memory handoff still reports the error to its caller; the destructor afterwards is quiet.
    {
        bool threw = false;
        {
            engine_gpu eng(p, 3, device, false);
            g_lost = true;
            try { eng.release_gpu_resources_for_lowmem_handoff(); }
            catch (const util::gpulost::GpuLostError&) { threw = true; }
        }
        g_lost = false;
        expect(threw, "release_gpu_resources_for_lowmem_handoff on a lost device did not throw GpuLostError");
        std::cout << "ok   explicit release still throws GpuLostError\n";
    }

    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Marin destructor lost-device test: PASS\n";
    return 0;
}
