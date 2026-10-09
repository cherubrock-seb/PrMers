// Meaning of an OpenCL error status, for the user-facing message.
//
// When a GPU is reset or lost (a driver timeout or reset, a removed or crashed device) every later OpenCL call
// on the old context fails.  The codes below are what the drivers return then.  They used to surface as a raw
// code, or (Aevum) as std::bad_alloc, which reads as an out-of-memory error.
//
// Header-only and free of OpenCL types so any engine can include it.  Aevum is a separate tree and keeps its own
// copy (third_party/aevum/src/GpuLost.h); tests/gpu_lost_message_test.cpp checks that both agree.
//
// The classification depends on the call that failed, not only on the code:
//   Phase::Create  an allocation or creation call (clCreateBuffer, clCreateContext, clCreateCommandQueue,
//                  clCreateProgramWithSource, clBuildProgram, clCreateKernel) and the zero-fill that
//                  immediately initialises a new buffer;
//   Phase::Run     everything else: enqueue, finish, flush, read, write, copy, fill, wait, event status.
//
//   code                                   Create          Run
//   -4  CL_MEM_OBJECT_ALLOCATION_FAILURE   out of memory   out of memory
//   -6  CL_OUT_OF_HOST_MEMORY              out of memory   out of memory
//   -5  CL_OUT_OF_RESOURCES                out of memory   GPU reset or lost
//   -2  CL_DEVICE_NOT_AVAILABLE            GPU reset or lost  (both)
//   -34 CL_INVALID_CONTEXT                 GPU reset or lost  (both)
//   -36 CL_INVALID_COMMAND_QUEUE           GPU reset or lost  (both)
//   -9999 (NVIDIA driver error)            GPU reset or lost  (both)
//   anything else                          other: the caller keeps its existing message
//
// -5 is the one ambiguous code.  A creation call returns it when the device cannot satisfy the allocation,
// which is a size problem.  An enqueue on a context that was working returns it when the command could not run
// any more, which is what a reset looks like to the caller.  The other lost codes cannot come from a healthy
// context in either phase.
#ifndef UTIL_GPULOST_HPP
#define UTIL_GPULOST_HPP

#include <stdexcept>
#include <string>

namespace util {
namespace gpulost {

enum class Phase { Create, Run };
enum class Kind { Other, OutOfMemory, DeviceLost };

constexpr int kDeviceNotAvailable = -2;
constexpr int kMemObjectAllocationFailure = -4;
constexpr int kOutOfResources = -5;
constexpr int kOutOfHostMemory = -6;
constexpr int kInvalidContext = -34;
constexpr int kInvalidCommandQueue = -36;
// Not in the OpenCL headers: the NVIDIA driver returns it for a failed or faulted context.
constexpr int kNvidiaDriverError = -9999;

// Every message of the lost class starts with this, so a message that crossed a C boundary as a plain string
// (the Aevum plugin) can be recognised again.
constexpr const char* kLostPrefix = "GPU reset or lost";

constexpr Kind classify(int err, Phase phase) {
    switch (err) {
        case kMemObjectAllocationFailure:
        case kOutOfHostMemory:
            return Kind::OutOfMemory;
        case kOutOfResources:
            return phase == Phase::Create ? Kind::OutOfMemory : Kind::DeviceLost;
        case kDeviceNotAvailable:
        case kInvalidContext:
        case kInvalidCommandQueue:
        case kNvidiaDriverError:
            return Kind::DeviceLost;
        default:
            return Kind::Other;
    }
}

// Name for the one code the OpenCL headers do not define; nullptr for every other code.
constexpr const char* vendor_name(int err) {
    return err == kNvidiaDriverError ? "CL_NVIDIA_DRIVER_ERROR" : nullptr;
}

// codeText is the engine's own rendering of the code, for example "CL_OUT_OF_RESOURCES (-5)".  what names the
// call that failed and may be empty.
inline std::string lost_message(const std::string& codeText, const std::string& what) {
    std::string m = kLostPrefix;
    m += ": OpenCL error " + codeText;
    if (!what.empty()) m += " in " + what;
    m += ". The GPU was reset (driver timeout or reset) or the device or context was lost. "
         "Your work is safe up to the last checkpoint; restart PrMers to resume from it.";
    return m;
}

inline std::string oom_message(const std::string& codeText, const std::string& what) {
    std::string m = "Out of GPU memory: ";
    m += what.empty() ? std::string("OpenCL call") : what;
    m += " failed with " + codeText +
         ". The problem does not fit in the available memory (device or host); try a smaller exponent or free memory.";
    return m;
}

inline bool is_lost_message(const std::string& text) {
    return text.compare(0, std::char_traits<char>::length(kLostPrefix), kLostPrefix) == 0;
}

// Thrown for the lost class so a caller can tell it from any other OpenCL failure.  what() is the one-line
// message and begins with kLostPrefix.
class GpuLostError : public std::runtime_error {
public:
    explicit GpuLostError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace gpulost
} // namespace util

#endif // UTIL_GPULOST_HPP
