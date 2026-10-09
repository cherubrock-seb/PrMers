#ifndef UTIL_OPENCLERROR_HPP
#define UTIL_OPENCLERROR_HPP
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

#include "util/GpuLost.hpp"

namespace util {

const char* getCLErrorString(cl_int err);

// Throws for a failed OpenCL call in the legacy NTT engine.  `what` names the call and `phase` says whether it
// was an allocation or creation call or something that runs on the device (see util/GpuLost.hpp).  A GPU that was
// reset or lost throws gpulost::GpuLostError, an allocation failure throws a std::runtime_error saying the GPU is
// out of memory, and any other code throws std::runtime_error(generic), the caller's existing message.
// The text throwClError() would throw, for callers that report and exit instead of throwing.
std::string describeClError(cl_int err, gpulost::Phase phase, const std::string& what, const std::string& generic);

// "Failed to allocate <name>: <err>" for a failed clCreateBuffer, worded as out of memory or as a lost GPU when the
// code says so.
std::string describeClAllocFailure(cl_int err, const std::string& name);

[[noreturn]] void throwClError(cl_int err, gpulost::Phase phase, const std::string& what, const std::string& generic);

} // namespace util

#endif // UTIL_OPENCLERROR_HPP
