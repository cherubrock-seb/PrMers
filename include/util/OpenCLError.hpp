#ifndef UTIL_OPENCLERROR_HPP
#define UTIL_OPENCLERROR_HPP
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace util {

const char* getCLErrorString(cl_int err);

} // namespace util

#endif // UTIL_OPENCLERROR_HPP
