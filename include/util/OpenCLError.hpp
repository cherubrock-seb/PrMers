#ifndef UTIL_OPENCLERROR_HPP
#define UTIL_OPENCLERROR_HPP

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace util {

const char* getCLErrorString(cl_int err);

} // namespace util

#endif // UTIL_OPENCLERROR_HPP
