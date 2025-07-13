#ifndef IO_JSONBUILDER_HPP
#define IO_JSONBUILDER_HPP

#include <string>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "io/CliParser.hpp" 
#include <cstdint>
#include <vector>

namespace io {

class JsonBuilder {
public:
    static std::string generate(const std::vector<uint64_t>& x,
                                 const CliOptions& opts,
                                 const std::vector<int>& digit_width,
                                 double elapsed,
                                 int transform_size);

    // Write JSON string to a file.
    static void write(const std::string& json,
                      const std::string& path);

    static std::string computeRes2048(
         const std::vector<uint64_t>& x,
         const CliOptions& opts,
         const std::vector<int>& digit_width,
         double /*elapsed*/,
         int /*transform_size*/);

    static std::string computeRes64(
         const std::vector<uint64_t>& x,
         const CliOptions& opts,
         const std::vector<int>& digit_width,
         double /*elapsed*/,
         int /*transform_size*/);
     static std::string computeRes64Iter(
         const std::vector<uint64_t>& x,
         const CliOptions& opts,
         const std::vector<int>& digit_width,
         double /*elapsed*/,
         int /*transform_size*/);

    static std::vector<uint64_t> compactBits(
        const std::vector<uint64_t>& x,
        const std::vector<int>& digit_width,
        uint64_t E);
};

} // namespace io

#endif // IO_JSONBUILDER_HPP
