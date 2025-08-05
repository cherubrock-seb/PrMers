#ifndef IO_JSONBUILDER_HPP
#define IO_JSONBUILDER_HPP

#include <string>
#include <tuple>
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
    static std::tuple<bool, std::string, std::string> computeResult(
        const std::vector<uint64_t>& hostResult,
        const CliOptions& opts,
        const std::vector<int>& digit_width);

    static std::string generate(const CliOptions& opts,
                                 int transform_size,
                                 bool isPrime,
                                 const std::string& res64,
                                 const std::string& res2048);

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

    static std::vector<uint32_t> compactBits(
        const std::vector<uint64_t>& x,
        const std::vector<int>& digit_width,
        uint32_t E);
    
    static std::vector<uint64_t> expandBits(
        const std::vector<uint32_t>& compactWords,
        const std::vector<int>& digit_width,
        uint32_t E);
};

} // namespace io

#endif // IO_JSONBUILDER_HPP
