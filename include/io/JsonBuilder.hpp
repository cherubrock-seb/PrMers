#ifndef IO_JSONBUILDER_HPP
#define IO_JSONBUILDER_HPP

#include <string>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "io/CliParser.hpp" 

namespace io {

class JsonBuilder {
public:
    // Generate JSON for PrimeNet submission.
    // buffer : OpenCL buffer with final result words.
    // queue  : command queue (needed to clEnqueueReadBuffer).
    // opts   : parsed command-line options (CliOptions).
    // elapsed: total run time in seconds (for logging, if desired).
    static std::string generate(std::vector<unsigned long> x,
                                const CliOptions& opts,
                                const std::vector<int>& digit_width,
                                double elapsed,
                                int transform_size);

    // Write JSON string to a file.
    static void write(const std::string& json,
                      const std::string& path);

    static std::string computeRes2048(
    std::vector<unsigned long> x,
    const CliOptions& opts,
    const std::vector<int>& digit_width,
    double /*elapsed*/,
    int /*transform_size*/);

    static std::string computeRes64(
        std::vector<unsigned long> x,
        const CliOptions& opts,
        const std::vector<int>& digit_width,
        double /*elapsed*/,
        int /*transform_size*/);
};

} // namespace io

#endif // IO_JSONBUILDER_HPP
