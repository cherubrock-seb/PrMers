// include/core/BackupManager.hpp
#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <cstdint>
#include <string>
#include <vector>

namespace core {

class BackupManager {
public:
    BackupManager(cl_command_queue queue,
                  unsigned interval,
                  size_t vectorSize,
                  const std::string& savePath,
                  unsigned exponent,
                  const std::string& mode);

    // read existing .loop/.mers into x; return resume iteration
    uint32_t loadState(std::vector<uint64_t>& x);

    // read back from device and write .mers/.loop files at iteration iter
    void saveState(cl_mem buffer, uint32_t iter);

    void clearState() const;
private:
    cl_command_queue queue_;
    unsigned         backupInterval_;
    size_t           vectorSize_;
    std::string      savePath_;
    unsigned         exponent_;
    std::string      mode_;
    std::string      mersFilename_;
    std::string      loopFilename_;
};

} // namespace core
