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
#include <gmpxx.h>

namespace core {

class BackupManager {
public:
    BackupManager(cl_command_queue queue,
                  unsigned interval,
                  size_t vectorSize,
                  const std::string& savePath,
                  unsigned exponent,
                  const std::string& mode,
                  const uint64_t b1,
                  const uint64_t b2,
                  bool wagstaff,
                  bool marin
                  );

    // read existing .loop/.mers into x; return resume iteration
    uint64_t loadState(std::vector<uint64_t>& x);
    void loadGerbiczLiBufDState(std::vector<uint64_t>& x);
    void loadGerbiczLiCorrectState(std::vector<uint64_t>& x);
    void loadGerbiczLiCorrectBufDState(std::vector<uint64_t>& x);
    uint64_t loadGerbiczIterSave();
    uint64_t loadGerbiczJSave();
    
    uint64_t loadStatePM1S2(cl_mem hqBuf, cl_mem qBuf, size_t bytes);
    void     saveStatePM1S2(cl_mem hqBuf, cl_mem qBuf, uint64_t idx, size_t bytes);
    // read back from device and write .mers/.loop files at iteration iter
    void saveState(cl_mem buffer, uint64_t iter, const mpz_class* E_ptr = nullptr);
    void saveGerbiczLiState(cl_mem correctbuffer,cl_mem bufferd,cl_mem last_correctbufferd, uint64_t itersave, uint64_t jsave, const mpz_class* E_ptr = nullptr);
    
    mpz_class loadExponent() const;

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
    std::string      GerbiczLiBufDFilename_;
    std::string      GerbiczLiCorrectBufFilename_;
    std::string      GerbiczLiIterSaveFilename_;
    std::string      GerbiczLiJSaveFilename_;
    std::string      GerbiczLiLastBufDFilename_;
    std::string      exponentFilename_;
    uint64_t b1_;
    uint64_t b2_;
    bool             wagstaff_;
    bool             marin_;
    std::string hqFilename_, qFilename_, loop2Filename_;

};

} // namespace core
