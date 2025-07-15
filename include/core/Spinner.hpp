// include/core/Spinner.hpp
#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>

namespace core {

class Spinner {
public:
    Spinner() = default;
    ~Spinner() = default;

    void displayProgress(uint64_t iter,
                        uint64_t totalIters,
                        double elapsedTime,
                        double elapsedTime2,
                        uint64_t expo,
                        uint64_t resumeIter = 0,
                        uint64_t startIter = 0,
                        std::string res64 = "");

    void displayBackupInfo(uint64_t iter,
                           uint64_t totalIters,
                           double elapsedTime,
                           std::string res64);

    void displaySpinner(std::atomic<bool>& waiting,
                        double estimatedSeconds,
                        std::atomic<bool>& isFirst);
};

} // namespace core
