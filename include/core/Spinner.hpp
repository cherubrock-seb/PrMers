// include/core/Spinner.hpp
#pragma once

#include <atomic>
#include <cstdint>

namespace core {

class Spinner {
public:
    Spinner() = default;
    ~Spinner() = default;

    void displayProgress(uint32_t iter,
                        uint32_t totalIters,
                        double elapsedTime,
                        uint32_t expo,
                        uint32_t resumeIter = 0);

    void displayBackupInfo(uint32_t iter,
                           uint32_t totalIters,
                           double elapsedTime);

    void displaySpinner(std::atomic<bool>& waiting,
                        double estimatedSeconds,
                        std::atomic<bool>& isFirst);
};

} // namespace core
