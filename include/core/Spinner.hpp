// include/core/Spinner.hpp
#pragma once

#include <atomic>
#include <cstdint>

namespace core {

class Spinner {
public:
    Spinner() = default;
    ~Spinner() = default;

    // Called once at startup to show “0%”:
    void displayProgress(uint32_t iter,
                         uint32_t totalIters,
                         double elapsedTime,
                         uint32_t expo);

    // Called whenever you save a backup:
    void displayBackupInfo(uint32_t iter,
                           uint32_t totalIters,
                           double elapsedTime);

    // (Optional) call around a long clFinish to show a spinner:
    void displaySpinner(std::atomic<bool>& waiting,
                        double estimatedSeconds,
                        std::atomic<bool>& isFirst);
};

} // namespace core
