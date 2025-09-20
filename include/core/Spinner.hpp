// include/core/Spinner.hpp
#pragma once
#include <atomic>
#include <cstdint>
#include <string>

namespace ui { class WebGuiServer; }

namespace core {

class Spinner {
public:
  void displayProgress(uint64_t iter,
                       uint64_t totalIters,
                       double elapsedTime,
                       double elapsedTime2,
                       uint64_t expo,
                       uint64_t resumeIter,
                       uint64_t startIter,
                       std::string res64,
                       ui::WebGuiServer* gui = nullptr);

  void displayProgress2(uint64_t iter,
                       uint64_t totalIters,
                       double elapsedTime,
                       double elapsedTime2,
                       uint64_t expo,
                       uint64_t resumeIter,
                       uint64_t startIter,
                       std::string res64,
                       ui::WebGuiServer* gui,
                       uint64_t chunkIndex,
                       uint64_t chunkCount,
                       uint64_t chunkIter,
                       uint64_t chunkTotal,
                       bool reset);


  void displayBackupInfo(uint64_t iter,
                         uint64_t totalIters,
                         double elapsedTime,
                         std::string res64,
                         ui::WebGuiServer* gui = nullptr);

  void displaySpinner(std::atomic<bool>& waiting,
                      double estimatedSeconds,
                      std::atomic<bool>& isFirst);
};

}
