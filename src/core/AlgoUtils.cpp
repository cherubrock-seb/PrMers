// AlgoUtils.cpp
#include "core/AlgoUtils.hpp"
#include <atomic>

namespace core { namespace algo {
  std::atomic<bool> interrupted{false};

  void handle_sigint(int) noexcept {
    interrupted.store(true, std::memory_order_relaxed);
  }
}}
