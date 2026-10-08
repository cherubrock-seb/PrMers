// AlgoUtils.cpp
#include "core/AlgoUtils.hpp"
#include <atomic>
#include "core/StopRestartGate.hpp"

namespace core { namespace algo {
  std::atomic<bool> interrupted{false};

  void handle_sigint(int) noexcept {
    interrupted.store(true, std::memory_order_relaxed);
    // Also a Stop for the restart logic: the GUI idle loop must not restart the queue after it, and
    // a restart already committed must not go ahead.
    core::stop_or_exit();
  }
}}
