/*
Aevum runtime autotune, Copyright 2026 cherubrock-seb.
Licensed under GNU GPL version 3. See LICENSE and UPSTREAM.md.
*/

#pragma once

#include <chrono>
#include <cstdint>

// The time budget of the plan search.  It counts the time spent measuring
// candidates, not the setup around them: creating the tuning context and
// building each engine (which compiles its kernels on a cold kernel cache) are
// excluded.  Counting them let a cold start spend the whole budget compiling
// and measure only about one candidate.  The candidate cap still bounds how
// many engines a search builds.
template <class Clock = std::chrono::steady_clock>
class TuneBudget {
public:
  explicit TuneBudget(typename Clock::time_point start) : start_(start) {}

  // Remove a stretch of setup time from the budget.
  void exclude(typename Clock::duration setup) { excluded_ += setup; }

  // Milliseconds of the budget used so far.
  int64_t elapsedMs(typename Clock::time_point now) const {
    const auto used = now - start_ - excluded_;
    if (used.count() <= 0) return 0;
    return std::chrono::duration_cast<std::chrono::milliseconds>(used).count();
  }

  // Milliseconds of setup excluded so far.
  int64_t excludedMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(excluded_).count();
  }

private:
  typename Clock::time_point start_;
  typename Clock::duration excluded_{};
};
