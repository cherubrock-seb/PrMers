// The plan search's time budget counts measurement, not setup (tuning context, engine builds, kernel compiles).
#include "TuneBudget.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace std::chrono;
using Clock = steady_clock;

static int failures = 0;
static void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

// The search loop of Runtime: check the budget before each candidate, then build (setup) and measure it.
// Returns how many candidates were measured.
static unsigned search(bool exclude_setup, milliseconds first_build, milliseconds build, milliseconds measure,
                       int64_t budget_ms, unsigned cap) {
  Clock::time_point now{};
  TuneBudget<Clock> budget(now);
  unsigned tested = 0;
  for (unsigned i = 0; i < cap; ++i) {
    const int64_t elapsed = exclude_setup ? budget.elapsedMs(now)
                                          : duration_cast<milliseconds>(now - Clock::time_point{}).count();
    if (elapsed >= budget_ms) break;
    const auto setup = i == 0 ? first_build : build;
    now += setup;
    if (exclude_setup) budget.exclude(setup);
    now += measure;
    ++tested;
  }
  return tested;
}

int main() {
  {
    TuneBudget<Clock> b(Clock::time_point{});
    check(b.elapsedMs(Clock::time_point{} + milliseconds(1234)) == 1234, "elapsed without exclusions is wall time");
    b.exclude(milliseconds(1000));
    check(b.elapsedMs(Clock::time_point{} + milliseconds(1234)) == 234, "excluded setup is not counted");
    check(b.excludedMs() == 1000, "excludedMs reports the setup");
    b.exclude(milliseconds(5000));
    check(b.elapsedMs(Clock::time_point{} + milliseconds(1234)) == 0, "elapsed never goes negative");
  }
  // A cold start: the first candidate compiles both engines (12 s), later ones only their own kernels (3 s); each
  // measurement takes 1.5 s; default budget 7 s, cap 5.  Counting setup measures one candidate; counting only
  // measurement measures all five.
  const unsigned wall = search(false, milliseconds(12000), milliseconds(3000), milliseconds(1500), 7000, 5);
  const unsigned measured = search(true, milliseconds(12000), milliseconds(3000), milliseconds(1500), 7000, 5);
  check(wall == 1, "model check: a wall-clock budget measures one candidate on a cold start");
  check(measured == 5, "a measurement budget measures every candidate up to the cap on a cold start");
  // A warm cache: builds are cheap, measurements slow (4 s): the budget still stops the search.
  check(search(true, milliseconds(100), milliseconds(100), milliseconds(4000), 7000, 5) == 2,
        "the budget still bounds the measurement time");
  if (failures) return 1;
  std::puts("tune_budget_test: OK");
  return 0;
}
