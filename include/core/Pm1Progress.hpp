#ifndef CORE_PM1_PROGRESS_HPP
#define CORE_PM1_PROGRESS_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <string>

namespace core::pm1 {

// A Stage-1 factor normally completes the assignment even if B2 was provided.
// Otherwise only the completed Stage-2 result should be submitted for a two-stage job.
inline bool stage2_required(std::uint64_t b1, std::uint64_t b2,
                            bool new_stage1_factor, bool continue_after_factor) {
    return b2 > b1 && (!new_stage1_factor || continue_after_factor);
}

// Periodic diagnostics are observational only: never use this gate for a
// checkpoint, error check, interrupt poll, or stage-transition decision.
template<class Clock, class Duration>
inline bool progress_due(const std::chrono::time_point<Clock, Duration>& now,
                         const std::chrono::time_point<Clock, Duration>& previous,
                         std::uint64_t interval_seconds) {
    return interval_seconds != 0 &&
           now - previous >= std::chrono::seconds(std::min<std::uint64_t>(interval_seconds, 86400));
}

inline std::string progress_prefix(std::uint64_t exponent, unsigned stage) {
    const std::time_t t = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &t);
#else
    localtime_r(&t, &local);
#endif
    char stamp[32] = {};
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &local);
    return std::string("[") + stamp + "] [P-1 stage " + std::to_string(stage) +
           " p=" + std::to_string(exponent) + "] ";
}

} // namespace core::pm1
#endif
