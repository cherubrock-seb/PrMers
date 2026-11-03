#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace core::bench {

// A complete first-use probe is discarded, including its correctness check.
// Retain one real median sample so its boundary/full-check components still
// reconstruct the reported production estimate.
inline constexpr std::size_t kProductionProbeRepetitions = 3;
inline constexpr const char* kProductionTimingMethod = "warm-gerbicz-v1";

struct ProductionProbeStats {
    std::size_t repetitions = 0;
    double boundary_min_us = 0.0;
    double boundary_max_us = 0.0;
    double full_check_min_us = 0.0;
    double full_check_max_us = 0.0;
    double production_min_us = 0.0;
    double production_max_us = 0.0;
};

template<class Sample>
struct ProductionProbeResult {
    Sample median;
    ProductionProbeStats stats;
};

// probe(0) warms every operation; probe(1..3) must each reconstruct the same
// fresh register state. Exceptions and failed exactness checks invalidate the
// whole measurement, including failures in the discarded warmup.
template<class Probe>
auto sample_production_probe(Probe&& probe) {
    using Sample = std::decay_t<decltype(probe(std::size_t{}))>;
    auto checked = [&](std::size_t trial) {
        Sample sample = probe(trial);
        if (!sample.exact)
            throw std::runtime_error("production probe exactness failure");
        if (!std::isfinite(sample.boundary_us) || sample.boundary_us < 0.0 ||
            !std::isfinite(sample.full_check_us) || sample.full_check_us < 0.0 ||
            !std::isfinite(sample.production_us_per_iter) ||
            sample.production_us_per_iter <= 0.0)
            throw std::runtime_error("invalid production probe timing");
        return sample;
    };
    (void)checked(0);
    std::array<Sample, kProductionProbeRepetitions> samples;
    for (std::size_t i = 0; i < samples.size(); ++i)
        samples[i] = checked(i + 1);

    std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
        return a.production_us_per_iter < b.production_us_per_iter;
    });
    ProductionProbeStats stats;
    stats.repetitions = samples.size();
    stats.boundary_min_us = stats.boundary_max_us = samples[0].boundary_us;
    stats.full_check_min_us = stats.full_check_max_us = samples[0].full_check_us;
    stats.production_min_us = samples.front().production_us_per_iter;
    stats.production_max_us = samples.back().production_us_per_iter;
    for (const auto& sample : samples) {
        stats.boundary_min_us = std::min(stats.boundary_min_us, sample.boundary_us);
        stats.boundary_max_us = std::max(stats.boundary_max_us, sample.boundary_us);
        stats.full_check_min_us = std::min(stats.full_check_min_us, sample.full_check_us);
        stats.full_check_max_us = std::max(stats.full_check_max_us, sample.full_check_us);
    }
    return ProductionProbeResult<Sample>{samples[samples.size() / 2], stats};
}

} // namespace core::bench
