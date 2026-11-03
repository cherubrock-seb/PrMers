#include "core/ProductionProbe.hpp"

#include <array>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

struct Sample {
    double boundary_us = 0;
    double full_check_us = 0;
    double production_us_per_iter = 0;
    bool exact = true;
    int identity = 0;
};

int main() {
    // The cold outlier must affect neither the median nor any reported range.
    // Each production estimate is a paired boundary/full-check measurement.
    const std::array<Sample, 4> input{{
        {9000, 8000, 17001, true, 0},
        {8, 1, 10, true, 1},
        {2, 4, 7, true, 2},
        {1, 7, 9, true, 3}
    }};
    std::vector<std::size_t> calls;
    const auto result = core::bench::sample_production_probe([&](std::size_t trial) {
        calls.push_back(trial);
        return input.at(trial);
    });
    assert((calls == std::vector<std::size_t>{0, 1, 2, 3}));
    assert(result.median.identity == 3);
    assert(result.median.production_us_per_iter ==
           1 + result.median.boundary_us + result.median.full_check_us);
    assert(result.stats.repetitions == 3);
    assert(result.stats.boundary_min_us == 1 && result.stats.boundary_max_us == 8);
    assert(result.stats.full_check_min_us == 1 && result.stats.full_check_max_us == 7);
    assert(result.stats.production_min_us == 7 && result.stats.production_max_us == 10);

    // Exactness failure, non-finite clocks and cancellation must invalidate
    // the result at any phase, including the excluded first-use probe.
    for (std::size_t failing = 0; failing < input.size(); ++failing) {
        for (int kind = 0; kind < 4; ++kind) {
            std::size_t count = 0;
            bool threw = false;
            try {
                (void)core::bench::sample_production_probe([&](std::size_t trial) {
                    ++count;
                    auto value = input.at(trial);
                    if (trial == failing) {
                        if (kind == 0) value.exact = false;
                        if (kind == 1) value.boundary_us = std::numeric_limits<double>::infinity();
                        if (kind == 2) value.production_us_per_iter = 0;
                        if (kind == 3) throw std::runtime_error("interrupted");
                    }
                    return value;
                });
            } catch (const std::runtime_error&) {
                threw = true;
            }
            assert(threw);
            assert(count == failing + 1);
        }
    }
}
