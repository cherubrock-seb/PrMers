#include "core/Pm1Progress.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

int main() {
    using core::pm1::stage2_required;
    // Two-stage worktodo: Stage 1 must not yield an NF submission.
    assert(stage2_required(1600000, 325000000, false, false));
    assert(stage2_required(2000000, 450000000, false, false));
    // A newly discovered Stage-1 factor completes the job by default.
    assert(!stage2_required(1600000, 325000000, true, false));
    assert(stage2_required(1600000, 325000000, true, true));
    assert(!stage2_required(1600000, 1600000, false, false));

    using namespace std::chrono;
    const auto begin = steady_clock::time_point{};
    assert(!core::pm1::progress_due(begin + seconds(3), begin, 300));
    assert(!core::pm1::progress_due(begin + seconds(299), begin, 300));
    assert(core::pm1::progress_due(begin + seconds(300), begin, 300));
    assert(!core::pm1::progress_due(begin + hours(4), begin, 0));
    assert(core::pm1::progress_due(begin + seconds(3), begin, 3));

    const std::string line = core::pm1::progress_prefix(332746979, 2);
    assert(line.find("[P-1 stage 2 p=332746979]") != std::string::npos);
    assert(line.size() > 35 && line[0] == '[' && line[5] == '-');
    assert(line.find('\r') == std::string::npos);
    std::cout << "PM1_PROGRESS_REPORT_POLICY=PASS\n";
}
