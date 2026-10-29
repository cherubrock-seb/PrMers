#include "core/Pm1NoProgressWatchdog.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <string>
#include <thread>

int main() {
    using namespace std::chrono_literals;

    {
        std::atomic<bool> fired{false};
        core::pm1::NoProgressWatchdog disabled(
            std::chrono::milliseconds(0),
            [&](const std::string&, std::uint64_t) { fired = true; });
        disabled.set_phase("disabled");
        disabled.advance("disabled");
        std::this_thread::sleep_for(30ms);
        assert(!fired.load());
    }

    std::atomic<bool> fired{false};
    std::atomic<std::uint64_t> observed_progress{0};
    std::string observed_phase;

    {
        core::pm1::NoProgressWatchdog watchdog(
            120ms,
            [&](const std::string& phase, std::uint64_t progress) {
                observed_phase = phase;
                observed_progress = progress;
                fired = true;
            });

        watchdog.advance("first-item");
        std::this_thread::sleep_for(45ms);
        watchdog.advance("second-item");
        std::this_thread::sleep_for(45ms);
        assert(!fired.load());

        watchdog.set_phase("blocked-gpu-operation");

        for (int i = 0; i < 100 && !fired.load(); ++i) {
            std::this_thread::sleep_for(10ms);
        }

        assert(fired.load());
        assert(observed_phase == "blocked-gpu-operation");
        assert(observed_progress.load() == 2);
    }

    return 0;
}
