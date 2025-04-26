// include/util/Timer.hpp
#pragma once
#include <chrono>

namespace util {

class Timer {
public:
    Timer();
    void start();
    double elapsed() const;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

} // namespace util
