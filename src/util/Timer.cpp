// src/util/Timer.cpp
#include "util/Timer.hpp"

namespace util {

Timer::Timer() {
    start();
}

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start_time).count();
}

} // namespace util
