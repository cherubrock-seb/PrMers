#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace core::pm1 {

class NoProgressWatchdog {
public:
    using Callback = std::function<void(const std::string&, std::uint64_t)>;

    NoProgressWatchdog(std::chrono::milliseconds timeout, Callback callback)
        : timeout_(timeout), callback_(std::move(callback)),
          last_progress_(std::chrono::steady_clock::now()) {
        if (timeout_.count() > 0) {
            worker_ = std::thread([this] { run(); });
        }
    }

    NoProgressWatchdog(std::uint64_t timeout_seconds, Callback callback)
        : NoProgressWatchdog(std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::seconds(timeout_seconds)),
                             std::move(callback)) {}

    ~NoProgressWatchdog() { stop(); }

    NoProgressWatchdog(const NoProgressWatchdog&) = delete;
    NoProgressWatchdog& operator=(const NoProgressWatchdog&) = delete;

    void set_phase(std::string phase) {
        if (timeout_.count() <= 0) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            phase_ = std::move(phase);
        }
        cv_.notify_all();
    }

    void advance(std::string phase) {
        if (timeout_.count() <= 0) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            phase_ = std::move(phase);
            ++progress_;
            last_progress_ = std::chrono::steady_clock::now();
        }
        cv_.notify_all();
    }

    std::uint64_t progress() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return progress_;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        std::uint64_t seen = progress_;

        while (!stopping_) {
            const auto deadline = last_progress_ + timeout_;
            const bool changed = cv_.wait_until(lock, deadline, [&] {
                return stopping_ || progress_ != seen;
            });
            if (stopping_) return;
            if (changed) {
                seen = progress_;
                continue;
            }

            const std::string phase = phase_;
            const std::uint64_t progress = progress_;
            lock.unlock();
            callback_(phase, progress);
            return;
        }
    }

    const std::chrono::milliseconds timeout_;
    Callback callback_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::uint64_t progress_ = 0;
    std::string phase_ = "stage2-start";
    std::chrono::steady_clock::time_point last_progress_;
    std::thread worker_;
};

}  // namespace core::pm1
