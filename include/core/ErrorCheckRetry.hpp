#pragma once

#include <cstdint>
#include <string>

namespace core {

// Counts consecutive failed error checks (Gerbicz-Li or LL block checks) that
// each roll back to the same verified state. A random fault disappears after
// one rollback, so repeating the same failure means the transform or plan is
// deterministically wrong and retrying would loop forever.
class ErrorCheckRetry {
public:
    static constexpr uint32_t kDefaultLimit = 3;

    explicit ErrorCheckRetry(uint32_t limit = kDefaultLimit) : limit_(limit) {}

    // Records a failed check. Returns true when the limit is reached and the
    // run must stop instead of rolling back again.
    bool failed() { return ++streak_ >= limit_; }

    // Records a passed check: the verified state moved forward.
    void passed() { streak_ = 0; }

    uint32_t streak() const { return streak_; }

    std::string reason(uint64_t verified_iter, const std::string& hint) const {
        return "Error check failed " + std::to_string(streak_) +
               " times in a row from verified iteration " +
               std::to_string(verified_iter) +
               "; retrying the same state will not help. " + hint;
    }

private:
    uint32_t limit_;
    uint32_t streak_ = 0;
};

} // namespace core
