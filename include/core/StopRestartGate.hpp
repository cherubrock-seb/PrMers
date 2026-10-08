#pragma once
// Orders a user Stop (Ctrl-C, SIGTERM/SIGHUP, the GUI's Stop button) against a self-restart
// (restart_self after a finished worktodo entry, or after a GUI "Append & Run" while idle).
//
// Everything lives in one lock-free atomic word, so the stop side can run inside a signal handler.
// A restart is committed with a compare-and-swap that fails once Stop is set, and Stop is a fetch_or
// that reports whether a restart had already been committed. Exactly one of them wins:
//   - Stop first:   the commit fails, the process does not restart, the worktodo is left as it is
//                   (an appended line stays queued for the next start).
//   - Commit first: the stop side ends the process at once (stop_or_exit), so the relaunch that
//                   was about to happen never does. A Stop after the exec reaches the new process.
#include <atomic>
#include <cstdlib>
#if !defined(_WIN32)
# include <unistd.h>
#endif

namespace core {

class StopRestartGate {
public:
    static constexpr unsigned kStop = 1u;
    static constexpr unsigned kAppendPending = 2u;
    static constexpr unsigned kRestartCommitted = 4u;

    enum class Claim { None, Stopped, Restart };

    constexpr StopRestartGate() noexcept = default;

    // Async-signal-safe. Returns true when a restart was already committed: the caller must then end
    // the process before the relaunch happens (stop_or_exit does that).
    bool requestStop() noexcept {
        return (state_.fetch_or(kStop, std::memory_order_acq_rel) & kRestartCommitted) != 0;
    }

    bool stopRequested() const noexcept {
        return (state_.load(std::memory_order_acquire) & kStop) != 0;
    }

    // A GUI "Append & Run" wrote a line to worktodo (call only after the write succeeded).
    void markAppendPending() noexcept { state_.fetch_or(kAppendPending, std::memory_order_acq_rel); }

    bool appendPending() const noexcept {
        return (state_.load(std::memory_order_acquire) & kAppendPending) != 0;
    }

    // Main thread, while idle: take the pending append unless Stop is set. A Stop leaves the pending
    // bit alone (the line is already in worktodo; only the restart is dropped).
    Claim claimAppendedEntry() noexcept {
        unsigned cur = state_.load(std::memory_order_acquire);
        for (;;) {
            if (cur & kStop) return Claim::Stopped;
            if (!(cur & kAppendPending)) return Claim::None;
            if (state_.compare_exchange_weak(cur, cur & ~kAppendPending,
                                             std::memory_order_acq_rel, std::memory_order_acquire))
                return Claim::Restart;
        }
    }

    // The point of no return before exec/CreateProcess. Fails (returns false) iff Stop is set.
    bool commitRestart() noexcept {
        unsigned cur = state_.load(std::memory_order_acquire);
        for (;;) {
            if (cur & kStop) return false;
            if (state_.compare_exchange_weak(cur, cur | kRestartCommitted,
                                             std::memory_order_acq_rel, std::memory_order_acquire))
                return true;
        }
    }

private:
    std::atomic<unsigned> state_{0};
};

static_assert(std::atomic<unsigned>::is_always_lock_free,
              "StopRestartGate is used from signal handlers and needs a lock-free atomic");

// The process-wide gate. Constant-initialised, so a signal handler may use it at any time.
inline StopRestartGate g_stop_restart_gate;

// Stop request from a signal handler or the GUI's Stop. Async-signal-safe. If a restart is already
// committed, end the process now instead of letting it relaunch.
inline void stop_or_exit(StopRestartGate& gate = g_stop_restart_gate) noexcept {
    if (!gate.requestStop()) return;
#if !defined(_WIN32)
    static const char msg[] = "\nStop requested during restart; exiting without restarting.\n";
    ssize_t r = ::write(2, msg, sizeof msg - 1);
    (void)r;
#endif
    std::_Exit(0);
}

} // namespace core
