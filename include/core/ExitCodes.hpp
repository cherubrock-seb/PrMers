#pragma once
// Process exit codes.
//
//   0    the requested work finished
//   1    the run was stopped by the user (SIGINT / Ctrl-C, or SIGTERM / SIGHUP, which stop a run the
//        same way) before it finished. The checkpoint was saved and the worktodo entry is still
//        queued, so the same command line resumes where it stopped. Memtest and P-1 stage 2 already
//        used 1 for a stopped run.
//        1 is also the Gaussian-Mersenne "composite / no factor" result code, so an exit status of 1
//        does not say by itself which of the two happened: check the log or the results file.
//   2    general errors (and the Gaussian-Mersenne error result)
//
// Nothing inside PrMers tells a stop apart from a result by the exit code: modes, restart and
// worktodo queue logic use the sticky stop bit (core::algo::stop_requested_any()). The code only
// leaves the process. It does not apply in GUI mode, where Stop is the normal way to quit.
namespace core {

inline constexpr int kExitInterrupted = 1;

// The code the process exits with: the mode's own result, unless a stop was requested (CLI mode).
inline constexpr int exitCodeForRun(const int modeResult, const bool stopSignalled, const bool guiMode) noexcept {
    return (stopSignalled && !guiMode) ? kExitInterrupted : modeResult;
}

} // namespace core
