#pragma once
// Process exit codes.
//
//   0    the requested work finished
//   1,2  Gaussian-Mersenne result codes (1 = composite / no factor) and general errors
//   130  the run was stopped by the user (SIGINT / Ctrl-C, or SIGTERM / SIGHUP, which stop a run the
//        same way) before it finished: 128 + SIGINT, the status a shell reports for a process killed by
//        Ctrl-C. The checkpoint was saved and the worktodo entry is still queued, so the same command
//        line resumes where it stopped.
//
// 130 is distinct from the Gaussian-Mersenne result codes, so a script can tell "stopped, resume me"
// apart from "finished". It does not apply in GUI mode, where Stop is the normal way to quit.
namespace core {

inline constexpr int kExitInterrupted = 130;

// The code the process exits with: the mode's own result, unless a stop was requested (CLI mode).
inline constexpr int exitCodeForRun(const int modeResult, const bool stopSignalled, const bool guiMode) noexcept {
    return (stopSignalled && !guiMode) ? kExitInterrupted : modeResult;
}

} // namespace core
