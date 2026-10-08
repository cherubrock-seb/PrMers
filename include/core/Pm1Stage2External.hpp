#ifndef CORE_PM1_STAGE2_EXTERNAL_HPP
#define CORE_PM1_STAGE2_EXTERNAL_HPP

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <system_error>
#include <thread>
#ifndef _WIN32
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace core {

/// What a P-1 run does once it has tried the external Prime95 stage 2.
enum class Pm1AfterExternalStage2 {
    Done,         // Prime95 searched the range; the internal stage 2 is skipped.
    RunInternal,  // Prime95 was not usable or failed; fall back to the internal stage 2.
    Interrupted   // The user stopped the run; keep the checkpoint and the worktodo line.
};

/// The internal stage 2 is only a fallback for a real Prime95 failure.  An
/// interrupt that arrived while Prime95 ran (or while its result was awaited)
/// must stop the run: starting the internal stage 2 would only run its setup
/// before it noticed the interrupt.
inline Pm1AfterExternalStage2 pm1AfterExternalStage2(bool externalUsed, bool interrupted) {
    if (externalUsed) return Pm1AfterExternalStage2::Done;
    if (interrupted) return Pm1AfterExternalStage2::Interrupted;
    return Pm1AfterExternalStage2::RunInternal;
}

/// True when the status returned by std::system() says the shell or Prime95
/// was ended by SIGINT/SIGTERM, either directly (the shell was signalled) or as
/// the shell's 128+signal exit status for a signalled child.  std::system()
/// ignores SIGINT in the calling process while the child runs, so the
/// process-wide interrupt flag is never set by a Ctrl-C during the Prime95
/// run; the child's fate is the only sign.
inline bool pm1SystemStatusInterrupted(int status) {
#ifdef _WIN32
    (void)status;
    return false;
#else
    if (status == -1) return false;
    if (WIFSIGNALED(status)) return WTERMSIG(status) == SIGINT || WTERMSIG(status) == SIGTERM;
    if (WIFEXITED(status)) return WEXITSTATUS(status) == 128 + SIGINT || WEXITSTATUS(status) == 128 + SIGTERM;
    return false;
#endif
}

/// The Prime95 state file `m<p>` is written from PrMers' stage-1 residue before
/// the handoff, and Prime95 then saves its own stage-2 progress into that same
/// file.  After an interrupt, rewriting it on the rerun would throw that
/// progress away.  A small marker next to the state file records the bounds of
/// the handoff in flight: while it exists and matches, the state file is kept.
inline std::string pm1Prime95HandoffKey(uint64_t b1, uint64_t b2, uint64_t b2Start) {
    return "B1=" + std::to_string(b1) + " B2=" + std::to_string(b2) + " B2Start=" + std::to_string(b2Start);
}

inline std::string pm1Prime95HandoffMarkerPath(const std::string& statePath) {
    return statePath + ".prmers";
}

/// True when an earlier handoff with the same bounds was interrupted, so the
/// state file holds Prime95's own progress and must not be rewritten.
inline bool pm1Prime95HandoffPending(const std::string& statePath, const std::string& key) {
    std::error_code ec;
    if (!std::filesystem::exists(statePath, ec)) return false;
    std::ifstream in(pm1Prime95HandoffMarkerPath(statePath));
    if (!in) return false;
    std::string stored;
    std::getline(in, stored);
    return stored == key;
}

/// Records a handoff that is about to start (after the state file was written).
inline void pm1Prime95HandoffBegin(const std::string& statePath, const std::string& key) {
    std::ofstream out(pm1Prime95HandoffMarkerPath(statePath), std::ios::trunc);
    out << key << '\n';
}

/// Forgets the handoff: Prime95 finished or failed, so the next one starts from
/// a freshly written state file.
inline void pm1Prime95HandoffEnd(const std::string& statePath) {
    std::error_code ec;
    std::filesystem::remove(pm1Prime95HandoffMarkerPath(statePath), ec);
}

#ifndef _WIN32
/// Runs `sh -lc <script>` and waits for it, polling every 100 ms.  Unlike
/// std::system(), the calling process keeps its own SIGINT/SIGTERM handlers
/// while the child runs, so a Ctrl-C reaches the process-wide interrupt flag
/// (`stop`).  When `stop` is set the child gets SIGTERM once (Prime95 stops its
/// workers gracefully and exits normally, so its exit status cannot tell an
/// interrupt from a finished run) and SIGKILL after `grace_seconds`; the child
/// is always reaped.  `script` should end in `exec <program> ...` so the shell
/// is replaced by the program and the signal reaches it.  `tick` is called on
/// every poll.  Returns the raw wait status, or -1 if the shell cannot start.
inline int pm1RunShellInterruptible(const std::string& script,
                                    const std::atomic<bool>& stop,
                                    const std::function<void()>& tick,
                                    bool& interrupted,
                                    int grace_seconds = 120) {
    interrupted = false;
    const char* argv[] = {"sh", "-lc", script.c_str(), nullptr};
    pid_t pid = 0;
    if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, const_cast<char* const*>(argv), environ) != 0) return -1;
    bool term_sent = false;
    auto term_at = std::chrono::steady_clock::now();
    bool killed = false;
    for (;;) {
        int status = 0;
        const pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) return status;
        if (r < 0 && errno != EINTR) return -1;
        if (stop.load(std::memory_order_relaxed)) {
            interrupted = true;
            if (!term_sent) {
                kill(pid, SIGTERM);
                term_sent = true;
                term_at = std::chrono::steady_clock::now();
            } else if (!killed &&
                       std::chrono::steady_clock::now() - term_at > std::chrono::seconds(grace_seconds)) {
                kill(pid, SIGKILL);
                killed = true;
            }
        }
        if (tick) tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
#else
/// Windows counterpart of pm1RunShellInterruptible: waits for an already started process, polling
/// every 100 ms, and reports through `interrupted` whether `stop` (the process-wide interrupt flag)
/// was set while it ran.  A Ctrl-C typed in the console reaches Prime95 as well and ends it, but
/// nothing tells the wait that the process ended *because of* the interrupt, so the flag is the
/// only sign; it is also checked once more after the process has exited, because the interrupt may
/// arrive between the last poll and the exit.  Windows has no SIGTERM to send, so a process that
/// is still running `grace_seconds` after the interrupt was seen is terminated.  `tick` is called
/// on every poll.  Returns the exit code, or -1 if the wait failed or the code is unavailable.
inline int pm1WaitProcessInterruptible(HANDLE process,
                                       const std::atomic<bool>& stop,
                                       const std::function<void()>& tick,
                                       bool& interrupted,
                                       int grace_seconds = 120) {
    interrupted = false;
    bool stop_seen = false;
    bool killed = false;
    auto stop_at = std::chrono::steady_clock::now();
    for (;;) {
        const DWORD w = WaitForSingleObject(process, 100);
        if (w == WAIT_OBJECT_0) break;
        if (w != WAIT_TIMEOUT) return -1;
        if (stop.load(std::memory_order_relaxed)) {
            interrupted = true;
            if (!stop_seen) {
                stop_seen = true;
                stop_at = std::chrono::steady_clock::now();
            } else if (!killed &&
                       std::chrono::steady_clock::now() - stop_at > std::chrono::seconds(grace_seconds)) {
                TerminateProcess(process, 1);
                killed = true;
            }
        }
        if (tick) tick();
    }
    if (stop.load(std::memory_order_relaxed)) interrupted = true;
    DWORD code = 0;
    return GetExitCodeProcess(process, &code) ? static_cast<int>(code) : -1;
}
#endif

} // namespace core

#endif
