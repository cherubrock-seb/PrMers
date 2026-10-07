#ifndef CORE_PM1_STAGE2_EXTERNAL_HPP
#define CORE_PM1_STAGE2_EXTERNAL_HPP

#include <cstdlib>
#ifndef _WIN32
#include <csignal>
#include <sys/wait.h>
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

} // namespace core

#endif
