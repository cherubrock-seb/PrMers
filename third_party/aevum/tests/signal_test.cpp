// Signal: SIGINT and SIGTERM both request a graceful stop; a second Ctrl-C ends the process at once.
// POSIX only (uses fork); on Windows the test is a no-op, as the console never delivers SIGTERM there.

#include "Signal.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
int main() { std::puts("signal_test: skipped on Windows"); return 0; }
#else
#include <sys/wait.h>
#include <unistd.h>

static int failures = 0;

static void check(bool ok, const char *what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) { ++failures; }
}

// Run fn in a child process and return its wait status.
template<typename F> static int inChild(F fn) {
  std::fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) { fn(); _exit(0); }
  int status = 0;
  waitpid(pid, &status, 0);
  return status;
}

int main() {
  // SIGTERM is a stop request, not a kill.
  int st = inChild([] {
    Signal sig;
    if (Signal::stopRequested() != 0) { _exit(10); }
    raise(SIGTERM);
    _exit(Signal::stopRequested() == SIGTERM ? 0 : 11);
  });
  check(WIFEXITED(st) && WEXITSTATUS(st) == 0, "SIGTERM sets stopRequested() and the process survives");

  // A single SIGINT is a graceful stop.
  st = inChild([] {
    Signal sig;
    raise(SIGINT);
    _exit(Signal::stopRequested() == SIGINT ? 0 : 11);
  });
  check(WIFEXITED(st) && WEXITSTATUS(st) == 0, "one SIGINT sets stopRequested() and the process survives");

  // A second SIGINT ends the process at once, as SIGINT.
  st = inChild([] {
    Signal sig;
    raise(SIGINT);
    raise(SIGINT);
    _exit(12);   // not reached
  });
  check(WIFSIGNALED(st) && WTERMSIG(st) == SIGINT, "a second SIGINT kills the process");

  // Repeated SIGTERM stays graceful (a supervisor's SIGKILL is the backstop).
  st = inChild([] {
    Signal sig;
    raise(SIGTERM);
    raise(SIGTERM);
    _exit(Signal::stopRequested() == SIGTERM ? 0 : 11);
  });
  check(WIFEXITED(st) && WEXITSTATUS(st) == 0, "repeated SIGTERM does not kill the process");

  // SIGTERM then SIGINT: the SIGINT is the first Ctrl-C, so still graceful.
  st = inChild([] {
    Signal sig;
    raise(SIGTERM);
    raise(SIGINT);
    _exit(Signal::stopRequested() == SIGINT ? 0 : 11);
  });
  check(WIFEXITED(st) && WEXITSTATUS(st) == 0, "SIGTERM followed by one SIGINT does not kill the process");

  // release() puts both handlers back.
  st = inChild([] {
    Signal sig;
    sig.release();
    raise(SIGTERM);
    _exit(13);   // not reached: default action terminates
  });
  check(WIFSIGNALED(st) && WTERMSIG(st) == SIGTERM, "after release() SIGTERM has its default action again");

  return failures ? 1 : 0;
}
#endif
