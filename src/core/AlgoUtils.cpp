// AlgoUtils.cpp
#include "core/AlgoUtils.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>
#include "core/StopRestartGate.hpp"
#include "core/InheritedSignals.hpp"
#ifdef _WIN32
# include <windows.h>
#else
# include <signal.h>
# include <unistd.h>
#endif

namespace core { namespace algo {
  std::atomic<bool> interrupted{false};

  void handle_sigint(int) noexcept {
    interrupted.store(true, std::memory_order_relaxed);
    // Also a Stop for the restart logic: the GUI idle loop must not restart the queue after it, and
    // a restart already committed must not go ahead.
    core::stop_or_exit();
  }

  bool stop_requested_any() noexcept {
    return core::g_stop_restart_gate.stopRequested();
  }

#ifdef _WIN32
  void note_inherited_signals() noexcept {}
  bool sighup_inherited_ignored() noexcept { return false; }

  static BOOL WINAPI stopCtrlHandler(DWORD type) {
    switch (type) {
      case CTRL_C_EVENT:
      case CTRL_BREAK_EVENT:
        handle_sigint(SIGINT);
        return TRUE;
      case CTRL_CLOSE_EVENT:
      case CTRL_LOGOFF_EVENT:
      case CTRL_SHUTDOWN_EVENT:
        handle_sigint(SIGTERM);
        // Windows ends the process when this handler returns (after at most ~5 s for a close). Keep the
        // console thread here so the main thread can save its checkpoint and leave by itself.
        for (int i = 0; i < 100; ++i) Sleep(50);
        return TRUE;
      default:
        return FALSE;
    }
  }

  void install_stop_handlers() {
    static std::atomic<bool> done{false};
    if (done.exchange(true)) return;
    std::signal(SIGINT, handle_sigint);
    SetConsoleCtrlHandler(stopCtrlHandler, TRUE);
  }
#else
  static std::atomic<bool> g_hupInheritedIgnored{false};

  void note_inherited_signals() noexcept {
    struct sigaction old;
    if (sigaction(SIGHUP, nullptr, &old) == 0 && old.sa_handler == SIG_IGN)
      g_hupInheritedIgnored.store(true, std::memory_order_relaxed);
  }

  bool sighup_inherited_ignored() noexcept { return g_hupInheritedIgnored.load(std::memory_order_relaxed); }

  static void installOne(const int sig, const bool keepIgnored) {
    struct sigaction old;
    if (keepIgnored && g_hupInheritedIgnored.load(std::memory_order_relaxed)) return;
    if (sigaction(sig, nullptr, &old) == 0 && keepIgnored && old.sa_handler == SIG_IGN) return;
    struct sigaction sa;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;   // as std::signal(SIGINT, ...) has always behaved
    sa.sa_handler = handle_sigint;
    sigaction(sig, &sa, nullptr);
  }

  void install_stop_handlers() {
    installOne(SIGINT, false);
    installOne(SIGTERM, false);
    installOne(SIGHUP, true);   // `nohup prmers ...` must keep surviving the terminal closing
  }
#endif
}}
