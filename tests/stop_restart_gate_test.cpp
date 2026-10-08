// A Stop (signal or GUI) must never be lost to a restart: either the Stop comes first and the restart
// does not happen, or the restart was already committed and the Stop ends the process before the
// relaunch. Drives core::StopRestartGate through every interleaving of the GUI idle restart.
#include "core/StopRestartGate.hpp"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#if !defined(_WIN32)
# include <sys/wait.h>
# include <unistd.h>
#endif

using core::StopRestartGate;
using Claim = StopRestartGate::Claim;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

// The idle loop body of App::run: loop condition, then claim, then (stop GUI and) restart_self's commit.
// `stopAt` injects a Stop at one step. Returns true if the process would exec/relaunch.
enum class Step { None, AfterLoopCheck, AfterClaim, AfterGuiStop, AfterCommit };
struct Outcome { bool loopEntered = false; bool relaunched = false; bool exitedOnStop = false; bool stillPending = false; };

static Outcome idle_iteration(Step stopAt, bool appended) {
    StopRestartGate g;
    Outcome o;
    if (appended) g.markAppendPending();
    if (g.stopRequested()) return o;           // while (!stop_requested()) { ...
    o.loopEntered = true;
    if (stopAt == Step::AfterLoopCheck) o.exitedOnStop |= g.requestStop();
    if (g.claimAppendedEntry() != Claim::Restart) { o.stillPending = g.appendPending(); return o; }
    if (stopAt == Step::AfterClaim) o.exitedOnStop |= g.requestStop();
    // gui->stop()
    if (stopAt == Step::AfterGuiStop) o.exitedOnStop |= g.requestStop();
    if (!g.commitRestart()) { o.stillPending = g.appendPending(); return o; }  // restart_self returns
    if (stopAt == Step::AfterCommit) {
        // The stop side sees the commit and must end the process (stop_or_exit) before the exec.
        o.exitedOnStop |= g.requestStop();
        if (o.exitedOnStop) return o;
    }
    o.relaunched = true;
    return o;
}

static void test_interleavings() {
    // No append: never restarts.
    {
        Outcome o = idle_iteration(Step::None, false);
        CHECK(o.loopEntered && !o.relaunched);
    }
    // Append, no Stop: restarts.
    {
        Outcome o = idle_iteration(Step::None, true);
        CHECK(o.relaunched && !o.exitedOnStop);
    }
    // The reported interleaving: the loop condition saw no Stop, then Stop ran, then the helper ran.
    {
        Outcome o = idle_iteration(Step::AfterLoopCheck, true);
        CHECK(!o.relaunched);
        CHECK(!o.exitedOnStop);          // nothing committed yet: plain stop, loop exits
        CHECK(o.stillPending);           // the appended line stays queued
    }
    // Stop after the claim (before or after gui->stop()): restart_self's commit fails.
    for (Step s : {Step::AfterClaim, Step::AfterGuiStop}) {
        Outcome o = idle_iteration(s, true);
        CHECK(!o.relaunched);
        CHECK(!o.exitedOnStop);
    }
    // Stop after the commit: the Stop wins by ending the process before the relaunch.
    {
        Outcome o = idle_iteration(Step::AfterCommit, true);
        CHECK(!o.relaunched);
        CHECK(o.exitedOnStop);
    }
}

static void test_state_rules() {
    // Double append collapses into one restart.
    {
        StopRestartGate g;
        g.markAppendPending();
        g.markAppendPending();
        CHECK(g.claimAppendedEntry() == Claim::Restart);
        CHECK(g.claimAppendedEntry() == Claim::None);
    }
    // Append after Stop: never claimed, stays pending (it is in worktodo for the next start).
    {
        StopRestartGate g;
        CHECK(!g.requestStop());
        g.markAppendPending();
        CHECK(g.claimAppendedEntry() == Claim::Stopped);
        CHECK(g.appendPending());
        CHECK(!g.commitRestart());
    }
    // Append while a restart is in progress (after the claim): the commit still goes ahead.
    {
        StopRestartGate g;
        g.markAppendPending();
        CHECK(g.claimAppendedEntry() == Claim::Restart);
        g.markAppendPending();
        CHECK(g.commitRestart());
    }
    // SIGINT and GUI Stop together (or twice): idempotent, the stop sticks.
    {
        StopRestartGate g;
        CHECK(!g.requestStop());
        CHECK(!g.requestStop());
        CHECK(g.stopRequested());
        CHECK(!g.commitRestart());
        CHECK(g.requestStop() == false);
    }
    // End-of-entry restart (no append) commits unless stopped; commit is idempotent.
    {
        StopRestartGate g;
        CHECK(g.commitRestart());
        CHECK(g.commitRestart());
        CHECK(g.requestStop());  // a later Stop must end the process
    }
}

// Real threads: a Stop racing the claim/commit sequence. Exactly one wins every time.
static void test_race() {
    for (int i = 0; i < 20000; ++i) {
        StopRestartGate g;
        g.markAppendPending();
        std::atomic<bool> go{false};
        bool stopSawCommit = false;
        std::thread stopper([&] {
            while (!go.load(std::memory_order_acquire)) {}
            stopSawCommit = g.requestStop();
        });
        go.store(true, std::memory_order_release);
        bool committed = false;
        if (g.claimAppendedEntry() == Claim::Restart) committed = g.commitRestart();
        stopper.join();
        // committed && !stopSawCommit would mean a relaunch that ignored a Stop.
        CHECK(committed == stopSawCommit);
        if (!committed) CHECK(g.stopRequested());
    }
}

#if !defined(_WIN32)
static void on_signal(int) { core::stop_or_exit(); }

// Child process: install the real handler, commit a restart, then get a signal before the "exec".
// stop_or_exit must _Exit(kExitInterrupted) so the marker (standing in for the relaunch) is never printed.
static int run_child(bool commitFirst, bool viaThread) {
    struct sigaction sa; std::memset(&sa, 0, sizeof sa); sa.sa_handler = on_signal; sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    if (commitFirst && !core::g_stop_restart_gate.commitRestart()) return 10;
    if (viaThread) {
        // GUI Stop: runs on another thread (the HTTP worker), calls the handler directly.
        std::thread t([] { on_signal(SIGINT); });
        t.join();
    } else {
        std::raise(SIGINT);
    }
    if (commitFirst) {
        std::printf("RELAUNCHED\n");
        return 11;
    }
    // Stop before the commit: the commit must now fail and the process carries on (restart_self returns).
    if (core::g_stop_restart_gate.commitRestart()) { std::printf("RELAUNCHED\n"); return 12; }
    return 0;
}

static void test_signal_paths() {
    for (int mode = 0; mode < 4; ++mode) {
        const bool commitFirst = (mode & 1) != 0, viaThread = (mode & 2) != 0;
        int pfd[2];
        if (pipe(pfd) != 0) { CHECK(false); return; }
        std::fflush(nullptr);
        pid_t pid = fork();
        if (pid == 0) {
            dup2(pfd[1], 1); close(pfd[0]); close(pfd[1]);
            int rc = run_child(commitFirst, viaThread);
            std::fflush(nullptr);
            std::_Exit(rc);
        }
        close(pfd[1]);
        char buf[64] = {0};
        ssize_t n = read(pfd[0], buf, sizeof buf - 1);
        close(pfd[0]);
        int st = 0;
        waitpid(pid, &st, 0);
        CHECK(WIFEXITED(st));
        // Stop after the restart was committed: the process ends with the interrupted code; a stop before
        // the commit makes the commit fail and the child carries on (exit 0 from run_child).
        CHECK(WIFEXITED(st) && WEXITSTATUS(st) == (commitFirst ? core::kExitInterrupted : 0));
        CHECK(n <= 0 || std::strstr(buf, "RELAUNCHED") == nullptr);
    }
}
#endif

int main() {
    test_interleavings();
    test_state_rules();
    test_race();
#if !defined(_WIN32)
    test_signal_paths();
#endif
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("stop/restart gate test passed");
    return 0;
}
