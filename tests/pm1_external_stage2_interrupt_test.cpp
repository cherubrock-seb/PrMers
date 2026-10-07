// Host test for the Prime95 stage-2 handoff decisions, using a stub "mprime"
// shell script that is started the same way the P-1 driver starts Prime95.
#include "core/Pm1Stage2External.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

using core::Pm1AfterExternalStage2;
using core::pm1AfterExternalStage2;
using core::pm1SystemStatusInterrupted;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; ++g_fail; } } while (0)

// Runs the stub the way p95_run_pm1_stage2_task does: sh -lc 'cd dir && exe -d > log'.
static int run_stub(const std::string& dir, const std::string& body) {
    const std::string exe = dir + "/mprime";
    { std::ofstream f(exe); f << "#!/bin/sh\n" << body << "\n"; }
    chmod(exe.c_str(), 0755);
    const std::string cmd = "sh -lc 'cd " + dir + " && " + exe + " -d > " + dir + "/log 2>&1'";
    return std::system(cmd.c_str());
}

static std::atomic<bool> g_flag{false};
static void on_sigint(int) { g_flag.store(true); }

static void write_stub(const std::string& dir, const std::string& body) {
    const std::string exe = dir + "/mprime";
    { std::ofstream f(exe); f << "#!/bin/sh\n" << body << "\n"; }
    chmod(exe.c_str(), 0755);
}

static double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";

    // Decision table.
    CHECK(pm1AfterExternalStage2(true,  false) == Pm1AfterExternalStage2::Done);
    CHECK(pm1AfterExternalStage2(true,  true)  == Pm1AfterExternalStage2::Done);
    CHECK(pm1AfterExternalStage2(false, false) == Pm1AfterExternalStage2::RunInternal);
    CHECK(pm1AfterExternalStage2(false, true)  == Pm1AfterExternalStage2::Interrupted);

    // Stub that finishes: not an interrupt.
    int rc = run_stub(dir, "exit 0");
    CHECK(!pm1SystemStatusInterrupted(rc));
    // Stub that fails without a signal: a real Prime95 failure -> internal fallback.
    rc = run_stub(dir, "exit 3");
    CHECK(!pm1SystemStatusInterrupted(rc));
    CHECK(pm1AfterExternalStage2(false, pm1SystemStatusInterrupted(rc)) == Pm1AfterExternalStage2::RunInternal);
    // Stub ended by SIGINT (Ctrl-C reaches the child's process group).
    rc = run_stub(dir, "kill -INT $$; sleep 5");
    CHECK(pm1SystemStatusInterrupted(rc));
    CHECK(pm1AfterExternalStage2(false, pm1SystemStatusInterrupted(rc)) == Pm1AfterExternalStage2::Interrupted);
    // Stub ended by SIGTERM.
    rc = run_stub(dir, "kill -TERM $$; sleep 5");
    CHECK(pm1SystemStatusInterrupted(rc));
    // Stub killed by an unrelated signal is a failure, not an interrupt.
    rc = run_stub(dir, "kill -KILL $$; sleep 5");
    CHECK(!pm1SystemStatusInterrupted(rc));
    CHECK(!pm1SystemStatusInterrupted(-1));

    // Before: std::system() ignores SIGINT in this process while the child runs,
    // so a SIGINT sent to us (Ctrl-C) never reaches the interrupt flag.
    {
        std::signal(SIGINT, on_sigint);
        g_flag = false;
        write_stub(dir, "sleep 1.5");
        std::thread t([] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); kill(getpid(), SIGINT); });
        const int src = run_stub(dir, "sleep 1.5");
        t.join();
        std::cout << "std::system(): SIGINT during the child -> flag=" << g_flag.load()
                  << " status=" << src << " (flag stays 0: the interrupt is lost)\n";
        CHECK(!g_flag.load());
    }

    // After: the runner leaves SIGINT to our handler, forwards SIGTERM to the
    // child and reports the interrupt.  The stub behaves like Prime95: it stops
    // gracefully on SIGTERM/SIGINT and exits 0, so the exit status alone looks
    // like a normal end.
    const std::string exec_stub = "cd " + dir + " && exec " + dir + "/mprime -d > " + dir + "/log 2>&1";
    {
        write_stub(dir, "trap 'exit 0' TERM INT\nwhile :; do sleep 0.1; done");
        g_flag = false;
        std::thread t([] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); kill(getpid(), SIGINT); });
        bool interrupted = false;
        int ticks = 0;
        const auto t0 = std::chrono::steady_clock::now();
        const int st = core::pm1RunShellInterruptible(exec_stub, g_flag, [&] { ++ticks; }, interrupted);
        t.join();
        std::cout << "runner: SIGINT during the child -> interrupted=" << interrupted << " status=" << st
                  << " after " << secs_since(t0) << " s, ticks=" << ticks << "\n";
        CHECK(interrupted);
        CHECK(g_flag.load());
        CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
        CHECK(!pm1SystemStatusInterrupted(st));   // the exit status alone cannot tell
        CHECK(ticks > 0);
        CHECK(secs_since(t0) < 5.0);
        CHECK(pm1AfterExternalStage2(false, interrupted) == Pm1AfterExternalStage2::Interrupted);
    }
    {   // No interrupt: a finishing stub is not reported as interrupted.
        write_stub(dir, "sleep 0.3\nexit 3");
        std::atomic<bool> never{false};
        bool interrupted = true;
        const int st = core::pm1RunShellInterruptible(exec_stub, never, nullptr, interrupted);
        CHECK(!interrupted);
        CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 3);
        CHECK(pm1AfterExternalStage2(false, interrupted) == Pm1AfterExternalStage2::RunInternal);
    }
    {   // A child that ignores SIGTERM is killed after the grace period and reaped.
        write_stub(dir, "trap '' TERM INT\nwhile :; do sleep 0.1; done");
        std::atomic<bool> stop{false};
        std::thread t([&] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); stop = true; });
        bool interrupted = false;
        const auto t0 = std::chrono::steady_clock::now();
        const int st = core::pm1RunShellInterruptible(exec_stub, stop, nullptr, interrupted, 1);
        t.join();
        std::cout << "runner: child ignoring SIGTERM -> status=" << st << " after " << secs_since(t0) << " s\n";
        CHECK(interrupted);
        CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
        CHECK(secs_since(t0) < 8.0);
    }

    {   // The Prime95 state file is kept after an interrupt, rewritten otherwise.
        const std::string state = dir + "/m0000113";
        const std::string key = core::pm1Prime95HandoffKey(10, 20000, 0);
        std::remove(state.c_str());
        core::pm1Prime95HandoffEnd(state);
        CHECK(!core::pm1Prime95HandoffPending(state, key));             // nothing written yet
        { std::ofstream f(state); f << "stage-1 residue"; }
        CHECK(!core::pm1Prime95HandoffPending(state, key));             // no marker: rewrite
        core::pm1Prime95HandoffBegin(state, key);
        CHECK(core::pm1Prime95HandoffPending(state, key));              // interrupted handoff: keep
        CHECK(!core::pm1Prime95HandoffPending(state, core::pm1Prime95HandoffKey(10, 30000, 0)));  // other B2
        CHECK(!core::pm1Prime95HandoffPending(state, core::pm1Prime95HandoffKey(11, 20000, 0)));  // other B1
        CHECK(!core::pm1Prime95HandoffPending(state, core::pm1Prime95HandoffKey(10, 20000, 5000)));  // other B2 start
        std::remove(state.c_str());
        CHECK(!core::pm1Prime95HandoffPending(state, key));             // state file gone: rewrite
        { std::ofstream f(state); f << "x"; }
        core::pm1Prime95HandoffEnd(state);
        CHECK(!core::pm1Prime95HandoffPending(state, key));             // handoff finished or failed
        std::remove(state.c_str());
    }

    if (g_fail) return 1;
    std::cout << "pm1 external stage-2 interrupt test passed\n";
    return 0;
}
