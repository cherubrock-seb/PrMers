// Host test for the Prime95 stage-2 handoff decisions, using a stub "mprime"
// shell script that is started the same way the P-1 driver starts Prime95.
#include "core/Pm1Stage2External.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
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

    if (g_fail) return 1;
    std::cout << "pm1 external stage-2 interrupt test passed\n";
    return 0;
}
