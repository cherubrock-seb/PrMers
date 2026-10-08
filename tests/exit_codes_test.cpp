// Host test: core::exitCodeForRun maps a stop signal to the interrupted code in CLI mode only.
#include "core/ExitCodes.hpp"

#include <cstdio>

using core::exitCodeForRun;
using core::kExitInterrupted;

static_assert(kExitInterrupted == 1, "same as memtest and P-1 stage 2");
static_assert(exitCodeForRun(0, false, false) == 0, "finished");
static_assert(exitCodeForRun(1, false, false) == 1, "GM result code 1 is kept");
static_assert(exitCodeForRun(2, false, false) == 2, "error code is kept");
static_assert(exitCodeForRun(0, true, false) == 1, "stopped, mode returned 0 (PRP, LL, P-1, ECM, GM ...)");
static_assert(exitCodeForRun(1, true, false) == 1, "stopped, mode returned 1 (memtest, GM no-factor code: indistinguishable by design)");
static_assert(exitCodeForRun(2, true, false) == 1, "stopped while failing is still a stop");
static_assert(exitCodeForRun(0, true, true) == 0, "GUI mode: Stop is the normal way to quit");
static_assert(exitCodeForRun(2, true, true) == 2, "GUI mode keeps the mode result");

int main() { std::puts("exit codes test passed"); return 0; }
