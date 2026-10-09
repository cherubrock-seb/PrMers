// Windows counterpart of the POSIX Prime95 handoff interrupt test: core::pm1WaitProcessInterruptible must
// report an interrupt that arrives while the child runs (and one that is already pending when it exits),
// and must terminate a child that ignores it. The child is this same executable run with --sleep.
// Built with MinGW and run under Wine by tests/pm1_p95_windows_interrupt_test.sh.
#include "core/Pm1Stage2External.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

static HANDLE spawnSelf(const char* self, int sleepMs, int exitCode) {
    std::string cmd = std::string("\"") + self + "\" --sleep " + std::to_string(sleepMs) + " " + std::to_string(exitCode);
    STARTUPINFOA si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return nullptr;
    CloseHandle(pi.hThread);
    return pi.hProcess;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--sleep") == 0) {
        Sleep(static_cast<DWORD>(std::atoi(argv[2])));
        return std::atoi(argv[3]);
    }
    using clock = std::chrono::steady_clock;

    {   // no interrupt: exit code passes through, not flagged
        std::atomic<bool> stop{false};
        int ticks = 0; bool intr = true;
        HANDLE h = spawnSelf(argv[0], 400, 7);
        CHECK(h != nullptr);
        const int rc = core::pm1WaitProcessInterruptible(h, stop, [&] { ++ticks; }, intr);
        CloseHandle(h);
        CHECK(rc == 7);
        CHECK(!intr);
        CHECK(ticks >= 1);
    }
    {   // interrupt while the child runs, child ends by itself within the grace period
        std::atomic<bool> stop{false};
        bool intr = false;
        std::thread t([&] { Sleep(300); stop = true; });
        HANDLE h = spawnSelf(argv[0], 1500, 0);
        CHECK(h != nullptr);
        const int rc = core::pm1WaitProcessInterruptible(h, stop, nullptr, intr, 30);
        t.join();
        CloseHandle(h);
        CHECK(rc == 0);          // like Prime95 stopping gracefully: the exit code cannot tell
        CHECK(intr);             // ... so the flag must
    }
    {   // interrupt arrives after the last poll but before the exit is seen
        std::atomic<bool> stop{true};
        bool intr = false;
        HANDLE h = spawnSelf(argv[0], 0, 0);
        CHECK(h != nullptr);
        const int rc = core::pm1WaitProcessInterruptible(h, stop, nullptr, intr, 30);
        CloseHandle(h);
        CHECK(rc == 0);
        CHECK(intr);
    }
    {   // a child that ignores the interrupt is terminated after the grace period
        std::atomic<bool> stop{false};
        bool intr = false;
        std::thread t([&] { Sleep(200); stop = true; });
        HANDLE h = spawnSelf(argv[0], 60000, 0);
        CHECK(h != nullptr);
        const auto t0 = clock::now();
        const int rc = core::pm1WaitProcessInterruptible(h, stop, nullptr, intr, 1);
        const double dt = std::chrono::duration<double>(clock::now() - t0).count();
        t.join();
        CloseHandle(h);
        CHECK(intr);
        CHECK(rc == 1);          // TerminateProcess(…, 1)
        CHECK(dt < 20.0);
    }
    {   // an invalid handle fails the wait instead of spinning
        std::atomic<bool> stop{false};
        bool intr = true;
        const int rc = core::pm1WaitProcessInterruptible(nullptr, stop, nullptr, intr);
        CHECK(rc == -1);
        CHECK(!intr);
    }
    std::printf("Windows Prime95 interrupt test: %s\n", g_fail ? "FAIL" : "OK");
    return g_fail ? 1 : 0;
}
