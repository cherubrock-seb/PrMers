// Restarting through util::execSelf must work when the program was started via
// PATH, i.e. when argv[0] is a bare name that execv() cannot resolve.
// First run: re-executes itself with "child".  Child run: prints a marker.
// "fail" mode: the exec cannot succeed (argument list too large); the process must
// then exit with util::kRestartFailedExitCode instead of returning to its caller.
#include "util/SelfExe.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "fail") == 0) {
        // A single argument far beyond ARG_MAX / MAX_ARG_STRLEN: every exec attempt fails with E2BIG.
        std::vector<std::string> args = {argv[0], std::string(4u << 20, 'x')};
        util::execSelf(args);
        std::perror("restart failed as expected");
        util::exitRestartFailed();
    }
    if (argc > 1 && std::strcmp(argv[1], "child") == 0) {
        std::puts("RESTART_OK");
        return 0;
    }

    const std::string exe = util::selfExecutablePath();
    if (exe.empty() || exe[0] != '/') {
        std::fprintf(stderr, "FAIL: selfExecutablePath() = '%s'\n", exe.c_str());
        return 1;
    }

    std::vector<std::string> args = {argv[0], "child"};
    util::execSelf(args);
    std::perror("FAIL: execSelf returned");
    return 1;
}
