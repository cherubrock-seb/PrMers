#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef __APPLE__
# include <mach-o/dyld.h>
# include <climits>
# include <cstdlib>
#endif
#if !defined(_WIN32)
# include <unistd.h>
#endif

namespace util {

// Absolute path of the running executable, or "" when it cannot be determined.
// argv[0] is not enough: when prmers is started through PATH it is a bare name
// that execv() does not search for.
inline std::string selfExecutablePath() {
#if defined(_WIN32)
    return "";
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(&buf[0], &size) != 0) return "";
    buf.resize(buf.find('\0'));
    char real[PATH_MAX];
    if (realpath(buf.c_str(), real)) return real;
    return buf;
#else
    std::string buf(4096, '\0');
    const ssize_t len = readlink("/proc/self/exe", &buf[0], buf.size() - 1);
    if (len <= 0) return "";
    buf.resize(static_cast<size_t>(len));
    return buf;
#endif
}

#if !defined(_WIN32)
// Replace the current process with a fresh copy of itself running args
// (args[0] is the program name, as in argv).  Returns only if every exec
// attempt failed, with errno set by the last one.
inline void execSelf(const std::vector<std::string>& args) {
    if (args.empty()) { errno = EINVAL; return; }
    std::vector<char*> argv;
    for (const auto& s : args) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);

    const std::string exe = selfExecutablePath();
    if (!exe.empty()) execv(exe.c_str(), argv.data());
    // Fall back to argv[0]; execvp also searches PATH when it has no slash.
    execvp(argv[0], argv.data());
}
#endif

// Exit status of a process that had work left (more worktodo entries, an appended GUI entry) but could
// not restart itself. Distinct from the 0/1 a finished test returns, so a wrapper script or service
// manager can tell "queue finished" from "queue stalled".
constexpr int kRestartFailedExitCode = 3;

// Terminate after a restart attempt failed and was reported. Like a successful exec, this skips static
// destructors (the caller may be the GUI's HTTP thread); buffered output is flushed first.
[[noreturn]] inline void exitRestartFailed() {
    std::fflush(nullptr);
    std::_Exit(kRestartFailedExitCode);
}

} // namespace util
