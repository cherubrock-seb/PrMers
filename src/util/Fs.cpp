#include "util/Fs.hpp"
#include <filesystem>
#include <stdexcept>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

std::string getExecutableDir() {
    char buffer[1024];

#ifdef __APPLE__
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) != 0)
        throw std::runtime_error("Cannot get executable path (macOS).");

#elif defined(_WIN32)
    if (!GetModuleFileNameA(NULL, buffer, sizeof(buffer)))
        throw std::runtime_error("Cannot get executable path (Windows).");

#else
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer)-1);
    if (len == -1)
        throw std::runtime_error("Cannot get executable path (Linux).");
    buffer[len] = '\0';
#endif

    std::string fullPath(buffer);
    return fullPath.substr(0, fullPath.find_last_of("/\\"));
}

void markJsonAsSent(const std::string& path) {
    std::filesystem::rename(path, path + ".sent");
}
