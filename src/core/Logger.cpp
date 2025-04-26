// src/core/Logger.cpp
#include "core/Logger.hpp"
#include "io/CliParser.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cstdarg>

namespace core {

static std::vector<std::string> s_messages;

Logger::Logger(const std::string& logFile)
  : _logFile(logFile)
{}

void Logger::logStart(const io::CliOptions& options) {
    logmsg("=== Début : exponent=%u, mode=%s\n",
           options.exponent,
           options.mode.c_str());
}

void Logger::logEnd(double elapsed) {
    logmsg("=== Fin : elapsed=%.3f s\n", elapsed);
    flush_log();
}

void Logger::logmsg(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    s_messages.emplace_back(buf);
}

void Logger::flush_log() {
    std::ofstream out(_logFile, std::ios::app);
    for (auto& m : s_messages) {
        out << m;
    }
    s_messages.clear();
}

} // namespace core
