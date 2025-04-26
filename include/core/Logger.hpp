// include/core/Logger.hpp
#pragma once

#include "io/CliParser.hpp"
#include <string>


namespace core {

class Logger {
public:
    explicit Logger(const std::string& logFile);
    void logStart(const io::CliOptions& options);
    void logEnd(double elapsed);
    void logmsg(const char* fmt, ...);
    void flush_log();

private:
    std::string _logFile;
};

} // namespace core
