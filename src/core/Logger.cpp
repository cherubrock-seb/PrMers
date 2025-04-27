// src/core/Logger.cpp
/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
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
