// src/core/Spinner.cpp
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
#include "core/Spinner.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

#ifdef _WIN32
  // Sur Windows (avant Win10 ou si ANSI non activé), on neutralise les couleurs
  #define COLOR_RED     ""
  #define COLOR_YELLOW  ""
  #define COLOR_GREEN   ""
  #define COLOR_MAGENTA ""
  #define COLOR_RESET   ""
#else
  // Sur les terminaux Unix/Linux et Windows 10+ en mode VTP
  #define COLOR_RED     "\033[31m"
  #define COLOR_YELLOW  "\033[33m"
  #define COLOR_GREEN   "\033[32m"
  #define COLOR_MAGENTA "\033[35m"
  #define COLOR_RESET   "\033[0m"
#endif

using namespace std::chrono;

namespace core {

void Spinner::displayProgress(uint32_t iter,
                              uint32_t totalIters,
                              double elapsedTime,
                              double elapsedTime2,
                              uint32_t expo,
                              uint32_t resumeIter,
                              std::string res64)
{
    double pct = totalIters
               ? (100.0 * iter) / totalIters
               : 100.0;

    uint32_t deltaIter = (iter > resumeIter) ? (iter - resumeIter) : iter;

    double currentIPS = (elapsedTime2 > 0)
                      ? deltaIter / elapsedTime2
                      : 0.0;

    static double smoothedIPS = 0.0;
    constexpr double alpha    = 0.1;
    if (smoothedIPS == 0.0) {
        smoothedIPS = currentIPS;
    } else {
        smoothedIPS = alpha * currentIPS
                    + (1.0 - alpha) * smoothedIPS;
    }

    double remaining = smoothedIPS > 0
                     ? (totalIters - iter) / smoothedIPS
                     : 0.0;

    const char* color = (pct < 50.0) ? COLOR_RED
                        : (pct < 90.0) ? COLOR_YELLOW
                                       : COLOR_GREEN;

    uint32_t sec  = static_cast<uint32_t>(remaining);
    uint32_t days = sec / 86400; sec %= 86400;
    uint32_t hrs  = sec / 3600;  sec %= 3600;
    uint32_t min  = sec / 60;    sec %= 60;

    std::cout
    << "\r" << color
    << "Progress: "  << std::fixed << std::setprecision(2) << pct << "% | "
    << "Exp: "       << expo                                << " | "
    << "Iter: "      << iter                                << " | "
    << "Elapsed: "   << std::fixed << elapsedTime           << "s | "
    << "IPS: "       << std::fixed << std::setprecision(2)
                    << smoothedIPS                          << " | "
    << "ETA: "       << days << "d " << hrs << "h "
                    << min  << "m " << sec << "s";
    
    if (!res64.empty()) {
    std::cout << " | RES64: " << res64;
    }

    std::cout << COLOR_RESET << std::endl;

}


void Spinner::displayBackupInfo(uint32_t iter,
                                uint32_t totalIters,
                                double elapsedTime,
                                std::string res64)
{
    double pct       = totalIters ? (100.0 * iter) / totalIters : 100.0;
    double ips       = elapsedTime > 0 ? iter / elapsedTime : 0.0;
    double remaining = ips > 0 ? (totalIters - iter) / ips : 0.0;

    std::cout
      << "\r" << COLOR_MAGENTA
      << "[Backup] " 
      << std::fixed << std::setprecision(2) << pct << "% | "
      << "Elapsed: " << elapsedTime << "s | "
      << "IPS: "     << std::fixed << ips << " | "
      << "ETA: "     << remaining << "s" << " | "
      << "RES64: "   << res64
      << COLOR_RESET
      << std::endl;
}

void Spinner::displaySpinner(std::atomic<bool>& waiting,
                             double estimatedSeconds,
                             std::atomic<bool>& isFirst)
{
    static const char symbols[] = {'|','/','-','\\'};
    size_t idx = 0;
    auto start = steady_clock::now();
    auto lastDraw = start;

    while (waiting.load()) {
        auto now = steady_clock::now();
        auto secs = duration_cast<seconds>(now - start).count();

        if (isFirst.load()) {
            std::cout << "\r🕒 First GPU flush "
                      << symbols[idx++ % 4]
                      << " (" << secs << "s";
            if (estimatedSeconds > 0)
                std::cout << " / ~" << (int)estimatedSeconds << "s";
            std::cout << ")..." << std::flush;
        } else if (duration_cast<seconds>(now - lastDraw).count() >= 1) {
            lastDraw = now;
        }
        std::this_thread::sleep_for(milliseconds(200));
    }

    std::cout << "\rGPU queue flushed.                   \n"
              << std::flush;
}

} // namespace core
