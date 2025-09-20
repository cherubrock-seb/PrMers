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
#include <sstream>
#include "ui/WebGuiServer.hpp"

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

void Spinner::displayProgress(uint64_t iter,
                              uint64_t totalIters,
                              double elapsedTime,
                              double elapsedTime2,
                              uint64_t expo,
                              uint64_t resumeIter,
                              uint64_t startIter,
                              std::string res64,
                              ui::WebGuiServer* gui)
{
    double pct = totalIters
               ? (100.0 * iter) / totalIters
               : 100.0;

    uint64_t deltaIter = (iter > resumeIter) ? (iter - resumeIter) : iter;

    double currentIPS = (elapsedTime2 > 0)
                      ? deltaIter / elapsedTime2
                      : 0.0;

    static double smoothedIPS = 0.0;
    constexpr double alpha    = 0.05;
//    constexpr double alpha = 1;
    if (smoothedIPS == 0.0) {
        smoothedIPS = currentIPS;
    } else {
        smoothedIPS = alpha * currentIPS
                    + (1.0 - alpha) * smoothedIPS;
    }

    double remaining = currentIPS > 0
                     ? (totalIters - iter) / currentIPS
                     : 0.0;

    const char* color = (pct < 50.0) ? COLOR_RED
                        : (pct < 90.0) ? COLOR_YELLOW
                                       : COLOR_GREEN;

    uint64_t sec  = static_cast<uint64_t>(remaining);
    uint64_t days = sec / 86400; sec %= 86400;
    uint64_t hrs  = sec / 3600;  sec %= 3600;
    uint64_t min  = sec / 60;    sec %= 60;

    std::cout
    << "\r" << color
    << "Progress: "  << std::fixed << std::setprecision(2) << pct << "% | "
    << "Exp: "       << expo                                << " | "
    << "Iter: "      << iter                                << " | "
    << "Elapsed: "   << std::fixed << elapsedTime           << "s | "
    << "IPS: "       << std::fixed << std::setprecision(2)
                    << currentIPS                          << " | "
    << "ETA: "       << days << "d " << hrs << "h "
                    << min  << "m " << sec << "s";
    
    if (!res64.empty()) {
    std::cout << " | RES64: " << res64;
    }
    if (gui) {
        gui->setProgress(iter, totalIters, res64);
        std::ostringstream ss;
        ss  << "Progress: "  << std::fixed << std::setprecision(2) << pct << "% | "
            << "Exp: "       << expo                                << " | "
            << "Iter: "      << iter                                << " | "
            << "Elapsed: "   << std::fixed << elapsedTime           << "s | "
            << "IPS: "       << std::fixed << std::setprecision(2)
                            << currentIPS                          << " | "
            << "ETA: "       << days << "d " << hrs << "h "
                            << min  << "m " << sec << "s";
        gui->appendLog(ss.str());
        if (!res64.empty()) {
            gui->appendLog(ss.str()); 
        }
    }
    std::cout << COLOR_RESET << std::endl;

}

void Spinner::displayProgress2(uint64_t iter,
                               uint64_t totalIters,
                               double elapsedTime,
                               double elapsedTime2,
                               uint64_t expo,
                               uint64_t resumeIter,
                               uint64_t startIter,
                               std::string res64,
                               ui::WebGuiServer* gui,
                               uint64_t chunkIndex,
                               uint64_t chunkCount,
                               uint64_t chunkIter,
                               uint64_t chunkTotal,
                               bool reset)
{
    double pctTotal = totalIters ? (100.0 * (double)iter) / (double)totalIters : 100.0;
    double pctChunk = chunkTotal ? (100.0 * (double)chunkIter) / (double)chunkTotal : 100.0;

    static uint64_t prevIter = 0;
    static uint64_t prevChunkIndex = 0;
    static bool first = true;
    uint64_t deltaIter = 0;
    if (reset || first || iter <= prevIter || chunkIndex != prevChunkIndex) {
        deltaIter = 0;
        first = false;
        prevIter = iter;
        prevChunkIndex = chunkIndex;
        static double sm = 0.0; sm = 0.0;
    } else {
        deltaIter = iter - prevIter;
        prevIter = iter;
    }

    double intervalT = (elapsedTime2 > 0.0) ? elapsedTime2 : ((elapsedTime > 0.0) ? elapsedTime : 1.0);
    double instIPS   = (intervalT > 0.0) ? ((double)deltaIter / intervalT) : 0.0;

    static double smoothedIPS = 0.0;
    const double alpha = 0.20;
    if (reset || smoothedIPS == 0.0) smoothedIPS = (elapsedTime > 0.0) ? ((double)iter / elapsedTime) : instIPS;
    else smoothedIPS = alpha * instIPS + (1.0 - alpha) * smoothedIPS;

    double remaining = (smoothedIPS > 0.0) ? ((double)(totalIters - iter) / smoothedIPS) : 0.0;
    uint64_t sec  = (uint64_t)(remaining + 0.5);
    uint64_t days = sec / 86400; sec %= 86400;
    uint64_t hrs  = sec / 3600;  sec %= 3600;
    uint64_t min  = sec / 60;    sec %= 60;

    const char* color = (pctTotal < 50.0) ? COLOR_RED : ((pctTotal < 90.0) ? COLOR_YELLOW : COLOR_GREEN);

    std::ostringstream line;
    line << "\r" << color
         << "Progress: "  << std::fixed << std::setprecision(2) << pctTotal << "% | "
         << "Chunk "      << chunkIndex << "/" << chunkCount << " " << std::fixed << std::setprecision(2) << pctChunk << "% | "
         << "Exp: "       << expo << " | "
         << "Iter: "      << iter << " | "
         << "ChIter: "    << chunkIter << " | "
         << "Elapsed: "   << std::fixed << elapsedTime << "s | "
         << "IPS: "       << std::fixed << std::setprecision(2)  << smoothedIPS   << " | "
         << "ETA: "       << days << "d " << hrs << "h " << min << "m " << sec << "s";
    if (!res64.empty()) line << " | RES64: " << res64;

    std::cout << line.str() << COLOR_RESET << std::endl;

    if (gui) {
        gui->setProgress(iter, totalIters, res64);
        gui->appendLog(line.str());
    }
}


void Spinner::displayBackupInfo(uint64_t iter,
                                uint64_t totalIters,
                                double elapsedTime,
                                std::string res64,
                                ui::WebGuiServer* gui)
{
    double pct       = totalIters ? (100.0 * iter) / totalIters : 100.0;
    double ips       = elapsedTime > 0 ? iter / elapsedTime : 0.0;
    double remaining = ips > 0 ? (totalIters - iter) / ips : 0.0;
    uint64_t sec  = static_cast<uint64_t>(remaining);
    uint64_t days = sec / 86400; sec %= 86400;
    uint64_t hrs  = sec / 3600;  sec %= 3600;
    uint64_t min  = sec / 60;    sec %= 60;

    std::cout
      << "\r" << COLOR_MAGENTA
      << "[Backup] " 
      << std::fixed << std::setprecision(2) << pct << "% | "
      << "Elapsed: " << elapsedTime << "s | "
      << "IPS: "     << std::fixed << ips << " | "
      << "ETA: "       << days << "d " << hrs << "h "
                       << min  << "m " << sec << "s";
    if (!res64.empty()) {
        std::cout << " | RES64: " << res64;
    }
    if (gui) {
        gui->setProgress(iter, totalIters, res64);
        std::ostringstream ss;
        ss  << "[Backup] " 
            << std::fixed << std::setprecision(2) << pct << "% | "
            << "Elapsed: " << elapsedTime << "s | "
            << "IPS: "     << std::fixed << ips << " | "
            << "ETA: "       << days << "d " << hrs << "h "
                            << min  << "m " << sec << "s";
        gui->appendLog(ss.str());
        if (!res64.empty()) {
            gui->appendLog(ss.str()); 
        }
    }
    std::cout << COLOR_RESET << std::endl;
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
