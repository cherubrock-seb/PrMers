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
#include "core/Printer.hpp"
#include "io/JsonBuilder.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace core {

void Printer::banner(const io::CliOptions& o) {
    std::cout << "PrMers : GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas Lehmer)\n"
              << "Testing : " << Printer::formatNumber(o) << "\n"
              << "Device OpenCL ID : " << o.device_id << "\n"
              << "Mode : " << (o.mode=="prp"?"PRP":"Lucas Lehmer") << "\n"
              << "Backup interval : " << o.backup_interval << " s\n"
              << "Save/Load path: " << o.save_path << "\n";
    if (o.profiling) {
        std::cout << "\n Kernel profiling is activated. Performance metrics will be displayed.\n";
    }
}

bool Printer::finalReport(const io::CliOptions& opts,
                          double elapsed,
                          const std::string& jsonResult,
                          bool isPrime) {
    const uint32_t p = opts.exponent;
    const std::string& mode = opts.mode;

    if (mode == "ll") {
        std::cout << "\nM" << p << " is " << (isPrime ? "prime" : "composite") << ".\n";
    } else {
        if (!opts.knownFactors.empty()) {
            std::cout << "\nM" << p;
            for (const auto& factor : opts.knownFactors) {
                std::cout << "/" << factor;
            }
            std::cout << " PRP test: " << (isPrime ? "probably prime" : "composite") << ".\n";
        } else {
            std::cout << "\nM" << p << " PRP test: " << (isPrime ? "probably prime (residue is 9)" : "composite") << ".\n";
        }
    }

    std::cout << "\nManual submission JSON:\n" << jsonResult << std::endl;
    std::cout << "\nTotal elapsed time: " << elapsed << " seconds.\n";

    return isPrime;
}
void Printer::displayVector(const std::vector<uint64_t>& vec, const std::string& label) {
    std::cout << "\n" << label << " [size=" << vec.size() << "]:\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::hex << std::setw(4) << std::setfill('0') << i << ": "
                  << std::hex << std::setw(16) << std::setfill('0') << vec[i]
                  << " (" << std::dec << vec[i] << ")" << "\n";
    }
    std::cout << std::dec; // Reset to decimal output
}
std::string Printer::formatNumber(const io::CliOptions& opts) {
    std::ostringstream oss;
    if (opts.wagstaff) {
        oss << "(2^" << opts.exponent/2 << "+1)/3";
    } else {
        oss << "(2^" << opts.exponent << "-1)";
    }
    if (!opts.knownFactors.empty()) {
        oss << "/";
        for (size_t i = 0; i < opts.knownFactors.size(); ++i) {
            if (i > 0) oss << "/";
            oss << opts.knownFactors[i];
        }
    }
    return oss.str();
} // namespace core
