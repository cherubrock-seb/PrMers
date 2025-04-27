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
    std::cout << "PrMers : GPU‐accelerated Mersenne primality test (OpenCL, NTT, Lucas Lehmer)\n"
              << "Testing exponent : " << o.exponent << "\n"
              << "Device OpenCL ID : " << o.device_id << "\n"
              << "Mode : " << (o.mode=="prp"?"PRP":"Lucas Lehmer") << "\n"
              << "Backup interval : " << o.backup_interval << " s\n"
              << "Save/Load path: " << o.save_path << "\n";
    if (o.profiling) {
        std::cout << "\n Kernel profiling is activated. Performance metrics will be displayed.\n";
    }
}

bool Printer::finalReport(const io::CliOptions& opts,
                          const std::vector<uint64_t>& resultVec,
                          std::string res64,
                          uint64_t n,
                          const std::string& timestampBuf,
                          double elapsed, std::string jsonResult) {
    const uint32_t p = opts.exponent;
    const std::string& mode = opts.mode;
    const bool proof = opts.proof;
    const std::string& proofFile = opts.proofFile;

    bool isPrime = false;
    std::string statusCode;

    if (mode == "ll") {
        isPrime = std::all_of(resultVec.begin(), resultVec.end(), [](uint64_t v) { return v == 0; });
        statusCode = isPrime ? "P" : "C";
        std::cout << "\nM" << p << " is " << (isPrime ? "prime" : "composite") << ".\n";
    } else {
        isPrime = (resultVec[0] == 9) && std::all_of(resultVec.begin() + 1, resultVec.end(), [](uint64_t v) { return v == 0; });
        statusCode = isPrime ? "P" : "C";
        std::cout << "\nM" << p << " PRP test: " << (isPrime ? "probably prime (residue is 9)" : "composite") << ".\n";
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


} // namespace core
