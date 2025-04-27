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
// src/io/WorktodoManager.cpp
#include "io/WorktodoManager.hpp"
#include "io/CliParser.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

namespace io {

WorktodoManager::WorktodoManager(const io::CliOptions& opts)
  : options_(opts)
{}

void WorktodoManager::saveIndividualJson(uint32_t p,
                                         const std::string& mode,
                                         const std::string& jsonResult) const
{
    // ex: ./save/100003_prp_result.json
    std::string file = options_.save_path + "/"
                     + std::to_string(p) + "_" + mode + "_result.json";
    std::ofstream out(file);
    if (!out) {
        std::cerr << "Cannot open " << file << " for writing JSON\n";
        return;
    }
    out << jsonResult;
    std::cout << "JSON result written to: " << file << "\n";
}

void WorktodoManager::appendToResultsTxt(const std::string& jsonResult) const
{
    // ex: ./save/results.txt
    std::string resultPath = options_.save_path + "/results.txt";
    std::ofstream resOut(resultPath, std::ios::app);
    if (resOut) {
        resOut << jsonResult << "\n";
        std::cout << "Result appended to: " << resultPath << "\n";
    } else {
        std::cerr << "Cannot open " << resultPath << " for appending\n";
    }
}

} // namespace core
