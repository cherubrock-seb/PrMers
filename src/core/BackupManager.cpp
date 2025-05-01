// src/core/BackupManager.cpp
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
#include "core/BackupManager.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace core {

BackupManager::BackupManager(cl_command_queue queue,
                             unsigned interval,
                             size_t vectorSize,
                             const std::string& savePath,
                             unsigned exponent,
                             const std::string& mode)
  : queue_(queue)
  , backupInterval_(interval)
  , vectorSize_(vectorSize)
  , savePath_(savePath.empty() ? "." : savePath)
  , exponent_(exponent)
  , mode_(mode)
{
    std::filesystem::create_directories(savePath_);
    auto base = std::to_string(exponent_) + mode_;
    mersFilename_ = savePath_ + "/" + base + ".mers";
    loopFilename_ = savePath_ + "/" + base + ".loop";
}

uint32_t BackupManager::loadState(std::vector<uint64_t>& x) {
    uint32_t resume = 0;

    // 1) Debug : afficher le chemin absolu du .loop
    auto absLoop = std::filesystem::absolute(loopFilename_);
    std::cout << "Looking for loop file at " << absLoop << std::endl;

    // 2) Essayer de lire resume, puis valider resume > 0
    std::ifstream loopIn(loopFilename_);
    if (loopIn >> resume && resume > 0) {
        std::cout << "Resuming from iteration " << resume
                  << " based on " << absLoop << std::endl;

        // 3) Charger le vecteur binaire
        std::ifstream mersIn(mersFilename_, std::ios::binary);
        if (mersIn) {
            mersIn.read(reinterpret_cast<char*>(x.data()),
                        x.size() * sizeof(uint64_t));
            std::cout << "Loaded state from "
                      << std::filesystem::absolute(mersFilename_)
                      << std::endl;
        } else {
            std::cerr << "Warning: could not open mers file at "
                      << std::filesystem::absolute(mersFilename_)
                      << " — starting with uninitialized data\n";
        }
    }
    else {
        // 4) Pas de fichier valide → nouvelle session
        std::cout << "No valid loop file, initializing fresh state\n";
        resume = 0;
        x.assign(x.size(), 0ULL);
        x[0] = (mode_ == "prp") ? 3ULL : 4ULL;
    }

    return resume;
}


void BackupManager::saveState(cl_mem buffer, uint32_t iter) {
    std::vector<uint64_t> x(vectorSize_);
    clEnqueueReadBuffer(queue_, buffer, CL_TRUE,
                        0, vectorSize_ * sizeof(uint64_t),
                        x.data(), 0, nullptr, nullptr);

    // write binary state
    std::ofstream mersOut(mersFilename_, std::ios::binary);
    if (mersOut) {
        mersOut.write(reinterpret_cast<const char*>(x.data()),
                      vectorSize_ * sizeof(uint64_t));
        std::cout << "\nState saved to " << mersFilename_ << std::endl;
    } else {
        std::cerr << "Error saving state to " << mersFilename_ << std::endl;
    }

    // write next-iteration
    std::ofstream loopOut(loopFilename_);
    if (loopOut) {
        loopOut << (iter + 1);
        std::cout << "Loop iteration saved to " << loopFilename_ << std::endl;
    } else {
        std::cerr << "Error saving loop state to " << loopFilename_ << std::endl;
    }
}

void BackupManager::clearState() const {
    std::error_code ec;
    if (std::filesystem::exists(mersFilename_, ec)) {
        std::filesystem::remove(mersFilename_, ec);
        std::cout << "Removed backup file: " << mersFilename_ << std::endl;
    }
    if (std::filesystem::exists(loopFilename_, ec)) {
        std::filesystem::remove(loopFilename_, ec);
        std::cout << "Removed loop file: " << loopFilename_ << std::endl;
    }
}

} // namespace core
