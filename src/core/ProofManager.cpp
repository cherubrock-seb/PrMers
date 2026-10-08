// core/ProofManager.cpp
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
#include "core/ProofManager.hpp"
#include "io/JsonBuilder.hpp"
#include <vector>
#include <iostream>
#include <filesystem>
#include <system_error>
#include <chrono>
#include <stdexcept>

namespace {

namespace fs = std::filesystem;

fs::path ensureDir(const fs::path& p) {
    std::error_code ec;
    fs::create_directories(p, ec);
    if (ec) {
        throw std::runtime_error("Cannot create directory '" + p.string() + "': " + ec.message());
    }
    return p;
}

void fancyRename(const fs::path& from, const fs::path& to) {
    std::error_code ec;

    fs::remove(to, ec);
    ec.clear();

    fs::rename(from, to, ec);
    if (!ec) return;

    ec.clear();
    fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        throw std::runtime_error("Cannot move '" + from.string() + "' -> '" + to.string() + "': " + ec.message());
    }
    ec.clear();
    fs::remove(from, ec);
}

fs::path uniqueTmpPath(const fs::path& dir, const std::string& baseName) {
    uint64_t ts = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return dir / (baseName + ".tmp-" + std::to_string(ts));
}

} // namespace

namespace core {

ProofManager::ProofManager(uint32_t exponent, int proofLevel,
                           cl_command_queue queue, uint32_t n,
                           const std::vector<int>& digitWidth,
                           const std::vector<std::string>& knownFactors,
                           const std::string& savePath)
  : proofSet_(exponent, static_cast<uint32_t>(proofLevel), knownFactors, ProofLocation(savePath))
  , queue_(queue)
  , n_(n)
  , exponent_(exponent)
  , digitWidth_(digitWidth)
{}

void ProofManager::checkpoint(cl_mem buf, uint32_t iter) {
    if (! proofSet_.shouldCheckpoint(iter)) return;

    std::vector<uint64_t> host(n_);
    clEnqueueReadBuffer(queue_, buf, CL_TRUE, 0,
                        n_ * sizeof(uint64_t),
                        host.data(), 0, nullptr, nullptr);

    auto words = io::JsonBuilder::compactBits(host, digitWidth_, exponent_);

    saveProofCheckpointVerified(proofSet_, iter, words);
}

void ProofManager::checkpointMarin(std::vector<uint64_t> host, uint32_t iter) {
    if (! proofSet_.shouldCheckpoint(iter)) return;

    auto words = io::JsonBuilder::compactBits(host, digitWidth_, exponent_);

    saveProofCheckpointVerified(proofSet_, iter, words);
}

std::filesystem::path ProofManager::proof(const prmers::ocl::Context& ctx, opencl::NttEngine& ntt, math::Carry& carry, uint32_t proofPower, bool verify) const {
    try {
        size_t limbBytes = n_ * sizeof(uint64_t);

        core::GpuContext gpu(exponent_, ctx, ntt, carry, digitWidth_, limbBytes);

        Proof proof = proofSet_.computeProof(gpu, proofPower);

        std::string filename = std::to_string(exponent_) + "-" +
                               std::to_string(proof.middles.size()) + ".proof";

        // Under the save path (-f) with the checkpoints and results.txt.
        const ProofLocation& where = proofSet_.location();
        fs::path proofDir = ensureDir(where.proofDir());
        fs::path tmpDir = ensureDir(where.proofTmpDir());

        fs::path finalPath = proofDir / filename;
        fs::path tmpPath = uniqueTmpPath(tmpDir, filename);

        proof.save(tmpPath);

        auto loadedProof = Proof::load(tmpPath);
        if (verify && !loadedProof.verify(gpu, proofPower)) {
            std::error_code ec;
            fs::remove(tmpPath, ec);
            throw ProofVerificationError(
                "proof for M" + std::to_string(exponent_) + " (power " +
                std::to_string(proofPower) + ") failed verification; the proof file was discarded");
        }

        fancyRename(tmpPath, finalPath);

        return finalPath;
    } catch (const std::exception& e) {
        std::cerr << "Error generating proof file: " << e.what() << std::endl;
        throw;
    }
}

} // namespace core
