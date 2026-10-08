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
#include "core/ProofSet.hpp"
#include "core/ProofCheckpoint.hpp"
#include "io/Sha3Hash.h"
#include "util/Crc32.hpp"
#include "util/Timer.hpp"
#include "util/GmpUtils.hpp"
#include "opencl/NttEngine.hpp"
#include "math/Carry.hpp"
#include "io/JsonBuilder.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace core {

// Words
Words::Words() = default;

Words::Words(const std::vector<uint64_t>& v)
  : data_{v} {}

const std::vector<uint64_t>& Words::data() const noexcept {
    return data_;
}

std::vector<uint64_t>& Words::data() noexcept {
    return data_;
}

Words Words::fromUint64(const std::vector<uint64_t>& host, uint32_t exponent) {
    (void)exponent;
    return Words(host);
}

// ProofSet
ProofSet::ProofSet(uint32_t exponent, uint32_t proofLevel, std::vector<std::string> factors,
                   ProofLocation location)
  : E{exponent}, power{proofLevel}, knownFactors{std::move(factors)}, location_{std::move(location)} {
  // Calculate checkpoint points using binary tree structure
  std::vector<uint32_t> spans;
  for (uint32_t span = (E + 1) / 2; spans.size() < power; span = (span + 1) / 2) { 
    spans.push_back(span); 
  }

  points.push_back(0);
  for (uint32_t p = 0, span = (E + 1) / 2; p < power; ++p, span = (span + 1) / 2) {
    for (uint32_t i = 0, end = static_cast<uint32_t>(points.size()); i < end; ++i) {
      points.push_back(points[i] + span);
    }
  }

  assert(points.size() == (1u << power));
  assert(points.front() == 0);

  points.front() = E;
  std::sort(points.begin(), points.end());

  assert(points.size() == (1u << power));
  assert(points.back() == E);

  points.push_back(uint32_t(-1)); // guard element

  // Verify all points are valid
  for (uint32_t p : points) {
    assert(p > E || isInPoints(E, power, p));
  }
}

bool ProofSet::shouldCheckpoint(uint32_t iter) const {
  return isInPoints(E, power, iter);
}

bool ProofSet::shouldCheckpoint2(uint32_t iter, uint32_t npower) const {
  return isInPoints(E, npower, iter);
}

void ProofSet::save(uint32_t iter, const std::vector<uint32_t>& words) {
  if (!shouldCheckpoint(iter)) {
    return;
  }

  // The directory is created with the first residue, so that tests that make
  // no proof (LL, P-1, ECM, -proof 0) leave nothing behind.
  std::error_code dirError;
  std::filesystem::create_directories(location_.residueDir(E), dirError);

  // Create the file path for this iteration
  auto filePath = location_.residueWriteFile(E, iter);
  
  // Write the words data to file
  std::ofstream file(filePath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot create proof checkpoint file: " + filePath.string());
  }
  
  // Write CRC32 first, then the data
  uint32_t crc = computeCRC32(words.data(), words.size() * sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
  file.write(reinterpret_cast<const char*>(words.data()),
          static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
  
  file.close();
  if (file.fail() || !syncFileToDisk(filePath)) {
    throw std::runtime_error("Error writing proof checkpoint file: " + filePath.string());
  }
}

Words ProofSet::fromUint64(const std::vector<uint64_t>& host, uint32_t exponent) {
    return Words::fromUint64(host, exponent);
}

uint32_t ProofSet::bestPower(uint32_t E) {
  // Best proof powers assuming no disk space concern.
  // We increment power by 1 for each fourfold increase of the exponent.
  // The values below produce power=10 at wavefront, and power=11 at 100Mdigits:
  // power=10 from 60M to 240M, power=11 from 240M up.

  //assert(E > 0);
  // log2(x)/2 is log4(x)
  int32_t power = 10 + static_cast<int32_t>(std::floor(std::log2(E / 60e6) / 2));
  power = std::max(power, 2);
  power = std::min(power, 12);
  return static_cast<uint32_t>(power);
}

bool ProofSet::isInPoints(uint32_t E, uint32_t npower, uint32_t k) {
  if (k == E) { return true; } // special-case E
  uint32_t start = 0;
  for (uint32_t p = 0, span = (E + 1) / 2; p < npower; ++p, span = (span + 1) / 2) {
    assert(k >= start);
    if (k > start + span) {
      start += span;
    } else if (k == start + span) {
      return true;
    }
  }
  return false;
}

std::filesystem::path ProofSet::proofPath(const ProofLocation& location, uint32_t E) {
  return location.residueDir(E);
}

bool ProofSet::adoptLegacyResidues(uint32_t resumeIter, std::string& note) {
  note.clear();
  if (resumeIter == 0) return false;
  // Every proof point the interrupted run had passed: the ones the resumed
  // run will not write again. The points are the ones of the power asked for.
  std::vector<uint32_t> needed;
  for (uint32_t point : points) {
    if (point < E && point <= resumeIter) needed.push_back(point);
  }
  return location_.adoptLegacy(E, needed, note);
}

bool ProofSet::isValidTo(uint32_t limitK) const {
  // Check if we have all required checkpoint files up to limitK
  for (uint32_t point : points) {
    if (point > limitK) break;
    if (point < E && !fileExists(point)) {
      return false;
    }
  }
  return true;
}

bool ProofSet::fileExists(uint32_t k) const {
  return std::filesystem::exists(location_.residueFile(E, k));
}

std::vector<uint32_t> ProofSet::load(uint32_t iter) const {
  if (!shouldCheckpoint(iter)) {
    throw std::runtime_error("Attempt to load non-checkpoint iteration: " + std::to_string(iter));
  }

  auto filePath = location_.residueFile(E, iter);
  std::ifstream file(filePath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open proof checkpoint file: " + filePath.string());
  }

  // Read CRC32 first
  uint32_t crc;
  file.read(reinterpret_cast<char*>(&crc), sizeof(crc));
  if (!file.good()) {
    throw std::runtime_error("Error reading CRC32 from proof checkpoint file: " + filePath.string());
  }

  // Calculate expected file size in 32-bit words: (E + 31) / 32
  uint32_t expectedWords = (E + 31) / 32;
  
  // Read the 32-bit words data
  std::vector<uint32_t> words(expectedWords);
  file.read(reinterpret_cast<char*>(words.data()), expectedWords * sizeof(uint32_t));
  if (!file.good()) {
    throw std::runtime_error("Error reading data from proof checkpoint file: " + filePath.string());
  }

  // Verify CRC32
  uint32_t computedCrc = computeCRC32(words.data(), words.size() * sizeof(uint32_t));
  if (crc != computedCrc) {
    throw std::runtime_error("CRC32 mismatch in proof checkpoint file: " + filePath.string());
  }

  return words;
}

std::vector<uint32_t> ProofSet::load2(uint32_t iter, uint32_t npower) const {
  if (!shouldCheckpoint2(iter,npower)) {
    throw std::runtime_error("Attempt to load non-checkpoint iteration: " + std::to_string(iter));
  }

  auto filePath = location_.residueFile(E, iter);
  std::ifstream file(filePath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open proof checkpoint file: " + filePath.string());
  }

  // Read CRC32 first
  uint32_t crc;
  file.read(reinterpret_cast<char*>(&crc), sizeof(crc));
  if (!file.good()) {
    throw std::runtime_error("Error reading CRC32 from proof checkpoint file: " + filePath.string());
  }

  // Calculate expected file size in 32-bit words: (E + 31) / 32
  uint32_t expectedWords = (E + 31) / 32;
  
  // Read the 32-bit words data
  std::vector<uint32_t> words(expectedWords);
  file.read(reinterpret_cast<char*>(words.data()), expectedWords * sizeof(uint32_t));
  if (!file.good()) {
    throw std::runtime_error("Error reading data from proof checkpoint file: " + filePath.string());
  }

  // Verify CRC32
  uint32_t computedCrc = computeCRC32(words.data(), words.size() * sizeof(uint32_t));
  if (crc != computedCrc) {
    throw std::runtime_error("CRC32 mismatch in proof checkpoint file: " + filePath.string());
  }

  return words;
}

void ProofSet::rebuildPointsForPower(uint32_t p) const {
    std::vector<uint32_t> newPoints;
    newPoints.push_back(0);

    for (uint32_t level = 0, span = (E + 1) / 2; level < p; ++level, span = (span + 1) / 2) {
        const uint32_t end = static_cast<uint32_t>(newPoints.size());
        for (uint32_t i = 0; i < end; ++i) {
            newPoints.push_back(newPoints[i] + span);
        }
    }

    if (!newPoints.empty()) newPoints.front() = E;
    std::sort(newPoints.begin(), newPoints.end());
    newPoints.push_back(uint32_t(-1));

    points.swap(newPoints);
}


Proof ProofSet::computeProof(const GpuContext& gpu, uint32_t npower) const {
    util::Timer timer;

    if (npower == 0) {
        auto B0 = load2(E, 0);
        return Proof{E, std::move(B0), {}, knownFactors};
    }

    power = npower;
    rebuildPointsForPower(power);

    std::vector<std::vector<uint32_t>> middles;
    std::vector<uint64_t> hashes;

    auto B = load2(E, power);
    auto hash = Proof::hashWords(E, B);

    // Online binary-tree reduction requires only stack depth O(power).
    // A newly pushed residue may temporarily occupy one extra slot.
    const uint32_t maxBuffers = power + 1u;
    std::vector<cl_mem> bufferPool(maxBuffers);

    cl_context cl_ctx = gpu.ctx.getContext();
    for (uint32_t i = 0; i < maxBuffers; ++i) {
        cl_int err;
        bufferPool[i] = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, gpu.limbBytes, nullptr, &err);
        if (err != CL_SUCCESS) {
            for (uint32_t j = 0; j < i; ++j) clReleaseMemObject(bufferPool[j]);
            throw std::runtime_error("Failed to create GPU buffer for proof computation");
        }
    }

    // Releases the buffers on every exit, including a GPU error thrown from the
    // transforms (the caller may retry at a lower power).
    struct PoolGuard {
        std::vector<cl_mem>& pool;
        ~PoolGuard() { for (cl_mem m : pool) clReleaseMemObject(m); }
    } poolGuard{bufferPool};

    for (uint32_t p = 0; p < power; ++p) {
        assert(p == hashes.size());

        const uint32_t s = (1u << (power - p - 1));
        const uint32_t levelBuffers = (1u << p);
        uint32_t bufIndex = 0;

        for (uint32_t i = 0; i < levelBuffers; ++i) {
            const uint32_t checkpointIndex = s * (i * 2 + 1) - 1;
            if (checkpointIndex >= points.size()) {
                throw std::runtime_error("Missing checkpoint index");
            }

            const uint32_t iteration = points[checkpointIndex];
            if (iteration > E) {
                throw std::runtime_error("Invalid checkpoint iteration");
            }
            if (!shouldCheckpoint2(iteration, power)) {
                throw std::runtime_error("Missing checkpoint file");
            }

            if (bufIndex >= maxBuffers) {
                throw std::runtime_error(
                    "Proof reduction stack exceeded O(power) buffer bound");
            }

            auto w = load2(iteration, power);
            gpu.write(bufferPool[bufIndex], w);
            ++bufIndex;

            for (uint32_t k = 0; (i & (1u << k)) != 0; ++k) {
                assert(k <= p - 1);
                if (bufIndex < 2) {
                    throw std::runtime_error("Insufficient buffers for reduction");
                }
                --bufIndex;
                uint64_t h = hashes[p - 1 - k];
                gpu.ntt.powInPlace(bufferPool[bufIndex - 1], bufferPool[bufIndex - 1], h, gpu.carry, gpu.limbBytes);
                gpu.ntt.mulInPlace5(bufferPool[bufIndex - 1], bufferPool[bufIndex], gpu.carry, gpu.limbBytes);
            }
        }

        if (bufIndex != 1) {
            throw std::runtime_error("Invalid buffer reduction at level");
        }

        auto levelResult = gpu.read(bufferPool[0]);
        // gpu.read always returns ceil(E/32) words, so test the value: a proof
        // middle is never 0 modulo 2^E - 1 (every term is a power of 3).
        if (util::isZeroResidue(levelResult, E)) {
            throw std::runtime_error("Read ZERO during proof generation");
        }

        middles.push_back(levelResult);
        hash = Proof::hashWords(E, hash, levelResult);
        uint64_t newHash = hash[0];
        hashes.push_back(newHash);

        uint64_t middleRes64 = Proof::res64(levelResult);
        std::cout << "proof [" << p << "] : M " << std::hex << std::setfill('0') << std::setw(16) << middleRes64
                  << ", h " << std::setw(16) << newHash << std::dec << std::endl;
    }

    double elapsed = timer.elapsed();
    std::cout << "Proof generated in " << std::fixed << std::setprecision(2) << elapsed << " seconds." << std::endl;

    return Proof{E, std::move(B), std::move(middles), knownFactors};
}



double ProofSet::diskUsageGB(uint32_t E, uint32_t power) {
  // Calculate disk usage in GB for proof files
  // Formula from PRPLL: ldexp(E, -33 + int(power)) * 1.05
  if (power == 0) return 0.0;
  return std::ldexp(static_cast<double>(E), -33 + static_cast<int>(power)) * 1.05;
}

void GpuContext::write(cl_mem buffer, const std::vector<uint32_t>& data) const {
  std::vector<uint64_t> gpu_data = io::JsonBuilder::expandBits(data, digitWidth);
  
  // Ensure we have the correct size for the GPU buffer
  size_t numWords = limbBytes / sizeof(uint64_t);
  if (gpu_data.size() != numWords) {
    gpu_data.resize(numWords, 0);
  }
  
  cl_int err = clEnqueueWriteBuffer(ctx.getQueue(), buffer, CL_TRUE, 0, limbBytes, gpu_data.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to upload data to GPU buffer");
  }
}

std::vector<uint32_t> GpuContext::read(cl_mem buffer) const {
  size_t numWords = limbBytes / sizeof(uint64_t);
  std::vector<uint64_t> gpu_data(numWords);
  cl_int err = clEnqueueReadBuffer(ctx.getQueue(), buffer, CL_TRUE, 0, limbBytes, gpu_data.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to download data from GPU buffer");
  }
  
  return io::JsonBuilder::compactBits(gpu_data, digitWidth, exponent);

}

} // namespace core
