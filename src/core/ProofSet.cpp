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
#include "io/Sha3Hash.h"
#include "util/Crc32.hpp"
#include "util/Timer.hpp"
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
ProofSet::ProofSet(uint32_t exponent, uint32_t proofLevel, std::vector<std::string> factors)
  : E{exponent}, power{proofLevel}, knownFactors{std::move(factors)} {
  // Create proof directory
  std::filesystem::create_directories(proofPath(E));

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

void ProofSet::save(uint32_t iter, const std::vector<uint32_t>& words) {
  if (!shouldCheckpoint(iter)) {
    return;
  }

  // Create the file path for this iteration
  auto filePath = proofPath(E) / std::to_string(iter);
  
  // Write the words data to file
  std::ofstream file(filePath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot create proof checkpoint file: " + filePath.string());
  }
  
  // Write CRC32 first, then the data
  uint32_t crc = computeCRC32(words.data(), words.size() * sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
  file.write(reinterpret_cast<const char*>(words.data()), 
             words.size() * sizeof(uint32_t));
  
  if (!file.good()) {
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

bool ProofSet::isInPoints(uint32_t E, uint32_t power, uint32_t k) {
  if (k == E) { return true; } // special-case E
  uint32_t start = 0;
  for (uint32_t p = 0, span = (E + 1) / 2; p < power; ++p, span = (span + 1) / 2) {
    assert(k >= start);
    if (k > start + span) {
      start += span;
    } else if (k == start + span) {
      return true;
    }
  }
  return false;
}

std::filesystem::path ProofSet::proofPath(uint32_t E) {
  return std::filesystem::path(std::to_string(E)) / "proof";
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
  auto filePath = proofPath(E) / std::to_string(k);
  return std::filesystem::exists(filePath);
}

std::vector<uint32_t> ProofSet::load(uint32_t iter) const {
  if (!shouldCheckpoint(iter)) {
    throw std::runtime_error("Attempt to load non-checkpoint iteration: " + std::to_string(iter));
  }

  auto filePath = proofPath(E) / std::to_string(iter);
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

Proof ProofSet::computeProof(const GpuContext& gpu) const {
  // Start timing proof generation
  util::Timer timer;

  std::vector<std::vector<uint32_t>> middles;
  std::vector<uint64_t> hashes;

  // Initial hash of the final residue B
  auto B = load(E);
  auto hash = Proof::hashWords(E, B);

  // Pre-allocate maximum needed buffer pool (power levels use 2^p buffers max)
  uint32_t maxBuffers = (1u << power);
  std::vector<cl_mem> bufferPool(maxBuffers);
  
  // Get OpenCL context for buffer creation
  cl_context cl_ctx = gpu.ctx.getContext();
  
  // Initialize GPU buffers
  for (uint32_t i = 0; i < maxBuffers; ++i) {
    cl_int err;
    bufferPool[i] = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, gpu.limbBytes, nullptr, &err);
    if (err != CL_SUCCESS) {
      // Clean up on error
      for (uint32_t j = 0; j < i; ++j) {
        clReleaseMemObject(bufferPool[j]);
      }
      throw std::runtime_error("Failed to create GPU buffer for proof computation");
    }
  }

  // Main computation loop
  for (uint32_t p = 0; p < power; ++p) {
    assert(p == hashes.size());
    
    uint32_t s = (1u << (power - p - 1)); // Step size for this level
    uint32_t levelBuffers = (1u << p); // Number of buffers needed for this level
    uint32_t bufIndex = 0;
    
    // Load residues and apply binary tree algorithm
    for (uint32_t i = 0; i < levelBuffers; ++i) {
      // PRPLL's formula: load checkpoint at points[s * (i * 2 + 1) - 1]
      uint32_t checkpointIndex = s * (i * 2 + 1) - 1;
      
      if (checkpointIndex >= points.size()) {
        continue;
      }
      
      uint32_t iteration = points[checkpointIndex];
      
      if (iteration > E || !shouldCheckpoint(iteration)) {
        continue;
      }
      
      auto w = load(iteration);
      gpu.write(bufferPool[bufIndex], w);
      bufIndex++;
      
      // Apply hashes from previous levels
      for (uint32_t k = 0; i & (1u << k); ++k) {
        assert(k <= p - 1);
        if (bufIndex < 2) {
          std::cerr << "Error: need at least 2 buffers for expMul, have " << bufIndex << std::endl;
          continue;
        }
        
        bufIndex--;
        uint64_t h = hashes[p - 1 - k]; // Hash from previous level
        
        // (bufIndex-1) := (bufIndex-1)^h * bufIndex
        gpu.ntt.powInPlace(bufferPool[bufIndex - 1], bufferPool[bufIndex - 1], h, gpu.carry, gpu.limbBytes);
        gpu.ntt.mulInPlace5(bufferPool[bufIndex - 1], bufferPool[bufIndex], gpu.carry, gpu.limbBytes);
      }
    }
    
    if (bufIndex != 1) {
      std::cerr << "Warning: expected bufIndex=1, got " << bufIndex << std::endl;
    }
    
    // Convert the final result to words format

    auto levelResult = gpu.read(bufferPool[0]);
    
    if (levelResult.empty()) {
      throw std::runtime_error("Read ZERO during proof generation at level " + std::to_string(p));
    }
    
    // Store the result as middle for this level
    middles.push_back(levelResult);
    
    // Update hash chain with this level's middle
    hash = Proof::hashWords(E, hash, levelResult);
    uint64_t newHash = hash[0]; // The first 64 bits of the hash
    hashes.push_back(newHash);
    
    // Show middle and hash for the current level
    uint64_t middleRes64 = Proof::res64(levelResult);
    std::cout << "proof [" << p << "] : M " << std::hex << std::setfill('0') << std::setw(16) << middleRes64 
              << ", h " << std::setw(16) << newHash << std::dec << std::endl;
  }
  
  // Clean up GPU buffers
  for (uint32_t i = 0; i < maxBuffers; ++i) {
    clReleaseMemObject(bufferPool[i]);
  }
  
  // Display proof generation time
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
  std::vector<uint64_t> gpu_data = io::JsonBuilder::expandBits(data, digitWidth, exponent);
  
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
