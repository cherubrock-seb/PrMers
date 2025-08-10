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
#include "core/Proof.hpp"
#include "core/ProofSet.hpp"
#include "opencl/Context.hpp"
#include "opencl/NttEngine.hpp"
#include "math/Carry.hpp"
#include "io/Sha3Hash.h"
#include "util/Timer.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace core {

// Proof file I/O implementations
void Proof::save(const std::filesystem::path& filePath) const {
  std::ofstream file(filePath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot create proof file: " + filePath.string());
  }

  // Write ASCII header according to GIMPS specification
  file << "PRP PROOF\n";
  file << "VERSION=2\n";
  file << "HASHSIZE=64\n";
  file << "POWER=" << middles.size() << "\n";
  
  file << "NUMBER=M" << E;
  for (const auto& factor : knownFactors) {
    file << "/" << factor;
  }
  file << "\n";

  if (!file.good()) {
    throw std::runtime_error("Error writing proof file header: " + filePath.string());
  }

  // Helper function to write a residue
  auto writeResidue = [&](const std::vector<uint32_t>& residue) {
    uint32_t nBytes = (E - 1) / 8 + 1;
    
    // Write the residue data as bytes in little-endian format
    for (uint32_t byteIdx = 0; byteIdx < nBytes; ++byteIdx) {
      uint32_t wordIdx = byteIdx / 4;
      uint32_t byteInWord = byteIdx % 4;
      
      uint8_t byte = 0;
      if (wordIdx < residue.size()) {
        byte = static_cast<uint8_t>((residue[wordIdx] >> (byteInWord * 8)) & 0xFF);
      }
      file.write(reinterpret_cast<const char*>(&byte), 1);
    }
  };

  // Write final residue B first
  writeResidue(B);

  // Write all intermediate residues (middles)
  for (const auto& middle : middles) {
    writeResidue(middle);
  }

  if (!file.good()) {
    throw std::runtime_error("Error writing proof file data: " + filePath.string());
  }
}

Proof Proof::load(const std::filesystem::path& filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open proof file: " + filePath.string());
  }

  // Parse ASCII header
  std::string line;
  uint32_t version = 0, hashsize = 0, power = 0, exponent = 0;
  std::vector<std::string> factors;

  // Read and validate "PRP PROOF"
  if (!std::getline(file, line) || line != "PRP PROOF") {
    throw std::runtime_error("Invalid proof file header: " + filePath.string());
  }

  // Parse header fields (VERSION, HASHSIZE, POWER, NUMBER)
  // Binary data starts after NUMBER
  for (int i = 0; i < 4; ++i) {
    if (!std::getline(file, line)) {
      throw std::runtime_error("Incomplete header in proof file: " + filePath.string());
    }
    
    size_t eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      throw std::runtime_error("Malformed header line in proof file: " + line);
    }
    
    std::string key = line.substr(0, eq_pos);
    std::string value = line.substr(eq_pos + 1);
    
    if (key == "VERSION") {
      version = std::stoul(value);
    } else if (key == "HASHSIZE") {
      hashsize = std::stoul(value);
    } else if (key == "POWER") {
      power = std::stoul(value);
    } else if (key == "NUMBER" && !value.empty() && value[0] == 'M') {
      // Parse cofactor format: M18178631/36357263/145429049
      std::string numberStr = value.substr(1); // Remove 'M'
      size_t slashPos = numberStr.find('/');
      
      if (slashPos != std::string::npos) {
        // Cofactor format detected
        exponent = std::stoul(numberStr.substr(0, slashPos));
        
        // Parse factors separated by '/'
        std::string factorStr = numberStr.substr(slashPos + 1);
        size_t start = 0;
        size_t end = factorStr.find('/');
        
        while (end != std::string::npos) {
          factors.push_back(factorStr.substr(start, end - start));
          start = end + 1;
          end = factorStr.find('/', start);
        }
        factors.push_back(factorStr.substr(start)); // Last factor
      } else {
        // Mersenne number: M18178631
        exponent = std::stoul(numberStr);
      }
    } else {
      throw std::runtime_error("Unexpected header field: " + key);
    }
  }

  // Validate header values
  if (version != 2) {
    throw std::runtime_error("Unsupported proof file version: " + std::to_string(version));
  }
  if (hashsize != 64) {
    throw std::runtime_error("Unsupported hash size: " + std::to_string(hashsize));
  }
  if (power == 0 || power > 12) {
    throw std::runtime_error("Invalid proof power: " + std::to_string(power));
  }
  if (exponent == 0) {
    throw std::runtime_error("Invalid or missing exponent in proof file");
  }

  // Helper function to read a residue
  auto readResidue = [&]() -> std::vector<uint32_t> {
    uint32_t nBytes = (exponent - 1) / 8 + 1;
    uint32_t nWords = (exponent + 31) / 32;
    std::vector<uint32_t> data(nWords, 0);
    
    for (uint32_t byteIdx = 0; byteIdx < nBytes; ++byteIdx) {
      uint8_t byte;
      file.read(reinterpret_cast<char*>(&byte), 1);
      if (!file.good()) {
        throw std::runtime_error("Error reading residue data from proof file");
      }
      
      uint32_t wordIdx = byteIdx / 4;
      uint32_t byteInWord = byteIdx % 4;
      
      if (wordIdx < nWords) {
        data[wordIdx] |= (static_cast<uint32_t>(byte) << (byteInWord * 8));
      }
    }
    
    return data;
  };

  // Read final residue B
  auto B = readResidue();

  // Read intermediate residues (middles)
  std::vector<std::vector<uint32_t>> middles;
  for (uint32_t i = 0; i < power; ++i) {
    middles.push_back(readResidue());
  }

  return Proof(exponent, std::move(B), std::move(middles), std::move(factors));
}

// Hash functions for proof generation
std::array<uint64_t, 4> Proof::hashWords(uint32_t E, const std::vector<uint32_t>& words) {
  // Hash the words data
  io::SHA3 hasher;
  uint32_t nBytes = (E - 1) / 8 + 1;
  return std::move(hasher.update(words.data(), nBytes)).finish();
}

std::array<uint64_t, 4> Proof::hashWords(uint32_t E, 
                                         const std::array<uint64_t, 4>& prefix,
                                         const std::vector<uint32_t>& words) {
  // Hash the prefix first, then the words data
  io::SHA3 hasher;
  uint32_t nBytes = (E - 1) / 8 + 1;
  hasher.update(prefix.data(), static_cast<uint32_t>(prefix.size()) * sizeof(uint64_t));
  return std::move(hasher.update(words.data(), nBytes)).finish();
}

uint64_t Proof::res64(const std::vector<uint32_t>& words) {
  // Extract lower 64 bits from 32-bit words
  // Combines words[1] << 32 | words[0]
  if (words.empty()) return 0;
  
  uint64_t result = words[0];
  if (words.size() > 1) {
    result |= (static_cast<uint64_t>(words[1]) << 32);
  }
  return result;
}

bool Proof::verify(const GpuContext& gpu) const {
  uint32_t power = middles.size();
  if (power == 0) {
    throw std::runtime_error("Invalid proof: no middle residues");
  }

  // Initialize working values: A=3, B=finalResidue
  uint32_t nWords = (E + 31) / 32;
  std::vector<uint32_t> A(nWords, 0);
  A[0] = 3;
  auto B_work = B;

  // Allocate temporary GPU buffers
  cl_int err;
  cl_mem bufA = clCreateBuffer(gpu.ctx.getContext(), CL_MEM_READ_WRITE, gpu.limbBytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create GPU buffer A for verification");
    return false;
  }
  
  cl_mem bufB = clCreateBuffer(gpu.ctx.getContext(), CL_MEM_READ_WRITE, gpu.limbBytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create GPU buffer B for verification");
    clReleaseMemObject(bufA);
    return false;
  }

  // Start timing proof verification
  util::Timer timer;
  
  bool verificationResult = false;
  
  auto hash = hashWords(E, B);
  uint32_t span = E;

  std::cout << "Starting proof verification for M" << E;
  if (!knownFactors.empty()) {
    for (const auto& factor : knownFactors) {
      std::cout << "/" << factor;
    }
  }
  std::cout << " with power " << power << std::endl;

  for (uint32_t i = 0; i < power; ++i, span = (span + 1) / 2) {
    const auto& M = middles[i];
    hash = hashWords(E, hash, M);
    uint64_t h = hash[0];

    bool doSquareB = (span % 2 != 0);
    
    // B = M^h * (B^2 if span odd else B)
    gpu.write(bufB, B_work);
    if (doSquareB) {
      gpu.ntt.squareInPlace(bufB, gpu.carry, gpu.limbBytes);
    }
    gpu.write(bufA, M);
    gpu.ntt.powInPlace(bufA, bufA, h, gpu.carry, gpu.limbBytes);
    gpu.ntt.mulInPlace5(bufA, bufB, gpu.carry, gpu.limbBytes);
    B_work = gpu.read(bufA);

    // A = A^h * M  
    gpu.write(bufA, A);
    gpu.ntt.powInPlace(bufA, bufA, h, gpu.carry, gpu.limbBytes);
    gpu.write(bufB, M);
    gpu.ntt.mulInPlace5(bufA, bufB, gpu.carry, gpu.limbBytes);
    A = gpu.read(bufA);      
  }

  // Final step: A = A^(2^span)
  gpu.write(bufA, A);
  
  // Compute bufA^(2^span) by performing span consecutive squaring operations
  const uint32_t blockSize = 1000; // Process in blocks
  const uint32_t logStep = 10000;  // Log progress every logStep iterations
  
  uint32_t k = 0;
  while (k < span) {
    uint32_t its = std::min(blockSize, span - k);
    
    // Perform 'its' consecutive squaring operations
    for (uint32_t i = 0; i < its; ++i) {
      gpu.ntt.squareInPlace(bufA, gpu.carry, gpu.limbBytes);
    }
    
    k += its;
    
    // Flush GPU operations
    clFinish(gpu.ctx.getQueue());
    
    // Log progress for long operations
    if (span > logStep && k % logStep == 0) {
      std::cout << "Verification: " << k << " / " << span << " iterations completed" << std::endl;
    }
  }
  
  A = gpu.read(bufA);

  verificationResult = (A == B_work);
  std::cout << "Verification result: " << (verificationResult ? "SUCCESS" : "FAIL") << std::endl;

  // Display proof verification time
  double elapsed = timer.elapsed();
  std::cout << "Proof verified in " << std::fixed << std::setprecision(2) << elapsed << " seconds." << std::endl;

  clReleaseMemObject(bufA);
  clReleaseMemObject(bufB);

  return verificationResult;
}

} // namespace core
