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
#include "io/Sha3Hash.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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
  file << "NUMBER=M" << E << "\n";

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
      exponent = std::stoul(value.substr(1));
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

  return Proof(exponent, std::move(B), std::move(middles));
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
  hasher.update(prefix.data(), prefix.size() * sizeof(uint64_t));
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

} // namespace core
