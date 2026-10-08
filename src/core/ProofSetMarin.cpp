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
#include "core/ProofSetMarin.hpp"
#include "core/ProofCheckpoint.hpp"
#include "io/Sha3Hash.h"
#include "util/Crc32.hpp"
#include "util/Timer.hpp"
#include "util/GmpUtils.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace core {

// Words
WordsMarin::WordsMarin() = default;

WordsMarin::WordsMarin(const std::vector<uint64_t>& v)
  : data_{v} {}

const std::vector<uint64_t>& WordsMarin::data() const noexcept {
    return data_;
}

std::vector<uint64_t>& WordsMarin::data() noexcept {
    return data_;
}

WordsMarin WordsMarin::fromUint64(const std::vector<uint64_t>& host, uint32_t exponent) {
    (void)exponent;
    return WordsMarin(host);
}

// ProofSetMarin
ProofSetMarin::ProofSetMarin(uint32_t exponent, uint32_t proofLevel, std::vector<std::string> factors)
  : E{exponent}, power{proofLevel}, knownFactors{std::move(factors)} {
  if(exponent%2!=0){
      assert(E & 1); // E is supposed to be prime
    
    points = proofPoints(E, power);
    assert(points.size() == (1u << power));
    assert(points.front() > 0 && points.back() == E);

    points.push_back(uint32_t(-1)); // guard element

    // Verify all points are valid
    for (uint32_t p : points) {
      assert(p > E || isInPoints(E, power, p));
    }
  }
}

std::vector<uint32_t> ProofSetMarin::proofPoints(uint32_t E, uint32_t power) {
  // Checkpoint points using the binary tree structure.
  std::vector<uint32_t> pts;
  pts.push_back(0);
  for (uint32_t p = 0, span = (E + 1) / 2; p < power; ++p, span = (span + 1) / 2) {
    for (uint32_t i = 0, end = static_cast<uint32_t>(pts.size()); i < end; ++i) {
      pts.push_back(pts[i] + span);
    }
  }
  pts.front() = E;
  std::sort(pts.begin(), pts.end());
  return pts;
}

void ProofSetMarin::setPower(uint32_t newPower) {
  power = newPower;
  if (E % 2 != 0) {
    points = proofPoints(E, power);
    points.push_back(uint32_t(-1)); // guard element
  }
}

bool ProofSetMarin::shouldCheckpoint(uint32_t iter) const {
  return isInPoints(E, power, iter);
}

void ProofSetMarin::save(uint32_t iter, const std::vector<uint32_t>& words) {
  if (!shouldCheckpoint(iter)) {
    return;
  }

  // The directory is created with the first residue, so that tests that make
  // no proof (LL, P-1, ECM, -proof 0) leave nothing behind.
  std::error_code dirError;
  std::filesystem::create_directories(proofPath(E), dirError);

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
          static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
  
  file.close();
  if (file.fail() || !syncFileToDisk(filePath)) {
    throw std::runtime_error("Error writing proof checkpoint file: " + filePath.string());
  }
}

WordsMarin ProofSetMarin::fromUint64(const std::vector<uint64_t>& host, uint32_t exponent) {
    return WordsMarin::fromUint64(host, exponent);
}

uint32_t ProofSetMarin::bestPower(uint32_t E) {
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

bool ProofSetMarin::isInPoints(uint32_t E, uint32_t power, uint32_t k) {
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

std::filesystem::path ProofSetMarin::proofPath(uint32_t E) {
  return std::filesystem::path(std::to_string(E)) / "proof";
}

void ProofSetMarin::clearResidues(uint32_t E) {
  std::error_code ec;
  std::filesystem::remove_all(proofPath(E), ec);
  // Only the (now empty) exponent directory; anything else in it stays.
  std::filesystem::remove(std::filesystem::path(std::to_string(E)), ec);
}

ProofSetMarin::ResidueAction ProofSetMarin::residueAction(
    bool isPrp, bool wagstaff, bool proofRequested, bool proofCompleted,
    bool resultSaved, bool entryRetired) {
  if (!isPrp || wagstaff)
    return ResidueAction::NotApplicable;
  if (!resultSaved)
    return ResidueAction::KeepResultNotSaved;
  if (!entryRetired)
    return ResidueAction::KeepEntryNotRetired;
  if (proofRequested && !proofCompleted)
    return ResidueAction::KeepProofFailed;
  return ResidueAction::Clear;
}

std::string ProofSetMarin::residuesKeptMessage(uint32_t E, ResidueAction action) {
  std::error_code ec;
  auto dir = std::filesystem::absolute(proofPath(E), ec);
  if (ec)
    dir = proofPath(E);
  std::string why;
  switch (action) {
    case ResidueAction::KeepResultNotSaved:
      why = "the result could not be saved, so the test will be rerun";
      break;
    case ResidueAction::KeepEntryNotRetired:
      why = "the worktodo entry could not be removed, so the test will be rerun";
      break;
    case ResidueAction::KeepProofFailed:
      why = "no verified proof was made, so a later run can retry it";
      break;
    default:
      return std::string();
  }
  return "Proof residues kept in " + dir.string() + " (" + why +
         "). They take several GB; delete that directory by hand when you no longer need them.";
}

bool ProofSetMarin::fileExists(uint32_t E, uint32_t k) {
  // CRC32 followed by ceil(E/32) 32-bit words.
  std::error_code ec;
  const auto size = std::filesystem::file_size(proofPath(E) / std::to_string(k), ec);
  return !ec && size == sizeof(uint32_t) * (1u + (static_cast<uint64_t>(E) + 31) / 32);
}

std::vector<uint32_t> ProofSetMarin::loadResidue(uint32_t E, uint32_t iter) {
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

std::vector<uint32_t> ProofSetMarin::load(uint32_t iter) const {
  if (!shouldCheckpoint(iter)) {
    throw std::runtime_error("Attempt to load non-checkpoint iteration: " + std::to_string(iter));
  }
  return loadResidue(E, iter);
}

bool ProofSetMarin::canDo(uint32_t E, uint32_t power, uint32_t currentK) {
  std::map<uint32_t, bool> checked;
  return canDo(E, power, currentK, checked);
}

bool ProofSetMarin::canDo(uint32_t E, uint32_t power, uint32_t currentK,
                          std::map<uint32_t, bool>& checked) {
  // Every residue below E from iterations up to and including currentK must
  // be there (a residue at currentK itself is written before the test can be
  // saved at currentK), with the right size and a matching CRC: the proof
  // reads all of them, so one damaged residue makes the proof impossible.
  // This reads each of them once when a test is resumed (power 10 at the
  // wavefront: about 1000 files and up to the full proof disk usage).
  for (uint32_t point : proofPoints(E, power)) {
    if (point >= E || point > currentK) break;
    auto it = checked.find(point);
    if (it == checked.end()) {
      bool ok = fileExists(E, point);
      if (ok) {
        try {
          loadResidue(E, point);
        } catch (const std::exception&) {
          ok = false;
        }
      }
      it = checked.emplace(point, ok).first;
    }
    if (!it->second) return false;
  }
  return true;
}

uint32_t ProofSetMarin::effectivePower(uint32_t E, uint32_t power, uint32_t currentK) {
  // The points of a lower power are a subset of those of a higher one, so
  // each residue is checked at most once across the powers tried.
  std::map<uint32_t, bool> checked;
  for (uint32_t p = power; p > 0; --p) {
    if (canDo(E, p, currentK, checked)) return p;
  }
  return 0;
}

ProofMarin ProofSetMarin::computeProof() const {
  // Start timing proof generation
  util::Timer timer;

  std::vector<std::vector<uint32_t>> middles;
  std::vector<uint64_t> hashes;

  // Initial hash of the final residue B
  auto B = load(E);
  auto hash = ProofMarin::hashWords(E, B);

  // Pre-allocate maximum needed buffer pool (power levels use 2^p buffers max)
  uint32_t maxBuffers = (1u << power);
  std::vector<mpz_class> bufferPool(maxBuffers);

  // Main computation loop
  for (uint32_t p = 0; p < power; ++p) {
    assert(p == hashes.size());
    
    uint32_t s = (1u << (power - p - 1)); // Step size for this level
    uint32_t levelBuffers = (1u << p); // Number of buffers needed for this level
    uint32_t bufIndex = 0;
    
    // Clear buffers that will be used for this level
    for (uint32_t i = 0; i < levelBuffers; ++i) {
      bufferPool[i] = 0;
    }
    
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
      bufferPool[bufIndex] = util::convertToGMP(w);
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
        
        // PRPLL's expMul: (bufIndex-1) := (bufIndex-1)^h * bufIndex
        mpz_class temp = util::mersennePowMod(bufferPool[bufIndex - 1], h, E); // A^h mod (2^E - 1)
        mpz_class result = temp * bufferPool[bufIndex]; // A^h * B
        bufferPool[bufIndex - 1] = util::mersenneReduce(result, E); // Optimized Mersenne reduction
        
        // Clear the consumed buffer
        bufferPool[bufIndex] = 0;
      }
    }
    
    if (bufIndex != 1) {
      std::cerr << "Warning: expected bufIndex=1, got " << bufIndex << std::endl;
    }
    
    if (bufferPool[0] == 0) {
      throw std::runtime_error(
          "Read ZERO during proof generation at level " +
          std::to_string(p));
    }

    // Canonical proof residue: exactly ceil(E/32) little-endian words.
    auto levelResult = util::convertFromGMP(bufferPool[0]);
    const size_t expectedWords =
        (static_cast<size_t>(E) + 31u) / 32u;

    if (levelResult.size() > expectedWords) {
      throw std::runtime_error(
          "Proof residue exceeds exponent width at level " +
          std::to_string(p));
    }

    levelResult.resize(expectedWords, 0u);
    
    // Store the result as middle for this level
    middles.push_back(levelResult);
    
    // Update hash chain with this level's middle
    hash = ProofMarin::hashWords(E, hash, levelResult);
    uint64_t newHash = hash[0]; // The first 64 bits of the hash
    hashes.push_back(newHash);
    
    // Show middle and hash for the current level
    uint64_t middleRes64 = ProofMarin::res64(levelResult);
    std::cout << "proof [" << p << "] : M " << std::hex << std::setfill('0') << std::setw(16) << middleRes64 
              << ", h " << std::setw(16) << newHash << std::dec << std::endl;
  }
  
  // Display proof generation time
  double elapsed = timer.elapsed();
  std::cout << "Proof generated in " << std::fixed << std::setprecision(2) << elapsed << " seconds." << std::endl;
  
  return ProofMarin{E, std::move(B), std::move(middles), knownFactors};
}

double ProofSetMarin::diskUsageGB(uint32_t E, uint32_t power) {
  // Calculate disk usage in GB for proof files
  // Formula from PRPLL: ldexp(E, -33 + int(power)) * 1.05
  if (power == 0) return 0.0;
  return std::ldexp(static_cast<double>(E), -33 + static_cast<int>(power)) * 1.05;
}

} // namespace core