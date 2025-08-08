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
// io/WorktodoParser.cpp
#include "io/WorktodoParser.hpp"
#include "math/Cofactor.hpp"
#include "util/StringUtils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <vector>

namespace io {

WorktodoParser::WorktodoParser(const std::string& filename)
  : filename_(filename)
{}

static bool isHex(const std::string& s) {
    if (s.size() != 32) return false;
    for (char c : s)
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

// Split string respecting quoted sections (for PRP-CF assignment parsing)
std::vector<std::string> splitRespectingQuotes(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::string current;
  bool inQuotes = false;
  
  for (char c : s) {
    if (c == '"') {
      inQuotes = !inQuotes;
      current += c;
    } else if (c == delim && !inQuotes) {
      result.push_back(current);
      current.clear();
    } else {
      current += c;
    }
  }
  if (!current.empty()) {
    result.push_back(current);
  }
  return result;
}

// Parse comma-separated factors from quoted string like "36357263,145429049,8411216206439"
static std::vector<std::string> parseFactors(const std::string& factorStr) {
    std::vector<std::string> factors;
    
    // Remove leading/trailing whitespace and check for quotes
    std::string trimmed = factorStr;
    while (!trimmed.empty() && std::isspace(trimmed.back())) {
        trimmed.pop_back();
    }
    
    if (trimmed.size() >= 2 && trimmed.front() == '"' && trimmed.back() == '"') {
        // Remove quotes and split by comma
        std::string content = trimmed.substr(1, trimmed.size() - 2);
        factors = util::split(content, ',');
    }
    
    return factors;
}

std::optional<WorktodoEntry> WorktodoParser::parse() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Cannot open " << filename_ << "\n";
        return std::nullopt;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto top = util::split(line, '=');
        if (top.size() < 2) continue;

        bool isPRP  = (top[0] == "PRP" || top[0] == "PRPDC");
        bool isLL   = (top[0] == "Test" || top[0] == "DoubleCheck");
        bool isPF   = (top[0] == "PFactor");
        bool isPM1  = (top[0] == "Pminus1");
        if (!(isPRP || isLL || isPF || isPM1)) continue;

        auto parts = splitRespectingQuotes(top[1], ',');
        if (!parts.empty() && (parts[0].empty() || parts[0] == "N/A"))
            parts.erase(parts.begin());

        std::string aid;
        if (!parts.empty() && isHex(parts[0])) {
            aid = parts[0];
            parts.erase(parts.begin());
        }

        try {
            if (isPF) {
                if (parts.size() < 6) continue;
                if (parts[0] != "1" || parts[1] != "2" || parts[3] != "-1") continue;
                uint32_t exp = static_cast<uint32_t>(std::stoul(parts[2]));
                if (exp == 0) continue;
                WorktodoEntry entry;
                entry.pm1Test   = true;
                entry.exponent  = exp;
                entry.rawLine   = line;
                entry.aid       = aid;
                entry.B1        = static_cast<uint64_t>(std::stoull(parts[4]));
                entry.B2        = static_cast<uint64_t>(std::stod(parts[5])); 
                if (parts.size() >= 7) {
                    const std::string& lastPart = parts.back();
                    auto factors = parseFactors(lastPart);
                    if (!factors.empty()) {
                        entry.knownFactors = factors;
                    }
                }
                std::cout << "Loaded entry: PFactor exponent=" << entry.exponent
                          << " B1=" << entry.B1 << " B2=" << entry.B2
                          << (aid.empty() ? "" : " (AID=" + aid + ")") << "\n";
                return entry;
            }
            if (isPM1) {
                if (parts.size() < 6) continue;
                if (parts[0] != "1" || parts[1] != "2" || parts[3] != "-1") continue;
                uint32_t exp = static_cast<uint32_t>(std::stoul(parts[2]));
                if (exp == 0) continue;
                WorktodoEntry entry;
                entry.pm1Test   = true;
                entry.exponent  = exp;
                entry.rawLine   = line;
                entry.aid       = aid;
                entry.B1        = static_cast<uint64_t>(std::stoull(parts[4]));
                entry.B2        = static_cast<uint64_t>(std::stoull(parts[5]));
                if (parts.size() >= 7) {
                    const std::string& lastPart = parts.back();
                    auto factors = parseFactors(lastPart);
                    if (!factors.empty()) {
                        entry.knownFactors = factors;
                    }
                }
                std::cout << "Loaded entry: Pminus1 exponent=" << entry.exponent
                          << " B1=" << entry.B1 << " B2=" << entry.B2
                          << (aid.empty() ? "" : " (AID=" + aid + ")") << "\n";
                return entry;
            }

            if (parts.size() < 4
                || parts[0] != "1"
                || parts[1] != "2"
                || parts[3] != "-1")
                continue;

            uint32_t exp = static_cast<uint32_t>(std::stoul(parts[2]));
            if (exp == 0) continue;

            WorktodoEntry entry;
            entry.prpTest   = isPRP;
            entry.llTest    = isLL;
            entry.exponent  = exp;
            entry.rawLine   = line;
            entry.aid       = aid;
            std::cout << "Loaded entry: "
                      << (entry.prpTest ? "PRP" : "LL")
                      << " exponent=" << entry.exponent
                      << (aid.empty() ? "" : " (AID=" + aid + ")")
                      << "\n";
            if (isPRP && parts.size() == 7) {
                const std::string& lastPart = parts.back();
                auto factors = parseFactors(lastPart);
                if (!factors.empty() && math::Cofactor::validateFactors(exp, factors)) {
                    entry.knownFactors = factors;
                    entry.residueType = 5;
                    std::cout << "Known factors: ";
                    for (size_t i = 0; i < factors.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << factors[i];
                    }
                    std::cout << std::endl;
                }
                else {
                    continue;
                }
            }
            if (entry.llTest && !entry.knownFactors.empty()) {
                std::cerr << "Warning: Lucas-Lehmer test cannot be used on Mersenne cofactors." << std::endl;
                std::cerr << "Warning: Use PRP test for Mersenne cofactors instead." << std::endl;
                continue;
            }
            return entry;
        }
        catch (...) {
            continue;
        }
    }

    std::cerr << "No valid entry found in " << filename_ << "\n";
    return std::nullopt;
}


bool WorktodoParser::removeFirstProcessed() {
    std::ifstream inFile(filename_);
    std::ofstream tempFile(filename_ + ".tmp");
    std::ofstream saveFile("worktodo_save.txt", std::ios::app);

    if (!inFile || !tempFile || !saveFile)
        return false;

    std::string line;
    bool skipped = false;

    while (std::getline(inFile, line)) {
        if (!skipped && !line.empty()) {
            skipped = true;
            saveFile << line << "\n";
            continue;
        }
        tempFile << line << "\n";
    }

    inFile.close();
    tempFile.close();
    saveFile.close();

    std::remove(filename_.c_str());
    std::rename((filename_ + ".tmp").c_str(), filename_.c_str());

    return skipped;
}



} // namespace io
