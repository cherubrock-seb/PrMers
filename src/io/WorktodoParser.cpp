// io/WorktodoParser.cpp
#include "io/WorktodoParser.hpp"
#include "math/Cofactor.hpp"
#include "util/StringUtils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <vector>
#include <optional>
#include <limits>
#include <cctype>
#include <algorithm>

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

std::vector<std::string> splitRespectingQuotes(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::string current;
  bool inQuotes = false;
  for (char c : s) {
    if (c == '"') { inQuotes = !inQuotes; current += c; }
    else if (c == delim && !inQuotes) { result.push_back(current); current.clear(); }
    else { current += c; }
  }
  if (!current.empty()) result.push_back(current);
  return result;
}

static std::vector<std::string> parseFactors(const std::string& factorStr) {
    std::vector<std::string> factors;
    std::string trimmed = factorStr;
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back())))
        trimmed.pop_back();
    if (trimmed.size() >= 2 && trimmed.front() == '"' && trimmed.back() == '"') {
        std::string content = trimmed.substr(1, trimmed.size() - 2);
        factors = util::split(content, ',');
    }
    return factors;
}

static bool isQuoted(const std::string& s) {
    return s.size() >= 2 && s.front() == '"' && s.back() == '"';
}

static bool isIntegerToken(const std::string& s) {
    if (s.empty()) return false;
    size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    if (i == s.size()) return false;
    for (; i < s.size(); ++i) if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    return true;
}

static void trim_inplace(std::string& s){
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) { s.clear(); return; }
    s = s.substr(a, b - a + 1);
}
/*
static uint64_t mul_sat_u64(uint64_t a, uint64_t b){
    if (a == 0 || b == 0) return 0;
    if (a > std::numeric_limits<uint64_t>::max() / b) return std::numeric_limits<uint64_t>::max();
    return a * b;
}*/

std::optional<WorktodoEntry> WorktodoParser::parse() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Cannot open " << filename_ << "\n";
        return std::nullopt;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::string trimmedLine = line;
        trim_inplace(trimmedLine);
        if (trimmedLine.empty() || trimmedLine[0] == '#' || trimmedLine[0] == ';') continue;

        auto top = util::split(trimmedLine, '=');
        if (top.size() < 2) continue;

        std::string keyword = top[0];
        trim_inplace(keyword);
        std::string keywordUpper = keyword;
        std::transform(keywordUpper.begin(), keywordUpper.end(), keywordUpper.begin(),
                       [](unsigned char c){ return static_cast<char>(std::toupper(c)); });

        bool isPRP  = (keywordUpper == "PRP" || keywordUpper == "PRPDC");
        bool isLL   = (keywordUpper == "TEST" || keywordUpper == "DOUBLECHECK");
        bool isDoubleCheck = (keywordUpper == "DOUBLECHECK");
        bool isPF   = (keywordUpper == "PFACTOR");
        bool isPM1  = (keywordUpper == "PMINUS1");
        bool isECM2 = (keywordUpper == "ECM2");
        bool isGMPRP = (keywordUpper == "GMPRP");
        bool isGMPROTH = (keywordUpper == "GMPROTH" || keywordUpper == "GMTEST");
        bool isGMPM1 = (keywordUpper == "GMPMINUS1" || keywordUpper == "GMPM1");
        bool isGMECM = (keywordUpper == "GMECM");
        bool isGMCHAIN = (keywordUpper == "GMCHAIN" || keywordUpper == "GMCAMPAIGN");
        if (!(isPRP || isLL || isPF || isPM1 || isECM2 ||
              isGMPRP || isGMPROTH || isGMPM1 || isGMECM || isGMCHAIN)) continue;

        auto parts = splitRespectingQuotes(top[1], ',');
        if (!parts.empty() && (parts[0].empty() || parts[0] == "N/A"))
            parts.erase(parts.begin());

        std::string aid;
        if (!parts.empty() && (isHex(parts[0]) || parts[0] == "AID" || parts[0] == "N/A")) {
            aid = parts[0];
            parts.erase(parts.begin());
        }

        // Prime95-compatible LL worktodo formats:
        //   Test=exponent[,how_far_factored[,has_been_pminus1ed]]
        //   DoubleCheck=exponent[,how_far_factored[,has_been_pminus1ed]]
        // Assignment IDs may precede the exponent and are stripped above:
        //   DoubleCheck=<32-hex-aid>,85473391,76,1
        // Keep supporting the older PrMers/Proth-style k,b,n,c form:
        //   Test=1,2,n,-1
        const bool looksLikeKbncMersenne =
            (parts.size() >= 4 && parts[0] == "1" && parts[1] == "2" && parts[3] == "-1");
        if (isLL && !looksLikeKbncMersenne && !parts.empty() && isIntegerToken(parts[0])) {
            try {
                uint64_t exp74 = std::stoull(parts[0]);
                if (exp74 == 0 || exp74 > std::numeric_limits<uint32_t>::max()) continue;

                WorktodoEntry entry;
                entry.llTest      = true;
                entry.doubleCheck = isDoubleCheck;
                entry.exponent    = static_cast<uint32_t>(exp74);
                entry.rawLine     = line;
                entry.aid         = aid;

                if (parts.size() >= 2) {
                    std::string sdepth = parts[1];
                    trim_inplace(sdepth);
                    if (!sdepth.empty() && !isQuoted(sdepth)) entry.sieveDepth = std::stod(sdepth);
                }
                if (parts.size() >= 3) {
                    std::string spm1 = parts[2];
                    trim_inplace(spm1);
                    if (!spm1.empty() && !isQuoted(spm1)) entry.pminus1ed = (std::stoul(spm1) != 0);
                }

                std::cout << "Loaded entry: " << (entry.doubleCheck ? "DoubleCheck" : "Test")
                          << " exponent=" << entry.exponent
                          << (aid.empty() ? "" : " (AID=" + aid + ")")
                          << "\n";
                if (entry.sieveDepth > 0.0) {
                    std::cout << "Trial factoring completed to: 2^" << entry.sieveDepth << "\n";
                }
                std::cout << "P-1 pretest flag: " << (entry.pminus1ed ? 1 : 0) << "\n";
                return entry;
            } catch (...) {
                continue;
            }
        }

        try {

            // Native PrMers Gaussian-Mersenne worktodo formats. These are
            // deliberately separate from Prime95 syntax because the target is
            // G_p = 2^p - (2/p)2^((p+1)/2) + 1, not M_p.
            //
            //   GMPROTH=p[,sieve_limit]
            //   GMPRP=p[,sieve_limit]
            //   GMPMINUS1=p,B1,B2[,base[,sieve_limit[,chunk_bits]]]
            //   GMECM=p,B1,B2,curves[,sigma[,sieve_limit[,chunk_bits]]]
            //   GMCHAIN=p,pm1_B1,pm1_B2[,ecm_B1[,ecm_B2[,curves[,sieve_limit[,chunk_bits]]]]]
            // GMCHAIN is conditional: a P-1/ECM factor stops the line; otherwise
            // PrMers continues to the deterministic Proth test.
            if (isGMPRP || isGMPROTH || isGMPM1 || isGMECM || isGMCHAIN) {
                for (std::string& part : parts) trim_inplace(part);
                const size_t required = (isGMPRP || isGMPROTH) ? 1 :
                                        ((isGMPM1 || isGMCHAIN) ? 3 : 4);
                if (parts.size() < required || !isIntegerToken(parts[0])) continue;

                const uint64_t p64 = std::stoull(parts[0]);
                if (p64 < 3 || p64 > std::numeric_limits<uint32_t>::max()) continue;

                WorktodoEntry entry;
                entry.gaussianMersenne = true;
                entry.gmPrpOnly = isGMPRP;
                entry.gmPipeline = isGMCHAIN;
                entry.pm1Test = isGMPM1 || isGMCHAIN;
                entry.ecmTest = isGMECM;
                entry.prpTest = isGMPRP || isGMPROTH || isGMCHAIN;
                entry.exponent = static_cast<uint32_t>(p64);
                entry.rawLine = line;

                if (isGMPRP || isGMPROTH) {
                    if (parts.size() >= 2 && !parts[1].empty())
                        entry.gmSieveLimit = std::stoull(parts[1]);
                } else if (isGMPM1) {
                    entry.B1 = std::stoull(parts[1]);
                    entry.B2 = std::stoull(parts[2]);
                    if (parts.size() >= 4 && !parts[3].empty())
                        entry.gmBase = static_cast<uint32_t>(std::stoul(parts[3]));
                    if (parts.size() >= 5 && !parts[4].empty())
                        entry.gmSieveLimit = std::stoull(parts[4]);
                    if (parts.size() >= 6 && !parts[5].empty())
                        entry.gmFactorChunkBits = std::stoull(parts[5]);
                } else if (isGMCHAIN) {
                    entry.B1 = std::stoull(parts[1]);
                    entry.B2 = std::stoull(parts[2]);
                    if (parts.size() >= 4 && !parts[3].empty()) entry.gmEcmB1 = std::stoull(parts[3]);
                    if (parts.size() >= 5 && !parts[4].empty()) entry.gmEcmB2 = std::stoull(parts[4]);
                    if (parts.size() >= 6 && !parts[5].empty()) entry.gmEcmCurves = std::stoull(parts[5]);
                    if (parts.size() >= 7 && !parts[6].empty()) entry.gmSieveLimit = std::stoull(parts[6]);
                    if (parts.size() >= 8 && !parts[7].empty()) entry.gmFactorChunkBits = std::stoull(parts[7]);
                } else {
                    entry.B1 = std::stoull(parts[1]);
                    entry.B2 = std::stoull(parts[2]);
                    entry.curves = std::stoull(parts[3]);
                    if (entry.curves == 0) entry.curves = 1;
                    if (parts.size() >= 5 && !parts[4].empty() && parts[4] != "0")
                        entry.sigma = parts[4];
                    if (parts.size() >= 6 && !parts[5].empty())
                        entry.gmSieveLimit = std::stoull(parts[5]);
                    if (parts.size() >= 7 && !parts[6].empty())
                        entry.gmFactorChunkBits = std::stoull(parts[6]);
                }

                const char* kind = isGMPRP ? "GMPRP" : isGMPROTH ? "GMPROTH" :
                                   isGMPM1 ? "GMPMINUS1" : isGMCHAIN ? "GMCHAIN" : "GMECM";
                std::cout << "Loaded entry: " << kind << " exponent=" << entry.exponent;
                if (isGMPM1 || isGMECM || isGMCHAIN)
                    std::cout << " B1=" << entry.B1 << " B2=" << entry.B2;
                if (isGMECM) std::cout << " curves=" << entry.curves;
                if (isGMCHAIN) std::cout << " ecm=" << entry.gmEcmB1 << "/" << entry.gmEcmB2
                                          << " curves=" << entry.gmEcmCurves;
                std::cout << " sieve=" << entry.gmSieveLimit;
                if (entry.gmFactorChunkBits != 0)
                    std::cout << " chunk_bits=" << entry.gmFactorChunkBits;
                std::cout << "\n";
                return entry;
            }

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

                entry.B1 = static_cast<uint64_t>(std::stoull(parts[4]));
                entry.B2 = static_cast<uint64_t>(std::stod(parts[5]));

                if (parts.size() >= 7) {
                    std::vector<std::string> kf;
                    auto q = parseFactors(parts.back());
                    if (!q.empty()) {
                        kf = std::move(q);
                    } else {
                        for (size_t i = 6; i < parts.size(); ++i) {
                            std::string s = parts[i];
                            if (!s.empty() && s.front() == '"' && s.back() == '"')
                                s = s.substr(1, s.size() - 2);
                            trim_inplace(s);
                            if (!s.empty()) kf.push_back(std::move(s));
                        }
                    }
                    if (!kf.empty()) entry.knownFactors = std::move(kf);
                }

                std::cout << "Loaded entry: PFactor exponent=" << entry.exponent
                          << " B1=" << entry.B1 << " B2=" << entry.B2
                          << (aid.empty() ? "" : " (AID=" + aid + ")") << "\n";
                if (!entry.knownFactors.empty()) {
                    std::cout << "Known factors: ";
                    for (size_t i = 0; i < entry.knownFactors.size(); ++i) {
                        if (i) std::cout << ", ";
                        std::cout << entry.knownFactors[i];
                    }
                    std::cout << "\n";
                }
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

                entry.B1 = static_cast<uint64_t>(std::stoull(parts[4]));
                entry.B2 = static_cast<uint64_t>(std::stoull(parts[5]));

                // Prime95-compatible Pminus1 format:
                // Pminus1=k,b,n,c,B1,B2[,how_far_factored][,B2_start][,"factors"]
                // If an assignment id is present, it has already been removed above.
                size_t next = 6;
                if (next < parts.size() && !isQuoted(parts[next])) {
                    std::string s = parts[next];
                    trim_inplace(s);
                    if (!s.empty()) entry.sieveDepth = std::stod(s);
                    ++next;
                }
                if (next < parts.size() && !isQuoted(parts[next])) {
                    std::string s = parts[next];
                    trim_inplace(s);
                    if (!s.empty()) entry.B2Start = static_cast<uint64_t>(std::stoull(s));
                    ++next;
                }
                if (next < parts.size()) {
                    auto factors = parseFactors(parts[next]);
                    if (!factors.empty()) entry.knownFactors = std::move(factors);
                }

                std::cout << "Loaded entry: Pminus1 exponent=" << entry.exponent
                          << " B1=" << entry.B1 << " B2=" << entry.B2
                          << (aid.empty() ? "" : " (AID=" + aid + ")") << "\n";
                if (entry.sieveDepth > 0.0) {
                    std::cout << "Trial factoring completed to: 2^" << entry.sieveDepth << "\n";
                }
                if (entry.B2Start > 0) {
                    std::cout << "Stage 2 start bound: " << entry.B2Start << "\n";
                }
                if (!entry.knownFactors.empty()) {
                    std::cout << "Known factors: ";
                    for (size_t i = 0; i < entry.knownFactors.size(); ++i) {
                        if (i) std::cout << ", ";
                        std::cout << entry.knownFactors[i];
                    }
                    std::cout << "\n";
                }
                return entry;
            }

            if (isECM2) {
                if (parts.size() < 7) continue;

                const std::string k = parts[0], b = parts[1], nstr = parts[2], c = parts[3];
                if (k != "1" || b != "2" || c != "-1") continue;

                uint32_t exp = static_cast<uint32_t>(std::stoul(nstr));
                if (exp == 0) continue;

                uint64_t B1 = static_cast<uint64_t>(std::stoull(parts[4]));
                uint64_t B2 = static_cast<uint64_t>(std::stoull(parts[5]));
                uint64_t curves = static_cast<uint64_t>(std::stoull(parts[6]));
                if (curves == 0) curves = 1;

                if (B2 == 0 || B2 == B1) B2 = 0;//mul_sat_u64(B1, 100);

                WorktodoEntry entry;
                entry.ecmTest   = true;
                entry.exponent  = exp;
                entry.rawLine   = line;
                entry.aid       = aid;
                entry.B1        = B1;
                entry.B2        = B2;
                entry.curves    = curves;

                if (parts.size() >= 8) {
                    std::vector<std::string> kf;
                    auto q = parseFactors(parts.back());
                    if (!q.empty()) {
                        kf = std::move(q);
                    } else {
                        for (size_t i = 7; i < parts.size(); ++i) {
                            std::string s = parts[i];
                            if (!s.empty() && s.front() == '"' && s.back() == '"')
                                s = s.substr(1, s.size() - 2);
                            trim_inplace(s);
                            if (!s.empty()) kf.push_back(std::move(s));
                        }
                    }
                    if (!kf.empty()) {
                        if (math::Cofactor::validateFactors(exp, kf)) {
                            entry.knownFactors = std::move(kf);
                        } else {
                            continue;
                        }
                    }
                }

                std::cout << "Loaded entry: ECM2 exponent=" << entry.exponent
                          << " B1=" << entry.B1 << " B2=" << entry.B2
                          << " curves=" << entry.curves
                          << (aid.empty() ? "" : " (AID=" + aid + ")") << "\n";
                if (!entry.knownFactors.empty()) {
                    std::cout << "Known factors: ";
                    for (size_t i = 0; i < entry.knownFactors.size(); ++i) {
                        if (i) std::cout << ", ";
                        std::cout << entry.knownFactors[i];
                    }
                    std::cout << "\n";
                }
                return entry;
            }

            if (parts.size() < 4) continue;
            size_t idx = 0;
            const std::string k = parts[idx++], b = parts[idx++], nstr = parts[idx++], c = parts[idx++];
            if (k != "1" || b != "2" || c != "-1") continue;

            uint32_t exp = static_cast<uint32_t>(std::stoul(nstr));
            if (exp == 0) continue;

            WorktodoEntry entry;
            entry.prpTest   = isPRP;
            entry.llTest    = isLL;
            entry.exponent  = exp;
            entry.rawLine   = line;
            entry.aid       = aid;
            std::cout << "Loaded entry: " << (entry.prpTest ? "PRP" : "LL")
                      << " exponent=" << entry.exponent
                      << (aid.empty() ? "" : " (AID=" + aid + ")")
                      << "\n";

            int prpBase = 0;
            int residueType = 0;

            (void) prpBase;
            if (idx < parts.size() && !isQuoted(parts[idx]) && isIntegerToken(parts[idx])) {
                idx++;
                if (idx < parts.size() && !isQuoted(parts[idx]) && isIntegerToken(parts[idx])) {
                    idx++;
                }
            }

            if ((idx + 1) < parts.size()
                && !isQuoted(parts[idx]) && isIntegerToken(parts[idx])
                && !isQuoted(parts[idx+1]) && isIntegerToken(parts[idx+1])) {
                prpBase = std::stoi(parts[idx]);      idx++;
                residueType = std::stoi(parts[idx]);  idx++;
            }

            if (idx < parts.size() && isQuoted(parts.back()) && isPRP) {
                auto factors = parseFactors(parts.back());
                if (!factors.empty() && math::Cofactor::validateFactors(exp, factors)) {
                    entry.knownFactors = std::move(factors);
                    entry.residueType = static_cast<uint32_t>((residueType != 0) ? residueType : 5);
                    std::cout << "Known factors: ";
                    for (size_t i = 0; i < entry.knownFactors.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << entry.knownFactors[i];
                    }
                    std::cout << std::endl;
                } else {
                    continue;
                }
            } else if (residueType != 0) {
                entry.residueType = static_cast<uint32_t>(residueType);
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

bool WorktodoParser::removeProcessedLine(const std::string& rawLine) {
    std::ifstream inFile(filename_);
    std::ofstream tempFile(filename_ + ".tmp");
    std::ofstream saveFile("worktodo_save.txt", std::ios::app);
    if (!inFile || !tempFile || !saveFile) return false;

    std::string expected = rawLine;
    trim_inplace(expected);
    std::string line;
    bool removed = false;
    while (std::getline(inFile, line)) {
        std::string trimmed = line;
        trim_inplace(trimmed);
        if (!removed && trimmed == expected) {
            removed = true;
            saveFile << line << "\n";
            continue;
        }
        tempFile << line << "\n";
    }

    inFile.close();
    tempFile.close();
    saveFile.close();
    if (!removed) {
        std::remove((filename_ + ".tmp").c_str());
        return false;
    }
    if (std::remove(filename_.c_str()) != 0 ||
        std::rename((filename_ + ".tmp").c_str(), filename_.c_str()) != 0) {
        return false;
    }
    return true;
}

bool WorktodoParser::removeFirstProcessed() {
    std::ifstream inFile(filename_);
    std::ofstream tempFile(filename_ + ".tmp");
    std::ofstream saveFile("worktodo_save.txt", std::ios::app);
    if (!inFile || !tempFile || !saveFile) return false;

    std::string line;
    bool skipped = false;
    while (std::getline(inFile, line)) {
        std::string trimmed = line;
        trim_inplace(trimmed);
        const bool actionable = !trimmed.empty() && trimmed[0] != '#' && trimmed[0] != ';';
        if (!skipped && actionable) {
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
