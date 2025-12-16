#include "io/JsonBuilder.hpp"
#include "io/CliParser.hpp"
#include "math/Cofactor.hpp"
#include "util/GmpUtils.hpp"
#include "core/Version.hpp"
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include "io/MD5.h"
#include "util/Crc32.hpp"
#include <cstdint>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <tuple>
#include <cctype>
#include <regex>
#include <string>

namespace io{

static uint32_t mod3(const std::vector<uint32_t>& W) {
    uint32_t r = 0;
    for (uint32_t w : W) r = (r + (w % 3)) % 3;
    return r;
}

static void doDiv3(uint32_t E, std::vector<uint32_t>& W) {
    uint32_t r = (3 - mod3(W)) % 3;
    int topBits = E % 32;
    {
        uint64_t t = (uint64_t(r) << topBits) + W.back();
        W.back() = uint32_t(t / 3);
        r        = uint32_t(t % 3);
    }
    for (auto it = W.rbegin() + 1; it != W.rend(); ++it) {
        uint64_t t = (uint64_t(r) << 32) + *it;
        *it       = uint32_t(t / 3);
        r         = uint32_t(t % 3);
    }
}

static void doDiv9(uint32_t E, std::vector<uint32_t>& W) {
    doDiv3(E, W);
    doDiv3(E, W);
}

static std::string toLower(const std::string& s){
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });
    return t;
}

static std::string toUpper(const std::string& s){
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c){ return char(std::toupper(c)); });
    return t;
}

std::tuple<bool, std::string, std::string> JsonBuilder::computeResult(
    const std::vector<uint64_t>& hostResult,
    const CliOptions& opts,
    const std::vector<int>& digit_width) {
    mpz_class Mp = (mpz_class(1) << opts.exponent) - 1;
    mpz_class finalResidue = util::vectToMpz(hostResult, digit_width, Mp);
    bool isPrime;
    if (opts.mode == "prp") {
        mpz_class inv9;
        mpz_invert(inv9.get_mpz_t(), mpz_class(9).get_mpz_t(), Mp.get_mpz_t());
        finalResidue = (finalResidue * inv9) % Mp;
        if (!opts.knownFactors.empty()) isPrime = math::Cofactor::isCofactorPRP(opts.exponent, opts.knownFactors, finalResidue);
        else isPrime = (finalResidue == 1);
    } else {
        isPrime = (finalResidue == 0);
    }
    mpz_class mod2_64 = mpz_class(1) << 64;
    mpz_class mod2_2048 = mpz_class(1) << 2048;
    mpz_class res64_val = finalResidue % mod2_64;
    mpz_class res2048_val = finalResidue % mod2_2048;
    std::ostringstream oss64;
    oss64 << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << res64_val;
    std::string res64 = oss64.str();
    std::ostringstream oss2048;
    oss2048 << std::hex << std::nouppercase << std::setw(512) << std::setfill('0') << res2048_val;
    std::string res2048 = oss2048.str();
    return std::make_tuple(isPrime, res64, res2048);
}

std::tuple<bool, std::string, std::string> JsonBuilder::computeResultMarin(
    const std::vector<uint64_t>& hostResult,
    const CliOptions& opts)
{
    std::vector<uint64_t> digits(hostResult.size());
    std::vector<int> digit_width(hostResult.size());
    for (size_t i = 0; i < hostResult.size(); ++i) {
        uint64_t x = hostResult[i];
        uint32_t v = static_cast<uint32_t>(x);
        uint32_t w = static_cast<uint8_t>(x >> 32);
        digit_width[i] = static_cast<int>(w);
        if (w == 0) digits[i] = 0;
        else if (w >= 32) digits[i] = static_cast<uint64_t>(v);
        else digits[i] = static_cast<uint64_t>(v & ((1u << w) - 1));
    }
    mpz_class Mp = (mpz_class(1) << opts.exponent) - 1;
    mpz_class finalResidue = util::vectToMpz(digits, digit_width, Mp);
    bool isPrime;
    if (opts.mode == "prp") {
        mpz_class inv9;
        mpz_invert(inv9.get_mpz_t(), mpz_class(9).get_mpz_t(), Mp.get_mpz_t());
        finalResidue = (finalResidue * inv9) % Mp;
        if (!opts.knownFactors.empty()) isPrime = math::Cofactor::isCofactorPRP(opts.exponent, opts.knownFactors, finalResidue);
        else isPrime = (finalResidue == 1);
    } else {
        isPrime = (finalResidue == 0);
    }
    mpz_class res64_val = finalResidue & ((mpz_class(1) << 64) - 1);
    mpz_class res2048_val = finalResidue & ((mpz_class(1) << 2048) - 1);
    std::string s64 = res64_val.get_str(16);
    if (s64.size() < 16) s64.insert(s64.begin(), 16 - s64.size(), '0');
    for (auto& c : s64) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    std::string s2048 = res2048_val.get_str(16);
    if (s2048.size() < 512) s2048.insert(s2048.begin(), 512 - s2048.size(), '0');
    return std::make_tuple(isPrime, s64, s2048);
}

std::vector<uint32_t> JsonBuilder::compactBits(
    const std::vector<uint64_t>& x,
    const std::vector<int>&      digit_width,
    uint32_t                     E
) {
    uint32_t digitCount = static_cast<uint32_t>(x.size());
    uint32_t totalWords = (E - 1) / 32 + 1;
    std::vector<uint32_t> out(totalWords, 0u);
    int      carry    = 0;
    uint32_t outWord  = 0;
    int      haveBits = 0;
    uint32_t o        = 0;
    for (uint32_t p = 0; p < digitCount; ++p) {
        int      w   = digit_width[p];
        uint64_t v64 = uint64_t(carry) + x[p];
        carry        = int(v64 >> w);
        uint32_t v   = uint32_t(v64 & ((1ULL << w) - 1));
        int topBits = 32 - haveBits;
        outWord |= v << haveBits;
        if (w >= topBits) {
            out[o++]   = outWord;
            outWord    = (w > topBits) ? (v >> topBits) : 0;
            haveBits   = w - topBits;
        } else {
            haveBits  += w;
        }
    }
    if (haveBits > 0 || carry) {
        out[o++] = outWord;
        for (uint32_t i = 1; carry && i < o; ++i) {
            uint64_t sum = static_cast<uint64_t>(out[i]) + static_cast<uint64_t>(carry);
            out[i]       = uint32_t(sum & 0xFFFFFFFFu);
            carry        = int(sum >> 32);
        }
    }
    return out;
}

std::vector<uint64_t> JsonBuilder::expandBits(
    const std::vector<uint32_t>& compactWords,
    const std::vector<int>&      digit_width
) {
    uint32_t digitCount = static_cast<uint32_t>(digit_width.size());
    std::vector<uint64_t> out(digitCount, 0);
    uint64_t bitBuffer = 0;
    int      availableBits = 0;
    uint32_t wordIndex = 0;
    for (uint32_t p = 0; p < digitCount; ++p) {
        int w = digit_width[p];
        while (availableBits < w && wordIndex < compactWords.size()) {
            bitBuffer |= (uint64_t(compactWords[wordIndex]) << availableBits);
            availableBits += 32;
            wordIndex++;
        }
        uint64_t mask = (1ULL << w) - 1;
        out[p] = bitBuffer & mask;
        bitBuffer >>= w;
        availableBits -= w;
    }
    return out;
}

static std::string fileMD5(const std::string& filePath) {
    namespace fs = std::filesystem;
    FILE* f = nullptr;
#ifdef _WIN32
    fopen_s(&f, filePath.c_str(), "rb");
#else
    f = fopen(filePath.c_str(), "rb");
#endif
    if (!f) return "";
    char buf[65536];
    MD5 h;
    while (!feof(f)) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        if (n > 0) h.update(buf, (unsigned)n);
    }
    fclose(f);
    return std::move(h).finish();
}

static std::string jsonEscape(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b";  break;
            case '\f': o << "\\f";  break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:   o << c;      break;
        }
    }
    return "\"" + o.str() + "\"";
}

static std::string generatePrimeNetJson(
    const CliOptions &opts,
    const std::string &status,
    unsigned int exponent,
    const std::string &worktype,
    const std::string &res64,
    const std::string &res2048,
    int residueType,
    int gerbiczError,
    unsigned int fftLength,
    int proofVersion,
    int proofPower,
    int proofHashSize,
    const std::string &proofMd5,
    const std::string &programName,
    const std::string &programVersion,
    unsigned int programPort,
    const std::string &osName,
    const std::string &osArchitecture,
    const std::string &user,
    const std::string &aid,
    const std::string &timestamp,
    const std::string &computer,
    const std::vector<std::string>& knownFactors)
{
    std::string canonWT;
    if (worktype == "prp") canonWT = "PRP-3";
    else if (worktype == "ll" || worktype == "llsafe") canonWT = "LL";
    else canonWT = toUpper(worktype);
    bool isEdw = opts.compute_edwards;
    int torsion = opts.notorsion ? 0 : (opts.torsion16 ? 16 : 8);
    std::ostringstream oss;

    std::vector<std::string> uniqFactors;
    uniqFactors.reserve(knownFactors.size());
    for (const auto& f : knownFactors) {
        if (std::find(uniqFactors.begin(), uniqFactors.end(), f) == uniqFactors.end()) {
            uniqFactors.push_back(f);
        }
    }

    std::string knownFactorStr;        // e.g. 204453287,1255070097822910516312001       -- for use in checksum
    std::string knownFactorStrQuoted;  // e.g. "204453287","1255070097822910516312001"   -- for use in JSON, note is already quoted and MUST NOT be wrapped in jsonEscape()
    if (!uniqFactors.empty()) {
        knownFactorStr       =        uniqFactors[0];
        knownFactorStrQuoted = "\"" + uniqFactors[0];
        for (size_t i = 1; i < uniqFactors.size(); ++i) {
            knownFactorStr       += ","     + uniqFactors[i];
            knownFactorStrQuoted += "\",\"" + uniqFactors[i];
        }
        knownFactorStrQuoted += "\"";
    }

    oss << "{";
    oss <<  "\"status\":"                          << jsonEscape(status);
    oss << ",\"exponent\":"                        <<            exponent;
    oss << ",\"worktype\":"                        << jsonEscape(canonWT);
    if (!knownFactors.empty()) {
        // *** TODO: this is totally wrong, "known-factors" and "factors" are two ENTIRELY separate things, and should be better stored in the PrMers data structure
        if ((worktype == "ll") || (worktype == "llsafe") || (worktype == "prp")) {
            oss << ",\"known-factors\":[" << knownFactorStrQuoted << "]";
        } else {
            oss << ",\"factors\":[" << knownFactorStrQuoted << "]";
        }
    }
    if (opts.B1 > 0) {
        oss << ",\"b1\":" << opts.B1;
        if (opts.B2 > opts.B1) {
            oss << ",\"b2\":" << opts.B2;
        }
    }
    if ((worktype == "ll") || (worktype == "llsafe") || (worktype == "prp")) {
        oss << ",\"res64\":"                       << jsonEscape(res64);
        if (worktype == "prp") {
            oss << ",\"res2048\":"                 << jsonEscape(res2048);
            oss << ",\"residue-type\":"            <<            residueType;
        }
        oss << ",\"errors\":{\"gerbicz\":"         <<            gerbiczError << "}";
        oss << ",\"shift-count\":0";
    } else if (worktype == "ecm") {
        if (opts.curves_tested_for_found > 0) {
            oss << ",\"curves\":"                  << opts.curves_tested_for_found;
        }
        oss << ",\"curve-type\":" << jsonEscape(isEdw ? "Edwards" : "Montgomery");
        oss << ",\"torsion-subgroup\":"            << torsion;
        if (!opts.sigma_hex.empty()) {
            oss << ",\"sigma-hex\":"                   << jsonEscape(opts.sigma_hex);
        }
        oss << ",\"curve-seed\":"                  <<            opts.curve_seed;
        oss << ",\"base-seed\":"                   <<            opts.curve_seed;
        oss << ",\"errors\":{\"invariant\":"         <<            opts.invarianterror << "}";
    }
    else if (worktype == "pm1") {
        oss << ",\"errors\":{\"gerbicz\":"         <<            gerbiczError << "}";
        if (opts.nmax) {
            oss << ",\"crandall-nK\":{\"n\":" << opts.nmax << ",\"K\":" << opts.K << "}";
        }
    }
    if (fftLength > 0) oss << ",\"fft-length\":"  <<            fftLength;
    if (!proofMd5.empty()) {
        oss << ",\"proof\":{"
            <<    "\"version\":"                   <<            proofVersion
            <<   ",\"power\":"                     <<            proofPower
            <<   ",\"hashsize\":"                  <<            proofHashSize
            <<   ",\"md5\":"                       << jsonEscape(proofMd5)
            << "}";
    }
    oss << ",\"program\":{"
        <<    "\"name\":\"prmers\""
        <<   ",\"version\":" << jsonEscape(core::PRMERS_VERSION)
        <<   ",\"port\":"    <<            programPort
        << "}"; // close "program"
    oss << ",\"os\":{"
        <<   "\"os\":"            << jsonEscape(osName);
    if (!osArchitecture.empty() && (osArchitecture != "unknown")) {
        oss << ",\"architecture\":" << jsonEscape(osArchitecture);
    }
    oss << "}"; // close "os"
    if (!user.empty())     oss << ",\"user\":"     << jsonEscape(user);
    if (!computer.empty()) oss << ",\"computer\":" << jsonEscape(computer);
    if (!aid.empty())      oss << ",\"aid\":"      << jsonEscape(aid);
    oss << ",\"timestamp\":"                       << jsonEscape(timestamp);
    std::string prefix = oss.str();

    std::ostringstream canon;
    canon << exponent << ";";                                    // exponent
    std::string canonWTNorm = canonWT;
    if (canonWTNorm == "PRP-3" || canonWTNorm == "prp-3") canonWTNorm = "PRP";
    canon << canonWTNorm  << ";";                                // worktype
    if (!knownFactors.empty()) {
        canon << knownFactorStr;                                 // factors
    }
    canon << ";";
    canon << "" << ";";                                          // known-factors *** TODO: not yet supported (factors that were known BEFORE this factoring run and included in worktodo.txt, distinct from factors just found ***

    if (canonWT == "TF") {
        // *** TODO: not yet not yet implemented ***
        canon << "<BITLO>"         << ";";                       // bitlo (e.g. 68)
        canon << "<BITHI>"         << ";";                       // bithi (e.g. 75)
        canon << "<RANGECOMPLETE>" << ";";                       // rangecomplete (0 or 1)
    } else if (canonWT == "PRP-3") {
        canon << toLower(res64)    << ";";                       // res64
        canon << toLower(res2048)  << ";";                       // res2048
        canon << "0" << "_"                                      // shift-count
              << "3" << "_"                                      // prp-base
              << residueType       << ";";                       // residue-type
    } else if (canonWT == "LL") {
        canon << toLower(res64)    << ";";                       // res64
        canon << ""                << ";";                       // unused
        canon << "0"               << ";";                       // shift-count
    } else if (canonWT == "ECM") {
        canon << opts.B1           << ";";                       // b1
        if (opts.B2 > opts.B1) canon << opts.B2;                 // b2
        canon << ";";

        if (isEdw) canon << "E";                                 // sigma
        if (!opts.sigma_hex.empty()) {
            canon << "0x" << toLower(opts.sigma_hex);
        } else if (opts.sigma) {
            canon << opts.sigma;
        }
        if (torsion > 0) {
            canon << "_TSG" << torsion;                          // torsion-subgroup
        }
        canon << ";";
    } else if (canonWT == "PM1") {
        canon << opts.B1           << ";";                       // b1
        if (opts.B2 > opts.B1) canon << opts.B2;                 // b2
        canon << ";";
        if (opts.nmax) {                                         // Crandall-nK
            canon << opts.nmax << "_" << opts.K;
        }
        canon << ";";
    } else if (canonWT == "PP1") {
        // *** TODO: not yet not yet implemented ***
        canon << opts.B1           << ";";                       // b1
        if (opts.B2 > opts.B1) canon << opts.B2;                 // b2
        canon << ";";
        canon << "<START>" << ";";                               // P+1 start
    }

    canon << fftLength << ";";                                   // fft-length
    if (worktype != "ecm") {
        canon << "gerbicz:" << gerbiczError << ";";                  // errorsObjectStringified
    }
    else{
        canon << "invariant:" << gerbiczError << ";"; 
    }
    canon << programName << ";";                                 // program.name
    canon << programVersion << ";";                              // program.version
    canon << "" << ";";                                          // program.kernel|program.subversion
    canon << "" << ";";                                          // program.details
    canon << osName << ";";                                      // os.os
    canon << osArchitecture << ";";                              // os.architecture
    canon << timestamp;                                          // timestamp
    unsigned int crc = computeCRC32(canon.str());
    std::ostringstream hexss;
    hexss << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc;
    oss.str(""); oss.clear();
    oss << prefix
        << ",\"checksum\":{\"version\":1,\"checksum\":\"" << hexss.str() << "\"}"
        //<< ",hash:\"" << canon.str() << "\""
        //<< ",\"code-hash\":" << jsonEscape(util::code_hash_crc32_upper8())
        << "}";
    return oss.str();
}

std::string JsonBuilder::generate(const CliOptions& opts,
                                  int transform_size,
                                  bool isPrime,
                                  const std::string& res64,
                                  const std::string& res2048)
{
    std::time_t now = std::time(nullptr);
    std::tm timeinfo;
    #ifdef _WIN32
        gmtime_s(&timeinfo, &now);
    #else
        std::tm* tmp = std::gmtime(&now);
        if (tmp != nullptr) timeinfo = *tmp;
    #endif
    char timestampBuf[32];
    std::strftime(timestampBuf, sizeof(timestampBuf), "%Y-%m-%d %H:%M:%S", &timeinfo);

    std::string status;
    if ((opts.mode == "ll") || (opts.mode == "llsafe") || (opts.mode == "prp")) {
        status = (isPrime ? "P" : "C");
    } else {
        status = (!opts.knownFactors.empty() ? "F" : "NF");
    }
    int residueType = opts.knownFactors.empty() ? 1 : 5;
    return generatePrimeNetJson(
        opts,
        status,
        opts.exponent,
        opts.mode,
        res64,
        res2048,
        residueType,
        opts.gerbicz_error_count,
        static_cast<unsigned>(transform_size),
        opts.proof ? 2 : 0,
        static_cast<int>(opts.proof ? opts.proofPower : 0u),
        opts.proof ? 64 : 0,
        opts.proof ? fileMD5(opts.proofFile) : "",
        "prmers",
        core::PRMERS_VERSION,
        static_cast<unsigned>(opts.portCode),
        opts.osName,
        opts.osArch,
        opts.user.empty() ? "" : opts.user,
        opts.aid,
        timestampBuf,
        opts.computer_name,
        opts.knownFactors
    );
}

std::string JsonBuilder::computeRes64(
    const std::vector<uint64_t>& x,
    const CliOptions& opts,
    const std::vector<int>& digit_width,
    double,
    int)
{
    auto words = JsonBuilder::compactBits(x, digit_width, opts.exponent);
    if (opts.mode == "prp") doDiv9(opts.exponent, words);
    uint64_t finalRes64 = (uint64_t(words[1]) << 32) | words[0];
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << finalRes64;
    return oss.str();
}

std::string JsonBuilder::computeRes64Iter(
    const std::vector<uint64_t>& x,
    const CliOptions& opts,
    const std::vector<int>& digit_width,
    double,
    int)
{
    auto words = JsonBuilder::compactBits(x, digit_width, opts.exponent);
    uint64_t finalRes64 = (uint64_t(words[1]) << 32) | words[0];
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << finalRes64;
    return oss.str();
}

std::string JsonBuilder::computeRes2048(
    const std::vector<uint64_t>& x,
    const CliOptions& opts,
    const std::vector<int>& digit_width,
    double,
    int)
{
    auto words = JsonBuilder::compactBits(x, digit_width, opts.exponent);
    if (opts.mode == "prp") doDiv9(opts.exponent, words);
    std::ostringstream oss;
    for (int i = 63; i >= 0; --i) {
        oss << std::hex << std::nouppercase << std::setw(8) << std::setfill('0') << words[static_cast<size_t>(i)];
    }
    return oss.str();
}

void JsonBuilder::write(const std::string& json,
                        const std::string& path)
{
    std::ofstream out(path);
    out << json;
}


/*
// James says this is neat code that should NEVER be made available to the public

static bool j_get_str(const std::string& s, const char* key, std::string& out){
    std::string pat = "\\\"" + std::string(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"";
    std::regex re(pat);
    std::smatch m;
    if (std::regex_search(s, m, re)) { out = m[1].str(); return true; }
    return false;
}

static bool j_get_uint(const std::string& s, const char* key, unsigned& out){
    std::string pat = "\\\"" + std::string(key) + "\\\"\\s*:\\s*(\\d+)";
    std::regex re(pat);
    std::smatch m;
    if (std::regex_search(s, m, re)) { out = (unsigned)std::stoul(m[1].str()); return true; }
    return false;
}

static bool j_get_int(const std::string& s, const char* key, int& out){
    std::string pat = "\\\"" + std::string(key) + "\\\"\\s*:\\s*(-?\\d+)";
    std::regex re(pat);
    std::smatch m;
    if (std::regex_search(s, m, re)) { out = std::stoi(m[1].str()); return true; }
    return false;
}

static void j_get_program(const std::string& s, std::string& name, std::string& version){
    std::regex block(R"("program"\s*:\s*\{([^}]*)\})");
    std::smatch m;
    if (std::regex_search(s, m, block)) {
        const std::string body = m[1].str();
        j_get_str(body, "name", name);
        j_get_str(body, "version", version);
    }
}

static void j_get_os(const std::string& s, std::string& osName, std::string& osArch){
    std::regex block(R"("os"\s*:\s*\{([^}]*)\})");
    std::smatch m;
    if (std::regex_search(s, m, block)) {
        const std::string body = m[1].str();
        j_get_str(body, "os", osName);
        j_get_str(body, "architecture", osArch);
    }
}

static void j_get_errors(const std::string& s, int& gerbicz){
    std::regex block(R"("errors"\s*:\s*\{([^}]*)\})");
    std::smatch m;
    if (std::regex_search(s, m, block)) {
        const std::string body = m[1].str();
        j_get_int(body, "gerbicz", gerbicz);
    }
}

static std::vector<std::string> j_get_known_factors(const std::string& s){
    std::vector<std::string> out;
    std::smatch m;
    if (std::regex_search(s, m, std::regex(R"REGEX("known-factors"\s*:\s*\[([^]]*)\])REGEX"))) {
        std::string arr = m[1].str();
        std::regex item(R"ITEM("([^"]*)")ITEM");
        for (auto it = std::sregex_iterator(arr.begin(), arr.end(), item);
             it != std::sregex_iterator(); ++it) out.emplace_back((*it)[1].str());
    }
    return out;
}


static std::string buildCanonicalStringFromSubmitted(
    unsigned exponent,
    const std::string& worktype,
    const std::string& res64,
    const std::string& res2048,
    int residueType,
    int gerbiczError,
    unsigned fftLength,
    const std::string& programName,
    const std::string& programVersion,
    const std::string& osName,
    const std::string& osArch,
    const std::string& timestamp,
    const std::vector<std::string>& knownFactors
){
    std::string canonWT = worktype;
    if (canonWT.rfind("PRP", 0) == 0) canonWT = "PRP";
    else if (canonWT.rfind("LL", 0) == 0) canonWT = "LL";
    std::ostringstream canon;
    canon << exponent << ";";
    canon << canonWT  << ";";
    canon << ""       << ";";
    if (!knownFactors.empty()){
        std::string k = knownFactors[0];
        for (size_t i = 1; i < knownFactors.size(); ++i) k += "," + knownFactors[i];
        canon << k;
    }
    canon << ";";
    if (canonWT == "PRP") {
        canon << toLower(res64)   << ";";
        canon << toLower(res2048) << ";";
        canon << "0" << "_" << "3" << "_" << residueType << ";";
    } else if (canonWT == "LL") {
        canon << toLower(res64)   << ";";
        canon << ""               << ";";
        canon << "0"              << ";";
    }
    canon << fftLength << ";";
    canon << "gerbicz:" << gerbiczError << ";";
    canon << programName    << ";";
    canon << programVersion << ";";
    canon << "" << ";";
    canon << "" << ";";
    canon << osName << ";";
    canon << osArch << ";";
    canon << timestamp;
    return canon.str();
}

std::string recomputeChecksumFromSubmittedJson(const std::string& json)
{
    unsigned exponent = 0, fftLen = 0;
    int residueType = 0, gerbicz = 0;
    std::string worktype, res64, res2048, programName, programVersion, osName, osArch, timestamp;
    j_get_uint(json, "exponent", exponent);
    j_get_str (json, "worktype", worktype);
    j_get_str (json, "res64",    res64);
    j_get_str (json, "res2048",  res2048);
    j_get_int (json, "residue-type", residueType);
    j_get_uint(json, "fft-length", fftLen);
    j_get_str (json, "timestamp", timestamp);
    j_get_program(json, programName, programVersion);
    j_get_os(json, osName, osArch);
    j_get_errors(json, gerbicz);
    auto knownFactors = j_get_known_factors(json);
    const std::string canon =
        buildCanonicalStringFromSubmitted(exponent, worktype, res64, res2048,
                                          residueType, gerbicz, fftLen,
                                          programName, programVersion,
                                          osName, osArch, timestamp,
                                          knownFactors);
    unsigned crc = computeCRC32(canon);
    std::ostringstream hexss;
    hexss << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc;
    return hexss.str();
}

std::string rewriteChecksumInSubmittedJson(const std::string& json)
{
    const std::string newSum = recomputeChecksumFromSubmittedJson(json);

    std::regex chkStrict(R"("checksum"\s*:\s*\{\s*"version"\s*:\s*\d+\s*,\s*"checksum"\s*:\s*"[^"]*"\s*\})");
    if (std::regex_search(json, chkStrict)) {
        return std::regex_replace(json, chkStrict,
               std::string("\"checksum\":{\"version\":1,\"checksum\":\"") + newSum + "\"}");
    }

    std::regex chkAny(R"("checksum"\s*:\s*\{[^}]*\})");
    if (std::regex_search(json, chkAny)) {
        return std::regex_replace(json, chkAny,
               std::string("\"checksum\":{\"version\":1,\"checksum\":\"") + newSum + "\"}");
    }

    std::string out = json;
    auto pos = out.find_last_of('}');
    if (pos != std::string::npos) {
        std::string ins = std::string(",\"checksum\":{\"version\":1,\"checksum\":\"") + newSum + "\"}";
        out.insert(pos, ins);
    }
    return out;
}
*/

} // namespace io
