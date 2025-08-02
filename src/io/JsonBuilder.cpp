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
#include "io/JsonBuilder.hpp"
#include "io/CliParser.hpp"          // for CliOptions
#include "math/Cofactor.hpp"
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


namespace io{
    //-----------------------------------------------------------------------------
    // mod3, doDiv3 et doDiv9 : pour PRP-3 on divise deux fois par 3
    //-----------------------------------------------------------------------------
    static uint32_t mod3(const std::vector<uint32_t>& W) {
        uint32_t r = 0;
        for (uint32_t w : W) r = (r + (w % 3)) % 3;
        return r;
    }

    static void doDiv3(uint32_t E, std::vector<uint32_t>& W) {
        uint32_t r = (3 - mod3(W)) % 3;
        int topBits = E % 32;
        // mot de poids fort
        {
            uint64_t t = (uint64_t(r) << topBits) + W.back();
            W.back() = uint32_t(t / 3);
            r        = uint32_t(t % 3);
        }
        // descente sur les mots inférieurs
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

std::tuple<bool, std::string, std::string> JsonBuilder::computeResult(
    const std::vector<uint64_t>& hostResult,
    const CliOptions& opts,
    const std::vector<int>& digit_width) {
    
    std::vector<uint32_t> words = JsonBuilder::compactBits(hostResult, digit_width, opts.exponent);
    if (opts.mode == "prp") {
        doDiv9(opts.exponent, words);
    }
    
    bool isPrime;
    if (opts.mode == "prp") {
        if (!opts.knownFactors.empty()) {
            // Mersenne cofactor PRP
            isPrime = math::Cofactor::isCofactorPRP(opts.exponent, opts.knownFactors, words);
        } else {
            // Mersenne number PRP
            isPrime = (hostResult[0] == 9
                       && std::all_of(hostResult.begin()+1,
                                      hostResult.end(),
                                      [](uint64_t v){ return v == 0; }));
        }
    } else {
        // Mersenne number LL
        isPrime = std::all_of(hostResult.begin(),
                              hostResult.end(),
                              [](uint64_t v){ return v == 0; });
    }
    
    // Ensure words array has correct size for residue computation
    if (words.size() < 64) {
        words.resize(64, 0);
    } else if (words.size() > 64) {
        words.resize(64);
    }
    
    uint64_t finalRes64 = (uint64_t(words[1]) << 32) | words[0];
    std::ostringstream oss64;
    oss64 << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << finalRes64;
    std::string res64 = oss64.str();
    
    std::ostringstream oss2048;
    oss2048 << std::hex << std::nouppercase << std::setfill('0');
    for (int i = 63; i >= 0; --i) {
        oss2048 << std::setw(8) << words[i];
    }
    std::string res2048 = oss2048.str();

    return std::make_tuple(isPrime, res64, res2048);
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
            uint64_t sum = uint64_t(out[i]) + carry;
            out[i]       = uint32_t(sum & 0xFFFFFFFFu);
            carry        = int(sum >> 32);
        }
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
    if (!f) {
        std::cerr << "\nWarning: Cannot open file for MD5: " << filePath << std::endl;
        return "";
    }

    char buf[65536];
    MD5 h;
    while (!feof(f)) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        if (n > 0) h.update(buf, (unsigned)n);
    }
    fclose(f);
    return std::move(h).finish();
}
// escape helper (from prmers.cpp)
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


// Full JSON generator (copy/paste from prmers.cpp) …
static std::string generatePrimeNetJson(
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
    const std::string &osVersion,
    const std::string &osArchitecture,
    const std::string &user,
    const std::string &aid,
    const std::string &uid,
    const std::string &timestamp,
    const std::string &computer)
{
    std::ostringstream oss;
    bool isPRP = (worktype.rfind("PRP", 0) == 0);

    oss << "{"
        << "\"status\":"       << jsonEscape(status)    << ","
        << "\"exponent\":"     << exponent              << ","
        << "\"worktype\":"     << jsonEscape(worktype)  << ","
        << "\"res64\":"        << jsonEscape(res64)     << ",";
    if (isPRP)
        oss << "\"res2048\":" << jsonEscape(res2048) << ",";
    oss << "\"residue-type\":" << residueType        << ","
        << "\"errors\":{\"gerbicz\":" << gerbiczError << "},"
        << "\"fft-length\":"  << fftLength            << ","
        << "\"shift-count\":0,";
    if (isPRP && !proofMd5.empty()) {
        oss << "\"proof\":{"
            << "\"version\":"   << proofVersion         << ","
            << "\"power\":"     << proofPower           << ","
            << "\"hashsize\":"  << proofHashSize        << ","
            << "\"md5\":"       << jsonEscape(proofMd5)
            << "},";
    }
    oss << "\"program\":{"
        << "\"name\":"      << jsonEscape(programName)
        << ",\"version\":"  << jsonEscape(programVersion)
        << ",\"port\":"     << programPort << "},"
        << "\"os\":{"
        << "\"os\":"        << jsonEscape(osName)
        << ",\"architecture\":" << jsonEscape(osArchitecture)
        << "},"
        << "\"user\":"      << jsonEscape(user);
    if (!computer.empty()) oss << ",\"computer\":" << jsonEscape(computer);
    if (!aid.empty())      oss << ",\"aid\":"      << jsonEscape(aid);
    if (!uid.empty())      oss << ",\"uid\":"      << jsonEscape(uid);
    oss << ",\"timestamp\":" << jsonEscape(timestamp);

    // build canonical string …
    std::string prefix = oss.str();
    std::ostringstream canon;
    std::string canonWT = worktype;
    if      (canonWT.rfind("PRP", 0) == 0) canonWT = "PRP";
    else if (canonWT.rfind("LL",  0) == 0) canonWT = "LL";


    canon << exponent << ";" << canonWT << ";";
    canon << "" << ";";  // factor list
    canon << "" << ";";  // known factors
    if (canonWT == "LL" || canonWT == "PRP") canon << toLower(res64);
    canon << ";";
    if (canonWT == "PRP") canon << toLower(res2048);
    canon << ";";
    canon << "0_3_1;";                               // shift-count
    canon << fftLength << ";";
    canon << "gerbicz:" << gerbiczError << ";";
    canon << programName << ";";
    canon << programVersion << ";";
    canon << "" << ";";  // program.subversion
    canon << "" << ";";  // program.details
    canon << osName << ";";
    canon << osArchitecture << ";";
    canon << timestamp;
    //std::cout << "\n\n CRC32 to hash ==> " << canon.str() << "\n";
    unsigned int crc = computeCRC32(canon.str());
    std::ostringstream hex;
    hex << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc;

    oss.str(""); oss.clear();
    oss << prefix
        << ",\"checksum\":{\"version\":1,\"checksum\":\"" << hex.str() << "\"}"
        << "}";
    return oss.str();
}

std::string JsonBuilder::generate(const CliOptions& opts,
                                  int transform_size,
                                  bool isPrime,
                                  const std::string& res64,
                                  const std::string& res2048) 
{
    // ---------------------------------------------
    // timestamp
    std::time_t now = std::time(nullptr);
    std::tm timeinfo;

    #ifdef _WIN32
        gmtime_s(&timeinfo, &now);
    #else
        std::tm* tmp = std::gmtime(&now);
        if (tmp != nullptr)
            timeinfo = *tmp;
    #endif

    char timestampBuf[32];
    std::strftime(timestampBuf, sizeof(timestampBuf), "%Y-%m-%d %H:%M:%S", &timeinfo);

    std::string status;

    status = isPrime ? "P" : "C";
    
    // assemble JSON
    return generatePrimeNetJson(
        // status: P or C
        status,
        opts.exponent,
        opts.mode == "prp" ? "PRP-3" : "LL",
        res64,
        res2048,
        1,  // residueType
        opts.gerbicz_error_count,  // gerbiczError
        transform_size,
        opts.proof ? 2 : 0,
        opts.proof ? opts.proofPower : 0,
        opts.proof ? 64 : 0,
        opts.proof ? fileMD5(opts.proofFile) : "",
        "prmers",                  // programName
        "0.1.0",                    // programVersion
        opts.portCode,             // portCode
        opts.osName,               // osName
        opts.osVersion,            // osVersion
        opts.osArch,               // osArchitecture
        opts.user.empty() ? "cherubrock" : opts.user,                 // user
        opts.aid,                  // aid
        opts.uid,                  // uid
        timestampBuf,              // timestamp
        opts.computer_name         // computer
    );

}


std::string JsonBuilder::computeRes64(
    const std::vector<uint64_t>& x,
    const CliOptions& opts,
    const std::vector<int>& digit_width,
    double /*elapsed*/,
    int /*transform_size*/)
{
    auto words = JsonBuilder::compactBits(x, digit_width, opts.exponent);
    if (opts.mode == "prp") doDiv9(opts.exponent, words);
    uint64_t finalRes64 = (uint64_t(words[1]) << 32) | words[0];
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setw(16) << std::setfill('0')
        << finalRes64;
    return oss.str();
}

std::string JsonBuilder::computeRes64Iter(
    const std::vector<uint64_t>& x,
    const CliOptions& opts,
    const std::vector<int>& digit_width,
    double /*elapsed*/,
    int /*transform_size*/)
{
    auto words = JsonBuilder::compactBits(x, digit_width, opts.exponent);
    //if (opts.mode == "prp") doDiv9(opts.exponent, words);
    uint64_t finalRes64 = (uint64_t(words[1]) << 32) | words[0];
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setw(16) << std::setfill('0')
        << finalRes64;
    return oss.str();
}


std::string JsonBuilder::computeRes2048(
    const std::vector<uint64_t>& x,
    const CliOptions& opts,
    const std::vector<int>& digit_width,
    double /*elapsed*/,
    int /*transform_size*/)
{
    auto words = JsonBuilder::compactBits(x, digit_width, opts.exponent);
    if (opts.mode == "prp") doDiv9(opts.exponent, words);
    std::ostringstream oss;
    for (int i = 63; i >= 0; --i) {
        oss << std::hex << std::nouppercase << std::setw(8) << std::setfill('0')
            << words[i];
    }
    return oss.str();
}

void JsonBuilder::write(const std::string& json,
                        const std::string& path)
{
    std::ofstream out(path);
    out << json;
}

} // namespace io