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
    const std::string &uid,
    const std::string &timestamp,
    const std::string &computer,
    const std::vector<std::string>& knownFactors)
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
    if (!knownFactors.empty()) {
        oss << ",\"known-factors\":[";
        for (size_t i = 0; i < knownFactors.size(); ++i) {
            oss << jsonEscape(knownFactors[i]);
            if (i + 1 < knownFactors.size()) oss << ",";
        }
        oss << "]";
    }
    std::string prefix = oss.str();
    std::ostringstream canon;
    std::string canonWT = worktype;
    if      (canonWT.rfind("PRP", 0) == 0) canonWT = "PRP";
    else if (canonWT.rfind("LL",  0) == 0) canonWT = "LL";
    
    canon << exponent << ";" << canonWT << ";";
    canon << "" << ";";
    canon << "" << ";";
    if (canonWT == "LL" || canonWT == "PRP") canon << toLower(res64);
    canon << ";";
    std::string knownFactorStr;
    if (!knownFactors.empty()) {
        knownFactorStr = knownFactors[0];
        for (size_t i = 1; i < knownFactors.size(); ++i)
            knownFactorStr += "," + knownFactors[i];
        canon << knownFactorStr << ";";
    }
    if (canonWT == "PRP") canon << toLower(res2048);
    canon << ";";
    canon << "0_3_1;";
    canon << fftLength << ";";
    canon << "gerbicz:" << gerbiczError << ";";
    canon << programName << ";";
    canon << programVersion << ";";
    canon << "" << ";";
    canon << "" << ";";
    canon << osName << ";";
    canon << osArchitecture << ";";
    canon << timestamp;
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

    if (opts.mode == "pm1") {
        std::ostringstream oss;
        bool hasFactor = !opts.knownFactors.empty();
        oss << "{"
            << "\"status\":\"" << (hasFactor ? "F" : "NF") << "\","
            << "\"exponent\":" << opts.exponent << ","
            << "\"worktype\":\"PM1\",";
        if (hasFactor) {
            oss << "\"factors\":[";
            for (size_t i = 0; i < opts.knownFactors.size(); ++i) {
                if (i) oss << ",";
                oss << jsonEscape(opts.knownFactors[i]);
            }
            oss << "],";
        }
        oss << "\"b1\":" << opts.B1 << ",";
        if (opts.B2 > 0) oss << "\"b2\":" << opts.B2 << ",";
        oss << "\"fft-length\":" << transform_size << ","
            << "\"program\":{"
                << "\"name\":\"prmers\","
                << "\"version\":" << jsonEscape(core::PRMERS_VERSION) << ","
                << "\"port\":" << opts.portCode
            << "},"
            << "\"timestamp\":" << jsonEscape(timestampBuf) << ","
            << "\"user\":" << jsonEscape(opts.user.empty() ? "prmers" : opts.user);
        if (!opts.computer_name.empty())
            oss << ",\"computer\":" << jsonEscape(opts.computer_name);
        if (!opts.aid.empty())
            oss << ",\"aid\":" << jsonEscape(opts.aid);
        std::string prefix = oss.str();
        std::ostringstream canon;
        canon << opts.exponent << ";PM1;";
        canon << "" << ";";
        canon << "" << ";";
        canon << opts.B1 << ";";
        if (opts.B2 > 0) canon << opts.B2;
        canon << ";";
        canon << "" << ";";
        canon << transform_size << ";";
        canon << "" << ";";
        canon << "prmers" << ";";
        canon << core::PRMERS_VERSION << ";";
        canon << "" << ";";
        canon << "" << ";";
        canon << "" << ";";
        canon << "" << ";";
        canon << timestampBuf;
        unsigned int crc = computeCRC32(canon.str());
        std::ostringstream hex;
        hex << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc;
        oss.str(""); oss.clear();
        oss << prefix
            << ",\"checksum\":{\"version\":1,\"checksum\":\"" << hex.str() << "\"}"
            << "}";
        return oss.str();
    }

    if (opts.mode == "ecm") {
        std::ostringstream oss;
        bool hasFactor = !opts.knownFactors.empty();
        oss << "{"
            << "\"status\":\"" << (hasFactor ? "F" : "NF") << "\","
            << "\"exponent\":" << opts.exponent << ","
            << "\"worktype\":\"ECM\",";
        if (hasFactor) {
            oss << "\"factors\":[";
            for (size_t i = 0; i < opts.knownFactors.size(); ++i) {
                if (i) oss << ",";
                oss << jsonEscape(opts.knownFactors[i]);
            }
            oss << "],";
        }
        oss << "\"b1\":" << opts.B1 << ",";
        if (opts.B2 > 0) oss << "\"b2\":" << opts.B2 << ",";
        //uint64_t curves = opts.nmax ? opts.nmax : (opts.K ? opts.K : 0ULL);
        oss << "\"curves\":" << opts.curves_tested_for_found << ",";
        oss << "\"fft-length\":" << transform_size << ",";
        oss << "\"program\":{"
            << "\"name\":\"prmers\","
            << "\"version\":" << jsonEscape(core::PRMERS_VERSION) << ","
            << "\"port\":" << opts.portCode
            << "},"
            << "\"timestamp\":" << jsonEscape(timestampBuf) << ","
            << "\"user\":" << jsonEscape(opts.user.empty() ? "prmers" : opts.user);
        if (!opts.computer_name.empty()) oss << ",\"computer\":" << jsonEscape(opts.computer_name);
        if (!opts.aid.empty()) oss << ",\"aid\":" << jsonEscape(opts.aid);
        bool isEdw = opts.edwards;
        int torsion = opts.notorsion ? 0 : (opts.torsion16 ? 16 : 8);
        oss << ",\"curve-type\":" << jsonEscape(isEdw ? "Edwards" : "Montgomery")
            << ",\"torsion-subgroup\":" << torsion;
        /*if (isEdw) {
            if (opts.sigma) oss << ",\"Edwards\":{\"sigma\":" << opts.sigma << "}";
            else oss << ",\"Edwards\":{}";
        } else {
            if (opts.sigma) oss << ",\"sigma\":" << opts.sigma;
        }*/
        oss << ",\"sigma_hex\":" << opts.sigma_hex;
        oss << ",\"curve_seed\":" << opts.curve_seed;
        oss << ",\"base_seed\":" << opts.curve_seed;
        std::string prefix = oss.str();
        std::ostringstream canon;
        canon << opts.exponent << ";ECM;";
        canon << "" << ";";
        canon << "" << ";";
        canon << opts.B1 << ";";
        if (opts.B2 > 0) canon << opts.B2;
        canon << ";";
        if (isEdw) canon << "E";
        if (opts.sigma) canon << opts.sigma;
        canon << ";";
        canon << transform_size << ";";
        canon << "" << ";";
        canon << "prmers" << ";";
        canon << core::PRMERS_VERSION << ";";
        canon << "" << ";";
        canon << "" << ";";
        canon << opts.osName << ";";
        canon << opts.osArch << ";";
        canon << timestampBuf;
        unsigned int crc = computeCRC32(canon.str());
        std::ostringstream hex;
        hex << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc;
        oss.str(""); oss.clear();
        oss << prefix
            << ",\"checksum\":{\"version\":1,\"checksum\":\"" << hex.str() << "\"}"
            << "}";
        return oss.str();
    }

    std::string status = isPrime ? "P" : "C";
    int residueType = opts.knownFactors.empty() ? 1 : 5;
    return generatePrimeNetJson(
        status,
        opts.exponent,
        opts.mode == "prp" ? "PRP-3" : "LL",
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
        opts.user.empty() ? "prmers" : opts.user,
        opts.aid,
        opts.uid,
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

} // namespace io
