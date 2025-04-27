#include "io/JsonBuilder.hpp"
#include "io/CliParser.hpp"          // for CliOptions
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

namespace io{
    //-----------------------------------------------------------------------------
    // mod3, doDiv3 et doDiv9 : pour PRP‑3 on divise deux fois par 3
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
//-----------------------------------------------------------------------------
// compactBits : passe du tableau x[] (mixed‑radix) à un vecteur de mots 32 bits
//-----------------------------------------------------------------------------
static std::vector<uint32_t> compactBits(
    const std::vector<uint64_t>& x,
    const std::vector<int>& digit_width,
    uint32_t E
) {
    std::vector<uint32_t> out;
    out.reserve((E - 1) / 32 + 1);

    int carry = 0;
    uint32_t outWord = 0;
    int haveBits = 0;

    for (size_t p = 0; p < x.size(); ++p) {
        int w = digit_width[p];
        // on combine le carry avec votre digit x[p]
        uint64_t v64 = uint64_t(carry) + x[p];
        carry = int(v64 >> w);
        uint32_t v = uint32_t(v64 & ((1ULL << w) - 1));

        // on packe dans outWord
        int topBits = 32 - haveBits;
        outWord |= v << haveBits;
        if (w >= topBits) {
            out.push_back(outWord);
            outWord = (w > topBits) ? (v >> topBits) : 0;
            haveBits = w - topBits;
        } else {
            haveBits += w;
        }
    }

    // dernier mot s’il reste des bits ou un carry
    if (haveBits > 0 || carry) {
        out.push_back(outWord);
        for (size_t i = 1; carry && i < out.size(); ++i) {
            uint64_t sum = uint64_t(out[i]) + carry;
            out[i]   = uint32_t(sum & 0xFFFFFFFF);
            carry    = int(sum >> 32);
        }
    }

    return out;
}
static std::string fileMD5(const std::string& filePath) {
    namespace fs = std::filesystem;
    FILE *f = fopen(filePath.c_str(), "rb");
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
    if (isPRP) {
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

    unsigned int crc = computeCRC32(canon.str());
    std::ostringstream hex;
    hex << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc;

    oss.str(""); oss.clear();
    oss << prefix
        << ",\"checksum\":{\"version\":1,\"checksum\":\"" << hex.str() << "\"}"
        << "}";
    return oss.str();
}



std::string JsonBuilder::generate(std::vector<unsigned long> x,
                                  const CliOptions& opts,
                                  const std::vector<int>& digit_width,
                                  double /*elapsed*/) 
{
    
    
    auto words = compactBits(x, digit_width, opts.exponent);
    if (opts.mode == "prp") doDiv9(opts.exponent, words);

    uint64_t finalRes64 = (uint64_t(words[1]) << 32) | words[0];
    std::cout << "\finalRes64=" <<finalRes64 <<"\n";
    std::ostringstream oss64;
    oss64 << std::hex << std::uppercase << std::setw(16) << std::setfill('0')
        << finalRes64;
    std::string res64 = oss64.str();

    std::ostringstream oss2048;
    for (int i = 63; i >= 0; --i) {
        oss2048 << std::hex << std::nouppercase << std::setw(8) << std::setfill('0')
                << words[i];
    }
    std::string res2048 = oss2048.str();
    std::cout << "\nres2048=" << res2048 << "\n";
    std::cout << "\nres64=" << res64 << "\n";
    // ---------------------------------------------
    // 4) timestamp
    char timestampBuf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(timestampBuf, sizeof(timestampBuf),
                  "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    // 5) assemble JSON
    return generatePrimeNetJson(
        // status: P or C
        (opts.mode == "ll"
        ? (std::all_of(words.begin(), words.end(), [](uint32_t v){ return v == 0; }) ? std::string("P") : std::string("C"))
        : ((words[0] == 9 && std::all_of(words.begin() + 1, words.end(),
                                        [](uint32_t v){ return v == 0; })) ? std::string("P") : std::string("C"))),
        opts.exponent,
        opts.mode == "prp" ? "PRP-3" : "LL",
        res64,
        res2048,
        1,  // residueType
        0,  // gerbiczError
        unsigned(opts.exponent),
        opts.proof ? 1 : 0,
        opts.proof ? opts.proofPower : 0,
        opts.proof ? 64 : 0,
        opts.proof ? fileMD5(opts.proofFile) : "",
        "prmers",                  // programName
        "0.15",                    // programVersion
        opts.portCode,             // portCode
        opts.osName,               // osName
        opts.osVersion,            // osVersion
        opts.osArch,               // osArchitecture
        opts.user,                 // user
        opts.aid,                  // aid
        opts.uid,                  // uid
        timestampBuf,              // timestamp
        opts.computer_name         // computer
    );

}

void JsonBuilder::write(const std::string& json,
                        const std::string& path)
{
    std::ofstream out(path);
    out << json;
}

} // namespace io
