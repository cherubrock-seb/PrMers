// include/core/AlgoUtils.hpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#define NOMINMAX
#include "core/App.hpp"
#include "core/QuickChecker.hpp"
#include "core/Printer.hpp"
#include "core/ProofSet.hpp"
#include "core/ProofSetMarin.hpp"
#include "math/Carry.hpp"
#include "util/GmpUtils.hpp"
#include "io/WorktodoParser.hpp"
#include "io/WorktodoManager.hpp"
#include "io/CurlClient.hpp"
#include "marin/engine.h"
#include "marin/file.h"
#include "ui/WebGuiServer.hpp"
#include "core/Version.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <map>
#include <future>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#ifdef _WIN32
# include <windows.h>
#endif
#include <csignal>
#include <chrono>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <tuple>
#include <atomic>
#include <fstream>
#include <memory>
#include <optional>
#include <cmath>
#include <thread>
#include <gmp.h>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;


namespace core::algo {

inline static std::atomic<bool> interrupted{false};
inline static void handle_sigint(int) { interrupted = true; }

inline static std::vector<std::string> parseConfigFile(const std::string& config_path) {
    std::ifstream config(config_path);
    std::vector<std::string> args;
    std::string line;

    if (!config.is_open()) {
        std::cerr << "Warning: no config file: " << config_path << std::endl;
        if (auto g = ui::WebGuiServer::instance()) {
            std::ostringstream oss;
            oss << "Warning: no config file: " << config_path;
            g->appendLog(oss.str());
        }
        return args;
    }

    std::cout << "Loading options from config file: " << config_path << std::endl;
    if (auto g = ui::WebGuiServer::instance()) {
        std::ostringstream oss;
        oss << "Loading options from config file: " << config_path;
        g->appendLog(oss.str());
    }

    while (std::getline(config, line)) {
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) args.push_back(token);
    }

    if (!args.empty()) {
        std::cout << "Options from config file:" << std::endl;
        if (auto g = ui::WebGuiServer::instance()) g->appendLog("Options from config file:");
        for (const auto& arg : args) std::cout << "  " << arg << std::endl;
        if (auto g = ui::WebGuiServer::instance()) {
            std::ostringstream oss;
            for (const auto& arg : args) oss << "  " << arg << std::endl;
            g->appendLog(oss.str());
        }
    } else {
        std::cout << "No options found in config file." << std::endl;
        if (auto g = ui::WebGuiServer::instance()) g->appendLog("No options found in config file.");
    }

    return args;
}

inline void writeStageResult(const std::string& file, const std::string& message) {
    std::ofstream out(file, std::ios::app);
    if (!out) {
        std::cerr << "Cannot open " << file << " for writing\n";
        if (auto g = ui::WebGuiServer::instance()) {
            std::ostringstream oss;
            oss << "Cannot open " << file << " for writing";
            g->appendLog(oss.str());
        }
        return;
    }
    out << message << '\n';
}


inline void restart_self(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() > 1 && args[1].find_first_not_of("0123456789") == std::string::npos) {
        args.erase(args.begin() + 1);
    }

#ifdef _WIN32
    std::string command = "\"" + args[0] + "\"";
    for (size_t i = 1; i < args.size(); ++i) command += " \"" + args[i] + "\"";
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    if (CreateProcessA(NULL, const_cast<char*>(command.c_str()), NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        exit(0);
    } else {
        std::cerr << "Failed to restart program (CreateProcess failed)" << std::endl;
        if (auto g = ui::WebGuiServer::instance()) g->appendLog("Failed to restart program (CreateProcess failed)");
    }
#else
    std::cout << "\nRestarting program without exponent:\n";
    if (auto g = ui::WebGuiServer::instance()) g->appendLog("\nRestarting program without exponent:\n");
    for (const auto& arg : args) std::cout << "   " << arg << std::endl;
    if (auto g = ui::WebGuiServer::instance()) {
        std::ostringstream oss;
        for (const auto& arg : args) oss << "  " << arg << std::endl;
        g->appendLog(oss.str());
    }
    std::vector<char*> exec_args;
    for (auto& s : args) exec_args.push_back(const_cast<char*>(s.c_str()));
    exec_args.push_back(nullptr);
    execv(exec_args[0], exec_args.data());
    std::cerr << "Failed to restart program (execv failed)" << std::endl;
    if (auto g = ui::WebGuiServer::instance()) g->appendLog("Failed to restart program (execv failed)");
#endif
}


static inline std::vector<uint32_t> pack_words_from_eng_digits(const engine::digit& d, uint32_t E) {
    const size_t totalWords = (E + 31) / 32;
    std::vector<uint32_t> out(totalWords, 0u);

    uint64_t acc = 0;
    int acc_bits = 0;
    size_t o = 0;

    const size_t n = d.get_size();
    for (size_t i = 0; i < n; ++i) {
        uint32_t w = uint32_t(d.width(i));
        uint32_t v = d.val(i);
        if (w < 32) v &= uint32_t((uint64_t(1) << w) - 1);
        acc |= (uint64_t)v << acc_bits;
        acc_bits += int(w);
        while (acc_bits >= 32 && o < totalWords) {
            out[o++] = uint32_t(acc & 0xFFFFFFFFu);
            acc >>= 32;
            acc_bits -= 32;
        }
    }
    if (o < totalWords) out[o++] = uint32_t(acc & 0xFFFFFFFFu);
    return out;
}

static inline std::vector<uint64_t> helperu(const engine::digit& d) {
    std::vector<uint64_t> out;
    out.reserve(d.get_size());
    for (size_t i = 0; i < d.get_size(); ++i) {
        uint64_t x = (uint64_t(d.width(i)) << 32) | uint64_t(d.val(i));
        out.push_back(x);
    }
    return out;
}


static inline uint32_t mod3_words(const std::vector<uint32_t>& W) {
    uint32_t r = 0; for (uint32_t w : W) r = (r + (w % 3)) % 3; return r;
}
static inline void div3_words(uint32_t E, std::vector<uint32_t>& W) {
    uint32_t r = (3 - mod3_words(W)) % 3;
    int topBits = int(E % 32);
    { uint64_t t = (uint64_t(r) << topBits) + W.back(); W.back() = uint32_t(t / 3); r = uint32_t(t % 3); }
    for (auto it = W.rbegin() + 1; it != W.rend(); ++it) { uint64_t t = (uint64_t(r) << 32) + *it; *it = uint32_t(t / 3); r = uint32_t(t % 3); }
}
static inline void prp3_div9(uint32_t E, std::vector<uint32_t>& W) { div3_words(E, W); div3_words(E, W); }

static inline std::string format_res64_hex(const std::vector<uint32_t>& W) {
    uint64_t r64 = (uint64_t(W.size() > 1 ? W[1] : 0) << 32) | (W.empty() ? 0u : W[0]);
    std::ostringstream oss; oss << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << r64; return oss.str();
}
static inline std::string format_res2048_hex(const std::vector<uint32_t>& W) {
    std::ostringstream oss; oss << std::hex << std::nouppercase << std::setfill('0');
    for (int i = 63; i >= 0; --i) {
        uint32_t w = (static_cast<std::size_t>(i) < W.size()) ? W[static_cast<std::size_t>(i)] : 0u;
        oss << std::setw(8) << w;
    }
    return oss.str();
}

static inline void delete_checkpoints(uint32_t p, bool wagstaff,bool pm1, bool llsafe, const std::string& dir = ".")
{
    std::string prefix;

    if (wagstaff) prefix += "wagstaff_";
    if (pm1)      prefix += "pm1_";
    if (llsafe)   prefix += "llsafe_";
    
    fs::path base = fs::path(dir) / (prefix + "m_" + std::to_string(p) + ".ckpt");
    std::error_code ec;
    fs::remove(base, ec);
    fs::remove(base.string() + ".old", ec);
    fs::remove(base.string() + ".new", ec);
}



inline void to_uppercase(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::toupper(c); });
}


inline mpz_class buildE(uint64_t B1) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now(), last = t0;

    std::vector<uint8_t> sieve((B1 >> 1) + 1, 1);
    std::vector<uint64_t> primes;
    primes.reserve(B1 ? B1 / std::log(double(B1)) : 1);
    for (uint64_t i = 3; i <= B1; i += 2) {
        if (sieve[i >> 1]) {
            primes.push_back(i);
            if (uint64_t ii = uint64_t(i) * i; ii <= B1)
                for (uint64_t j = ii; j <= B1; j += (i << 1))
                    sieve[j >> 1] = 0;
        }
    }

    size_t total = primes.size();
    std::atomic<size_t> next{0}, done{0};
    unsigned th = std::thread::hardware_concurrency();
    if (!th) th = 4;
    std::vector<mpz_class> part(th, 1);
    std::vector<std::thread> workers;

    for (unsigned t = 0; t < th; ++t)
        workers.emplace_back([&, t] {
            while (true) {
                size_t idx = next.fetch_add(1);
                if (idx >= total) break;
                uint64_t p = primes[idx];
                mpz_class pw;
                mpz_set_ui(pw.get_mpz_t(), static_cast<unsigned long>(p));
                unsigned long lim1 = static_cast<unsigned long>(B1 / p);
                mpz_class limit(lim1);
                //while (pw <= limit) pw *= mpz_class(p);
                while (pw <= limit) pw *= mpz_class(static_cast<unsigned long>(p));
                part[t] *= pw;
                done.fetch_add(1, std::memory_order_relaxed);
                if (interrupted) return;
            }
        });

    mpz_class E = 1;
    mpz_class pw2 = 2;
    unsigned long lim2 = static_cast<unsigned long>(B1 / 2);
    mpz_class limit2; mpz_set_ui(limit2.get_mpz_t(), lim2);

    while (pw2 <= limit2) pw2 *= 2;
    E *= pw2;

    std::cout << "Building E:   0%  ETA  --:--:--" << std::flush;
    while (done.load() < total && !interrupted) {
        auto now = clock::now();
        if (now - last >= std::chrono::milliseconds(500)) {
            double prog = double(done.load()) / total;
            double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
            long sec = long(eta + 0.5);
            int h = int(sec / 3600), m = int((sec % 3600) / 60), s = int(sec % 60);
            std::cout << "\rBuilding E: " << std::setw(3) << int(prog * 100)
                      << "%  ETA "
                      << std::setw(2) << std::setfill('0') << h << ':'
                      << std::setw(2) << m << ':'
                      << std::setw(2) << s << std::setfill(' ')
                      << std::flush;
            last = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    for (auto &w : workers) w.join();
    for (auto &p : part) E *= p;
    if (interrupted) {
        std::cout << "\n\nInterrupted signal received — using partial E computed so far.\n\n";
        for (auto &w : workers)
            if (w.joinable()) w.join();

        for (auto &p : part) E *= p;
        mp_bitcnt_t bits = mpz_sizeinbase(E.get_mpz_t(), 2);
        std::cout << "\nlog2(E) ≈ " << bits << " bits" << std::endl;
        interrupted = false; 
        return E;
    }

    std::cout << "\rBuilding E: 100%  ETA  00:00:00\n";
    return E;
}




static inline unsigned long evenGapBound(const mpz_class& B2) {
    double ln = std::log(mpz_get_d(B2.get_mpz_t()));
    double bound = std::ceil((ln * ln) / 2.0);
    return bound < 2 ? 1 : static_cast<unsigned long>(bound);
}

static inline size_t primeCountApprox(const mpz_class& low, const mpz_class& high) {
    auto li = [](double x) {
        double l = std::log(x);
        return x / l + x / (l * l);
    };
    double a = mpz_get_d(low.get_mpz_t());
    double b = mpz_get_d(high.get_mpz_t());
    double diff = li(b) - li(a);
    return diff > 0.0 ? static_cast<size_t>(diff) : 0;
}

static inline constexpr uint64_t CHKSUMMOD = 4294967291ULL;

static inline uint32_t ecm_checksum_pminus1(uint64_t B1, uint32_t p, const mpz_class& X_raw) {
    mpz_class N = (mpz_class(1) << p) - 1;

    uint64_t n = mpz_fdiv_ui(N.get_mpz_t(), CHKSUMMOD);
    uint64_t x = mpz_fdiv_ui(X_raw.get_mpz_t(), CHKSUMMOD);
    uint64_t b = B1 % CHKSUMMOD;

    uint64_t acc = (b * n) % CHKSUMMOD;
    acc = (acc * x) % CHKSUMMOD;
    return static_cast<uint32_t>(acc);
}

static inline std::string mpz_to_lower_hex(const mpz_class& z){
    char* s = mpz_get_str(nullptr, 16, z.get_mpz_t());
    std::string hex = s ? s : "";
    std::free(s);
    size_t i = 0; while (i + 1 < hex.size() && hex[i] == '0') ++i;
    return hex.substr(i);
}

inline void writeEcmResumeLine(const std::string& path,
                        uint64_t B1, uint32_t p,
                        const mpz_class& X)
{
    const uint32_t chk = ecm_checksum_pminus1(B1, p, X);

    std::ofstream out(path);
    if (!out) {
        std::cerr << "Error: could not write GMP-ECM resume file to " << path << std::endl;
        return;
    }
    out << "METHOD=P-1; "
        << "B1=" << B1 << "; "
        << "N=2^" << p << "-1; "
        << "X=0x" << mpz_to_lower_hex(X) << "; "
        << "CHECKSUM=" << chk << "; "
        << "PROGRAM=PrMers; X0=0x3; Y=0x0; Y0=0x0; WHO=; TIME=;"
        << '\n';

    std::cout << "GMP-ECM resume file written to: " << path << std::endl;
}


static inline bool read_mers_file(const std::string& path, std::vector<uint64_t>& v) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Error: cannot open " << path << " for reading\n";
        return false;
    }

    //in.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(v.data()),
        static_cast<std::streamsize>(v.size() * sizeof(uint64_t)));

    std::streamsize readBytes = in.gcount();
    if (readBytes < static_cast<std::streamsize>(sizeof(uint64_t))) {
        std::cerr << "Error: file too small: " << path << std::endl;
        return false;
    }

    if (readBytes != static_cast<std::streamsize>(v.size() * sizeof(uint64_t))) {
        std::cerr << "Warning: partial read from " << path
                  << " (" << readBytes << " / " << v.size() * sizeof(uint64_t) << " bytes)\n";
    }

    return true;
}


static inline std::vector<uint8_t> hex_to_le_bytes_pad4(const mpz_class& X) {
    char* s = mpz_get_str(nullptr, 16, X.get_mpz_t());
    std::string hex = s ? s : "";
    std::free(s);
    if (hex.empty()) hex = "0";
    if (hex.size() & 1) hex.insert(hex.begin(), '0');
    size_t pad = (8 - (hex.size() & 7)) & 7;
    if (pad) hex.insert(0, pad, '0');
    std::vector<uint8_t> be; be.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        uint8_t b = (uint8_t)std::stoi(hex.substr(i, 2), nullptr, 16);
        be.push_back(b);
    }
    std::vector<uint8_t> le(be.rbegin(), be.rend());
    return le;
}

static inline void write_u32(std::ofstream& o, uint32_t v){ o.write(reinterpret_cast<const char*>(&v),4); }
static inline void write_i32(std::ofstream& o, int32_t v){ o.write(reinterpret_cast<const char*>(&v),4); }
static inline void write_u64(std::ofstream& o, uint64_t v){ o.write(reinterpret_cast<const char*>(&v),8); }
static inline void write_u16(std::ofstream& o, uint16_t v){ o.write(reinterpret_cast<const char*>(&v),2); }
static inline void write_f64(std::ofstream& o, double v){ o.write(reinterpret_cast<const char*>(&v),8); }
static inline void write_u8 (std::ofstream& o, uint8_t  v){ o.write(reinterpret_cast<const char*>(&v),1); }

static inline bool read_text_file(const std::string& path, std::string& out){
    std::ifstream f(path);
    if(!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

static inline bool parse_ecm_resume_line(const std::string& t, uint64_t& B1, uint32_t& p, std::string& hexX){
    size_t iB1 = t.find("B1=");
    if(iB1==std::string::npos) return false;
    size_t eB1 = t.find(';', iB1);
    if(eB1==std::string::npos) return false;
    B1 = std::stoull(t.substr(iB1+3, eB1-(iB1+3)));

    size_t iN = t.find("N=2^");
    if(iN==std::string::npos) return false;
    size_t eN = t.find('-', iN);
    if(eN==std::string::npos) return false;
    p = (uint32_t)std::stoul(t.substr(iN+4, eN-(iN+4)));

    size_t iX = t.find("X=0x");
    if(iX==std::string::npos) return false;
    size_t eX = t.find(';', iX);
    if(eX==std::string::npos) return false;
    hexX = t.substr(iX+4, eX-(iX+4));
    return true;
}

static inline std::vector<uint8_t> hex_to_bytes_reversed_pad8(const std::string& hex){
    std::string h = hex;
    if(h.empty()) h = "0";
    size_t pad = (8 - (h.size() & 7)) & 7;
    if(pad) h.insert(0, pad, '0');
    std::vector<uint8_t> data; data.reserve(h.size()/2);
    size_t bytes = h.size()/2;
    for(size_t i=0;i<bytes;++i){
        size_t pos = h.size() - (i+1)*2;
        uint8_t b = (uint8_t)std::stoul(h.substr(pos,2), nullptr, 16);
        data.push_back(b);
    }
    return data;
}

static inline uint32_t checksum_prime95_s1(uint64_t B1, const std::vector<uint8_t>& data){
    uint64_t sum32 = 0;
    for(size_t i=0;i+3<data.size();i+=4){
        uint32_t w = (uint32_t)data[i] | ((uint32_t)data[i+1]<<8) | ((uint32_t)data[i+2]<<16) | ((uint32_t)data[i+3]<<24);
        sum32 += w;
    }
    uint64_t chk64 = ((B1<<1) + 6u + (data.size()>>1) + sum32) & 0xFFFFFFFFULL;
    return (uint32_t)chk64;
}

static inline bool write_prime95_s1_from_bytes(const std::string& outPath, uint32_t p, uint64_t B1, const std::vector<uint8_t>& data, const std::string& date_start, const std::string& date_end){
    std::ofstream out(outPath, std::ios::binary);
    if(!out) return false;
    uint32_t chk = checksum_prime95_s1(B1, data);
    write_u32(out, 830093643u);
    write_u32(out, 8u);
    write_f64(out, 1.0);
    write_i32(out, 2);
    write_u32(out, p);
    write_i32(out, -1);
    write_u8 (out, (uint8_t)'S');
    write_u8 (out, (uint8_t)'1');
    write_u16(out, 0);
    write_u64(out, 0);
    write_f64(out, 1.0);
    write_u32(out, chk);
    write_i32(out, 5);
    write_u64(out, B1);
    write_u64(out, B1);
    write_i32(out, 1);
    write_i32(out, (int32_t)(data.size()>>2));
    out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)data.size());

    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tmv{};
    #if defined(_WIN32)
    gmtime_s(&tmv, &tt);
    #else
    std::tm* tmp = std::gmtime(&tt);
    if (tmp) tmv = *tmp;
    #endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
    std::ostringstream ts;
    ts << buf << '.' << std::setw(3) << std::setfill('0') << (int)ms.count();
    std::string ds = date_start.empty() ? ts.str() : date_start;
    std::string de = date_end.empty() ? ts.str() : date_end;

    std::string os_str;
    #if defined(_WIN32)
    os_str = "Windows";
    #elif defined(__APPLE__)
    os_str = "macOS";
    #elif defined(__linux__)
    os_str = "Linux";
    #else
    os_str = "Unknown";
    #endif

    std::string arch_str;
    #if defined(__x86_64__) || defined(_M_X64)
    arch_str = "x86_64";
    #elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
    arch_str = "x86_32";
    #elif defined(__aarch64__) || defined(_M_ARM64)
    arch_str = "ARM64";
    #elif defined(__arm__) || defined(_M_ARM)
    arch_str = "ARM";
    #else
    arch_str = "unknown";
    #endif

    std::string json = std::string("{\"programs\":[{\"work\":{\"type\":\"PM1\",\"stage\":\"1\"},\"program\":{\"name\":\"prmers\",\"version\":\"" + core::PRMERS_VERSION + "\"},\"os\":{\"os\":\"") + os_str + "\",\"architecture\":\"" + arch_str + "\"},\"date_start\":\"" + ds + "\",\"date_end\":\"" + de + "\"}]}";

    auto crc32 = [](const uint8_t* d, size_t n){
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = 0; i < n; ++i) {
            crc ^= d[i];
            for (int k = 0; k < 8; ++k) {
                uint32_t mask = -(crc & 1u);
                crc = (crc >> 1) ^ (0xEDB88320u & mask);
            }
        }
        return ~crc;
    };

    const char magic[16] = {'M','O','R','E','I','N','F','O','J','S','O','N','D','A','T','A'};
    uint32_t version = 1u;
    std::vector<uint8_t> json_bytes(json.begin(), json.end());
    uint32_t crc = crc32(json_bytes.data(), json_bytes.size());
    uint32_t chunk_size = 8u + (uint32_t)json_bytes.size();

    out.write(magic, 16);
    write_u32(out, chunk_size);
    write_u32(out, version);
    write_u32(out, crc);
    if (!json_bytes.empty())
        out.write(reinterpret_cast<const char*>(json_bytes.data()), (std::streamsize)json_bytes.size());

    return (bool)out;
}


static inline mpz_class compute_X_with_dots(engine* eng, engine::Reg reg, const mpz_class& Mp) {
    using namespace std::chrono;

    std::atomic<bool> done{false};
    std::thread ticker([&]{
        const char* msg = "Constructing and reducing large integer ";
        std::cout << msg << std::flush;
        size_t dots = 0, wrap = 60;
        while (!done.load(std::memory_order_relaxed)) {
            std::cout << '.' << std::flush;
            if (++dots % wrap == 0) std::cout << '\n' << msg << std::flush;
            std::this_thread::sleep_for(milliseconds(300));
        }
        std::cout << " done.\n";
    });

    mpz_class X(0);
    mpz_t tmp; mpz_init(tmp);
    eng->get_mpz(tmp, reg);
    mpz_set(X.get_mpz_t(), tmp);
    mpz_clear(tmp);

    X %= Mp;

    done.store(true, std::memory_order_relaxed);
    ticker.join();
    return X;
}



static inline mpz_class product_tree_range_u64(const std::vector<uint64_t>& v, size_t lo, size_t hi, size_t leaf, int par) {
    size_t n = hi - lo;
    if (n == 0) return mpz_class(1);
    if (n <= leaf) {
        mpz_class r = 1;
        for (size_t i = lo; i < hi; ++i) {
            mpz_class t; mpz_set_ui(t.get_mpz_t(), (unsigned long)v[i]);
            r *= t;
        }
        return r;
    }
    size_t mid = lo + (n >> 1);
    if (par > 1) {
        auto fut = std::async(std::launch::async, [&]{ return product_tree_range_u64(v, lo, mid, leaf, par >> 1); });
        mpz_class right = product_tree_range_u64(v, mid, hi, leaf, par - (par >> 1));
        mpz_class left = fut.get();
        return left * right;
    } else {
        mpz_class left  = product_tree_range_u64(v, lo, mid, leaf, 1);
        mpz_class right = product_tree_range_u64(v, mid, hi, leaf, 1);
        return left * right;
    }
}

static inline size_t product_prefix_fit_u64(const std::vector<uint64_t>& v, size_t lo, size_t hi, const mpz_class& Ecur, uint64_t maxBits, mpz_class& outProd, size_t leaf, int par) {
    mpz_class P = product_tree_range_u64(v, lo, hi, leaf, par);
    mpz_class Etmp = Ecur * P;
    if (mpz_sizeinbase(Etmp.get_mpz_t(), 2) <= maxBits) { outProd = P; return hi; }
    if (hi - lo == 1) { outProd = 1; return lo; }
    size_t mid = lo + ((hi - lo) >> 1);
    mpz_class Pleft;
    size_t k = product_prefix_fit_u64(v, lo, mid, Ecur, maxBits, Pleft, leaf, par > 1 ? (par >> 1) : 1);
    if (k == mid) {
        mpz_class E2 = Ecur * Pleft;
        mpz_class Pright;
        size_t k2 = product_prefix_fit_u64(v, mid, hi, E2, maxBits, Pright, leaf, par > 1 ? (par - (par >> 1)) : 1);
        outProd = Pleft * Pright;
        return k2;
    } else {
        outProd = Pleft;
        return k;
    }
}

inline mpz_class buildE2(uint64_t B1, uint64_t startPrime, uint64_t maxBits, uint64_t& nextStart, bool includeTwo) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now(), last = t0;
    nextStart = 0;
    if (B1 < 3) return includeTwo ? mpz_class(2) : mpz_class(1);

    uint64_t s = startPrime < 3 ? 3 : (startPrime | 1ULL);
    uint64_t R = (uint64_t)std::sqrt((long double)B1);
    if ((R & 1ULL) == 0) R -= 1;

    std::vector<uint8_t> base((R >> 1) + 1, 1);
    for (uint64_t i = 3; i * i <= R; i += 2)
        if (base[i >> 1])
            for (uint64_t j = i * i; j <= R; j += (i << 1))
                base[j >> 1] = 0;

    std::vector<uint64_t> small_primes;
    for (uint64_t i = 3; i <= R; i += 2)
        if (base[i >> 1]) small_primes.push_back(i);

    mpz_class E = 1;
    std::vector<uint64_t> batch;
    std::vector<uint64_t> batch_primes;
    batch.reserve(1u << 16);
    batch_primes.reserve(1u << 16);

    if (includeTwo) {
        uint64_t pw2 = 2;
        while (pw2 <= B1 / 2) pw2 <<= 1;
        batch.push_back(pw2);
        batch_primes.push_back(2);
    }

    const uint64_t span = 1ULL << 24;
    if ((s & 1ULL) == 0) s += 1;
    uint64_t totalOdd = (B1 >= s) ? (((B1 - s) >> 1) + 1) : 0;
    std::cout << "Building E-chunk:   0%  ETA  --:--:--" << std::flush;

    auto flush_batch = [&](bool final_segment)->bool{
        if (batch.empty()) return true;
        size_t leaf = 16;
        int par = (int)std::thread::hardware_concurrency(); if (par <= 0) par = 2;
        mpz_class Pfit;
        size_t used = product_prefix_fit_u64(batch, 0, batch.size(), E, maxBits, Pfit, leaf, par);
        E *= Pfit;
        if (used < batch.size() && mpz_cmp_ui(E.get_mpz_t(), 1) != 0) { nextStart = batch_primes[used]; return false; }
        batch.clear();
        batch_primes.clear();
        if (final_segment && nextStart == 0) std::cout << "\rBuilding E-chunk: 100%  ETA  00:00:00\n";
        return true;
    };

    uint64_t low = s;
    while (low <= B1 && !interrupted) {
        uint64_t high = low + span - 1;
        if (high > B1) high = B1;
        if ((high & 1ULL) == 0) high -= 1;
        if (high < low) break;

        size_t len = size_t(((high - low) >> 1) + 1);
        std::vector<uint8_t> seg(len, 1);

        for (uint64_t q : small_primes) {
            uint64_t q2 = q * q;
            uint64_t start = (q2 > low) ? q2 : ((low + q - 1) / q) * q;
            if ((start & 1ULL) == 0) start += q;
            if (start < low) start += q;
            for (uint64_t j = start; j <= high; j += (q << 1)) {
                size_t idx = size_t((j - low) >> 1);
                seg[idx] = 0;
            }
        }

        for (uint64_t n = low; n <= high; n += 2) {
            if (!seg[size_t((n - low) >> 1)]) {
                auto now = clock::now();
                if (now - last >= std::chrono::milliseconds(500)) {
                    uint64_t progressedOdd = ((n >= s) ? (((n - s) >> 1) + 1) : 0);
                    double prog = totalOdd ? (double)progressedOdd / (double)totalOdd : 1.0;
                    double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
                    long sec = long(eta + 0.5);
                    int h = int(sec / 3600), m = int((sec % 3600) / 60), ss = int(sec % 60);
                    std::cout << "\rBuilding E-chunk: " << std::setw(3) << int(prog * 100)
                              << "%  ETA "
                              << std::setw(2) << std::setfill('0') << h << ':'
                              << std::setw(2) << m << ':'
                              << std::setw(2) << ss << std::setfill(' ')
                              << std::flush;
                    last = now;
                }
                continue;
            }
            uint64_t p = n;
            uint64_t pw = p;
            while (pw <= B1 / p) pw *= p;
            batch.push_back(pw);
            batch_primes.push_back(p);

            if (batch.size() >= (1u << 16)) {
                if (!flush_batch(false)) goto done;
            }

            auto now = clock::now();
            if (now - last >= std::chrono::milliseconds(500)) {
                uint64_t progressedOdd = ((n >= s) ? (((n - s) >> 1) + 1) : 0);
                double prog = totalOdd ? (double)progressedOdd / (double)totalOdd : 1.0;
                double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
                long sec = long(eta + 0.5);
                int h = int(sec / 3600), m = int((sec % 3600) / 60), ss = int(sec % 60);
                std::cout << "\rBuilding E-chunk: " << std::setw(3) << int(prog * 100)
                          << "%  ETA "
                          << std::setw(2) << std::setfill('0') << h << ':'
                          << std::setw(2) << m << ':'
                          << std::setw(2) << ss << std::setfill(' ')
                          << std::flush;
                last = now;
            }
            if (interrupted) break;
        }

        if (!flush_batch(false)) goto done;

        low = high + 2;
        auto now = clock::now();
        uint64_t progressedOdd = ((low > s) ? (((std::min(low - 2, B1) - s) >> 1) + 1) : 0);
        double prog = totalOdd ? (double)progressedOdd / (double)totalOdd : 1.0;
        double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
        long sec = long(eta + 0.5);
        int h = int(sec / 3600), m = int((sec % 3600) / 60), ss = int(sec % 60);
        std::cout << "\rBuilding E-chunk: " << std::setw(3) << int(prog * 100)
                  << "%  ETA "
                  << std::setw(2) << std::setfill('0') << h << ':'
                  << std::setw(2) << m << ':'
                  << std::setw(2) << ss << std::setfill(' ')
                  << std::flush;
    }

done:
    if (!batch.empty() && nextStart == 0) flush_batch(true);
    if (interrupted) {
        std::cout << "\n\nInterrupted signal received — using partial E computed so far.\n\n";
        mp_bitcnt_t bits = mpz_sizeinbase(E.get_mpz_t(), 2);
        std::cout << "\nlog2(E) ≈ " << bits << " bits" << std::endl;
        interrupted = false;
        return E;
    }
    if (nextStart == 0) std::cout << "\rBuilding E-chunk: 100%  ETA  00:00:00\n";
    return E;
}

inline mpz_class gcd_with_dots(const mpz_class& A, const mpz_class& B) {
    std::atomic<bool> done{false};
    std::thread ticker([&]{
        using namespace std::chrono;
        const char* msg = "Computing GCD (this may take a while) ";
        std::cout << msg << std::flush;
        size_t dots = 0, wrap = 60;
        while (!done.load(std::memory_order_relaxed)) {
            std::cout << '.' << std::flush;
            if (++dots % wrap == 0) std::cout << '\n' << msg << std::flush;
            std::this_thread::sleep_for(milliseconds(300));
        }
        std::cout << " done.\n";
    });
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), A.get_mpz_t(), B.get_mpz_t());
    done.store(true, std::memory_order_relaxed);
    ticker.join();
    return g;
}

} // namespace core::algo
