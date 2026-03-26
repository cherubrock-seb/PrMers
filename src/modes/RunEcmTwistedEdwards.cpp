#include "core/App.hpp"
#include "core/AlgoUtils.hpp"
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
# include <io.h>
#else
# include <unistd.h>
# include <sys/wait.h>
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
#include <unordered_set>
#include <regex>
#include <ctime>
#include <cstdlib>
#include <cctype>

using namespace core;
using namespace std::chrono;
using std::string;
using std::vector;
using std::ostringstream;
using std::ofstream;
using std::cout;
using std::endl;

using core::algo::format_res64_hex;
using core::algo::format_res2048_hex;
using core::algo::helperu;
using core::algo::mod3_words;
using core::algo::delete_checkpoints;
using core::algo::to_uppercase;
using core::algo::div3_words;
using core::algo::pack_words_from_eng_digits;
using core::algo::prp3_div9;
using core::algo::parseConfigFile;
using core::algo::interrupted;
using core::algo::handle_sigint;
using core::algo::writeStageResult;
using core::algo::restart_self;
using core::algo::buildE;
using core::algo::evenGapBound;
using core::algo::primeCountApprox;
using core::algo::read_mers_file;
using core::algo::writeEcmResumeLine;
using core::algo::mpz_to_lower_hex;
using core::algo::ecm_checksum_pminus1;
using core::algo::CHKSUMMOD;
using core::algo::write_prime95_s1_from_bytes;
using core::algo::checksum_prime95_s1;
using core::algo::hex_to_bytes_reversed_pad8;
using core::algo::parse_ecm_resume_line;
using core::algo::read_text_file;
using core::algo::write_u32;
using core::algo::write_i32;
using core::algo::write_u64;
using core::algo::write_u16;
using core::algo::write_f64;
using core::algo::write_u8;
using core::algo::hex_to_le_bytes_pad4;
using core::algo::buildE2;
using core::algo::product_prefix_fit_u64;
using core::algo::product_tree_range_u64;
using core::algo::compute_X_with_dots;
using core::algo::gcd_with_dots;
namespace fs = std::filesystem;

static bool ecm_progress_use_single_line() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return ::isatty(fileno(stdout)) != 0;
#endif
}

static void ecm_print_progress_line(const std::string& s) {
    static const bool single_line = ecm_progress_use_single_line();
    if (single_line) {
        std::cout << '\r' << s << std::flush;
    } else {
        std::cout << s << std::endl;
    }
}


static uint64_t ecm_isqrt_u64(uint64_t x) {
    long double r = std::sqrt((long double)x);
    uint64_t y = (uint64_t)r;
    while ((y + 1) > 0 && (y + 1) <= x / (y + 1)) ++y;
    while (y > 0 && y > x / y) --y;
    return y;
}

static inline uint32_t compute_s2_chunk_bits(size_t transform_words) {
    if (transform_words <= 256)    return 1048576u;
    if (transform_words <= 1024)   return 524288u;
    if (transform_words <= 4096)   return 262144u;
    if (transform_words <= 16384)  return 131072u;
    return 65536u;
}

static inline mpz_class ecm_mpz_from_u64(uint64_t v) {
    mpz_class z;
    mpz_import(z.get_mpz_t(), 1, 1, sizeof(v), 0, 0, &v);
    return z;
}
static inline void ecm_mpz_mul_u64(mpz_class& a, uint64_t x) {
    if (x <= (uint64_t)(~0UL)) {
        mpz_mul_ui(a.get_mpz_t(), a.get_mpz_t(), (unsigned long)x);
    } else {
        mpz_class t = ecm_mpz_from_u64(x);
        mpz_mul(a.get_mpz_t(), a.get_mpz_t(), t.get_mpz_t());
    }
}

static std::vector<uint32_t> ecm_sieve_base_primes(uint32_t limit) {
    std::vector<uint8_t> isPrime(limit + 1, 1);
    if (limit == 0) {
        isPrime[0] = 0;
    } else {
        isPrime[0] = 0;
        isPrime[1] = 0;
    }
    for (uint32_t p = 2; (uint64_t)p * p <= limit; ++p) {
        if (!isPrime[p]) continue;
        for (uint64_t m = (uint64_t)p * p; m <= limit; m += p) isPrime[(size_t)m] = 0;
    }
    std::vector<uint32_t> primes;
    primes.reserve(limit / 10 + 16);
    for (uint32_t i = 2; i <= limit; ++i) if (isPrime[i]) primes.push_back(i);
    return primes;
}

static void ecm_segmented_primes_odd(uint64_t low, uint64_t high,
                                     const std::vector<uint32_t>& basePrimes,
                                     std::vector<uint64_t>& out) {
    out.clear();
    if (high < 2 || low > high) return;
    if (low <= 2 && 2 <= high) out.push_back(2);

    if (low < 3) low = 3;
    if ((low & 1ULL) == 0) ++low;
    if ((high & 1ULL) == 0) --high;
    if (low > high) return;

    const uint64_t nOdds = ((high - low) >> 1) + 1;
    std::vector<uint8_t> isPrime((size_t)nOdds, 1);

    for (uint32_t p32 : basePrimes) {
        if (p32 == 2) continue;
        const uint64_t p = (uint64_t)p32;
        const uint64_t p2 = p * p;
        if (p2 > high) break;
        uint64_t start = ((low + p - 1) / p) * p;
        if (start < p2) start = p2;
        if ((start & 1ULL) == 0) start += p;
        for (uint64_t x = start; x <= high; x += (p << 1)) {
            isPrime[(size_t)((x - low) >> 1)] = 0;
        }
    }

    out.reserve((size_t)(nOdds / 10 + 16));
    for (uint64_t i = 0; i < nOdds; ++i) {
        if (isPrime[(size_t)i]) out.push_back(low + (i << 1));
    }
}




struct Prime95Stage2Task {
    uint64_t curve_idx = 0;
    std::string resume_src_path;
    std::string resume_filename;
    std::string worktodo_line;
    std::vector<std::string> known_factors;
    std::string log_filename;
    std::string sigma_hex;
    uint64_t curve_seed = 0;
    uint64_t base_seed = 0;
};

struct Prime95Stage2TaskResult {
    uint64_t curve_idx = 0;
    bool success = false;
    bool factor_found = false;
    bool known_factor = false;
    std::string factor;
    std::string json_line;
    int exit_code = -1;
    std::string error;
};

static std::string p95_normalize_factor_string(std::string s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch){ return std::isspace(ch) != 0; }), s.end());
    if (!s.empty() && s.front() == '+') s.erase(s.begin());
    bool neg = false;
    if (!s.empty() && s.front() == '-') {
        neg = true;
        s.erase(s.begin());
    }
    size_t nz = 0;
    while (nz + 1 < s.size() && s[nz] == '0') ++nz;
    if (nz > 0) s.erase(0, nz);
    if (s.empty()) s = "0";
    if (neg && s != "0") s.insert(s.begin(), '-');
    return s;
}

static bool p95_is_known_factor_string(const std::string& factor, const std::vector<std::string>& known_factors) {
    const std::string nf = p95_normalize_factor_string(factor);
    for (const std::string& s : known_factors) {
        if (p95_normalize_factor_string(s) == nf) return true;
    }
    return false;
}

static std::string p95_join_known_factors_csv(const std::vector<std::string>& factors) {
    std::ostringstream oss;
    bool first = true;
    for (const std::string& s : factors) {
        if (s.empty()) continue;
        if (!first) oss << ',';
        oss << s;
        first = false;
    }
    return oss.str();
}

static bool p95_read_last_non_empty_line(const fs::path& file, std::string& last_line) {
    std::ifstream in(file, std::ios::in);
    if (!in.is_open()) return false;
    std::string line;
    std::string best;
    while (std::getline(in, line)) {
        if (!line.empty()) best = line;
    }
    if (best.empty()) return false;
    last_line = best;
    return true;
}

static bool p95_parse_result_json_line(const std::string& line, std::string& status_out, std::string& factor_out) {
    status_out.clear();
    factor_out.clear();
    std::smatch m;

    // Use raw strings with a custom delimiter so the JSON pattern can contain
    // the sequence `)"` safely.
    if (std::regex_search(line, m, std::regex(R"p95("status"\s*:\s*"([^"]+)")p95"))) {
        status_out = m[1].str();
    }
    if (std::regex_search(line, m, std::regex(R"p95("factors"\s*:\s*\[\s*"([^"]+)")p95"))) {
        factor_out = m[1].str();
    }
    if (factor_out.empty() &&
        std::regex_search(line, m, std::regex(R"p95("factor"\s*:\s*"([^"]+)")p95"))) {
        factor_out = m[1].str();
    }
    return !status_out.empty();
}

static std::string p95_backup_suffix() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    std::ostringstream oss;
    oss << ".bak." << std::put_time(&tmv, "%Y%m%d_%H%M%S");
    return oss.str();
}

static bool p95_backup_existing_file(const fs::path& src, std::string* moved_to = nullptr) {
    std::error_code ec;
    if (!fs::exists(src, ec)) return true;
    fs::path dst = src;
    dst += ".bak";
    if (fs::exists(dst, ec)) {
        dst = src;
        dst += p95_backup_suffix();
    }
    fs::rename(src, dst, ec);
    if (ec) return false;
    if (moved_to) *moved_to = dst.string();
    return true;
}

static bool p95_copy_overwrite(const fs::path& src, const fs::path& dst, std::string& err) {
    std::error_code ec;
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        err = ec.message();
        return false;
    }
    return true;
}

static std::string p95_shell_quote_posix(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char ch : s) {
        if (ch == '\'') out += "'\\''";
        else out.push_back(ch);
    }
    out.push_back('\'');
    return out;
}

static std::string p95_shell_quote_win(const std::string& s) {
    std::string out = "\"";
    for (char ch : s) {
        if (ch == '\"') out += '\\';
        out.push_back(ch);
    }
    out += "\"";
    return out;
}

static bool p95_read_text_file(const fs::path& file, std::string& out) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open()) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

static bool p95_write_text_file(const fs::path& file, const std::string& text, std::string& err) {
    std::ofstream out(file, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        err = "cannot open file for writing";
        return false;
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out.good()) {
        err = "write failed";
        return false;
    }
    out.flush();
    if (!out.good()) {
        err = "flush failed";
        return false;
    }
    return true;
}

static std::string p95_normalize_prime_txt(const std::string& original) {
    std::vector<std::string> lines;
    {
        std::istringstream iss(original);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines.push_back(line);
        }
    }

    struct ForcedSetting {
        const char* key;
        const char* value;
    };
    static const ForcedSetting forced[] = {
        {"AllowLowB1", "1"},
        {"Stage1GCD", "-1"},
        {"UsePrimenet", "0"},
        {"ExitWhenOutOfWork", "1"},
    };

    std::vector<std::string> missing_at_top;
    missing_at_top.reserve(sizeof(forced) / sizeof(forced[0]));

    for (const ForcedSetting& s : forced) {
        const std::string prefix = std::string(s.key) + "=";
        bool found = false;
        for (std::string& line : lines) {
            size_t pos = 0;
            while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])) != 0) ++pos;
            if (line.compare(pos, prefix.size(), prefix) == 0) {
                line = prefix + s.value;
                found = true;
                break;
            }
        }
        if (!found) missing_at_top.push_back(prefix + s.value);
    }

    std::ostringstream oss;
    for (const std::string& line : missing_at_top) oss << line << "\n";
    for (const std::string& line : lines) oss << line << "\n";
    return oss.str();
}

static Prime95Stage2TaskResult p95_run_stage2_task(const fs::path& p95_dir,
                                                   const fs::path& p95_exe,
                                                   const Prime95Stage2Task& task) {
    Prime95Stage2TaskResult result;
    result.curve_idx = task.curve_idx;

    if (p95_dir.empty() || p95_exe.empty()) {
        result.error = "Prime95 path or executable is empty";
        return result;
    }

    const fs::path dst_resume = p95_dir / task.resume_filename;
    std::string copy_err;
    if (!p95_copy_overwrite(task.resume_src_path, dst_resume, copy_err)) {
        result.error = "Failed to copy resume file to Prime95 directory: " + copy_err;
        return result;
    }

    const fs::path prime_txt = p95_dir / "prime.txt";
    const fs::path prime_txt_backup = p95_dir / (std::string("prime.txt") + p95_backup_suffix());
    bool had_prime_txt = false;
    std::string prime_txt_original;
    std::string prime_txt_write_err;
    auto restore_prime_txt = [&]() {
        std::error_code rec;
        if (had_prime_txt) {
            fs::copy_file(prime_txt_backup, prime_txt, fs::copy_options::overwrite_existing, rec);
            fs::remove(prime_txt_backup, rec);
        } else {
            fs::remove(prime_txt, rec);
        }
    };

    if (fs::exists(prime_txt)) {
        had_prime_txt = true;
        if (!p95_read_text_file(prime_txt, prime_txt_original)) {
            result.error = "Failed to read existing Prime95 prime.txt";
            return result;
        }
        if (!p95_copy_overwrite(prime_txt, prime_txt_backup, copy_err)) {
            result.error = "Failed to backup Prime95 prime.txt: " + copy_err;
            return result;
        }
    }

    const std::string prime_txt_modified = p95_normalize_prime_txt(prime_txt_original);
    if (!p95_write_text_file(prime_txt, prime_txt_modified, prime_txt_write_err)) {
        restore_prime_txt();
        result.error = "Failed to prepare Prime95 prime.txt: " + prime_txt_write_err;
        return result;
    }

    std::error_code ec;
    fs::remove(p95_dir / "worktodo.txt", ec);
    ec.clear();
    fs::remove(p95_dir / "results.json.txt", ec);

    {
        std::ofstream wt(p95_dir / "worktodo.txt", std::ios::out | std::ios::trunc);
        if (!wt.is_open()) {
            restore_prime_txt();
            result.error = "Failed to create Prime95 worktodo.txt";
            return result;
        }
        wt << task.worktodo_line << "\n";
        wt.flush();
    }

    std::ostringstream shell;
#ifdef _WIN32
    shell << "cd /d " << p95_shell_quote_win(p95_dir.string())
          << " && " << p95_shell_quote_win(p95_exe.string())
          << " -d > " << p95_shell_quote_win((p95_dir / task.log_filename).string()) << " 2>&1";
    const std::string cmd = std::string("cmd /C ") + p95_shell_quote_win(shell.str());
#else
    shell << "cd " << p95_shell_quote_posix(p95_dir.string())
          << " && " << p95_shell_quote_posix(p95_exe.string())
          << " -d > " << p95_shell_quote_posix((p95_dir / task.log_filename).string()) << " 2>&1";
    const std::string cmd = std::string("sh -lc ") + p95_shell_quote_posix(shell.str());
#endif

    int rc = std::system(cmd.c_str());
#ifdef _WIN32
    result.exit_code = rc;
#else
    if (rc == -1) result.exit_code = -1;
    else if (WIFEXITED(rc)) result.exit_code = WEXITSTATUS(rc);
    else result.exit_code = rc;
#endif

    const fs::path results_file = p95_dir / "results.json.txt";
    for (int attempt = 0; attempt < 200; ++attempt) {
        if (p95_read_last_non_empty_line(results_file, result.json_line)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    restore_prime_txt();

    if (result.json_line.empty()) {
        std::ostringstream oss;
        oss << "Prime95 did not produce results.json.txt for curve " << (task.curve_idx + 1)
            << " (exit_code=" << result.exit_code << ")"
            << " | log=" << (p95_dir / task.log_filename).string()
            << " | worktodo=" << task.worktodo_line;
        result.error = oss.str();
        return result;
    }

    std::string status;
    std::string factor;
    if (!p95_parse_result_json_line(result.json_line, status, factor)) {
        result.error = "Unable to parse Prime95 results.json.txt line";
        return result;
    }

    result.factor = factor;
    result.factor_found = (!factor.empty()) || (status == "F");
    result.known_factor = result.factor_found && !factor.empty() && p95_is_known_factor_string(factor, task.known_factors);
    result.success = (status == "NF") || (status == "F");

    if (!result.success && result.error.empty()) {
        result.error = "Prime95 returned an unsupported status: " + status;
    }
    return result;
}


namespace ecm_local {

struct EC_mod4 {
    struct Pt { mpz_class x, y; bool inf=false; };

    static inline void norm(mpz_class& z, const mpz_class& N) {
        mpz_mod(z.get_mpz_t(), z.get_mpz_t(), N.get_mpz_t());
        if (z < 0) z += N;
    }

    static Pt dbl(const Pt& P, const mpz_class& N) {
        if (P.inf) return P;
        mpz_class num = 3 * P.x * P.x + 4;
        mpz_class den = 2 * P.y, inv;
        norm(num, N); norm(den, N);
        if (mpz_sgn(den.get_mpz_t()) == 0) return Pt{{}, {}, true};
        if (!mpz_invert(inv.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()))
            return Pt{{}, {}, true};
        mpz_class lambda = (num * inv) % N; if (lambda < 0) lambda += N;

        mpz_class x3 = (lambda * lambda - 2 * P.x) % N; if (x3 < 0) x3 += N;
        mpz_class y3 = (lambda * (P.x - x3) - P.y) % N; if (y3 < 0) y3 += N;
        return Pt{x3, y3, false};
    }

    static Pt add(const Pt& P, const Pt& Q, const mpz_class& N) {
        if (P.inf) {return Q;}
        if (Q.inf) {return P;}
        if (P.x == Q.x) {
            mpz_class ysum = (P.y + Q.y) % N; if (ysum < 0) ysum += N;
            if (ysum == 0) return Pt{{}, {}, true};
            return dbl(P, N);
        }
        mpz_class num = Q.y - P.y; norm(num, N);
        mpz_class den = Q.x - P.x; norm(den, N);
        mpz_class inv;
        if (!mpz_invert(inv.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()))
            return Pt{{}, {}, true};
        mpz_class lambda = (num * inv) % N; if (lambda < 0) lambda += N;

        mpz_class x3 = (lambda * lambda - P.x - Q.x) % N; if (x3 < 0) x3 += N;
        mpz_class y3 = (lambda * (P.x - x3) - P.y) % N; if (y3 < 0) y3 += N;
        return Pt{x3, y3, false};
    }
    #ifdef _MSC_VER
    #  include <intrin.h>
    #endif

    static inline int msb_index_u64(uint64_t n) {
        if (!n) return -1;
    #if defined(_MSC_VER) && !defined(__clang__)
        unsigned long idx;
    #if defined(_M_X64) || defined(_M_ARM64)
        _BitScanReverse64(&idx, n);
        return (int)idx;
    #else
        // 32-bit MSVC fallback
        unsigned long hi = (unsigned long)(n >> 32);
        if (hi) { _BitScanReverse(&idx, hi); return (int)idx + 32; }
        _BitScanReverse(&idx, (unsigned long)(n & 0xFFFFFFFFu));
        return (int)idx;
    #endif
    #else
        // GCC/Clang
        return 63 - __builtin_clzll(n);
    #endif
    }

    static void get(uint64_t n, int s1, int t1, const mpz_class& N, mpz_class& s, mpz_class& t) {
        Pt P0, P;
        P0.x = s1; if (s1 < 0) P0.x += N; P0.x %= N;
        P0.y = t1; if (t1 < 0) P0.y += N; P0.y %= N;
        P    = P0;

        int msb = msb_index_u64(n);
        for (int b = msb - 1; b >= 0; --b) {
            P = dbl(P, N);
            if (((n >> b) & 1ULL) != 0) P = add(P, P0, N);
            if (P.inf) { s = 0; t = 0; return; }
        }
        s = P.x; t = P.y;
    }
};

} // namespace ecm_local

namespace core {
    static fs::path p95_expand_user_path(const std::string& in) {
        if (in.empty()) return fs::path();

        #ifdef _WIN32
            if (in.size() >= 2 && in[0] == '~' && (in[1] == '/' || in[1] == '\\')) {
                const char* home = std::getenv("USERPROFILE");
                if (!home) {
                    const char* drive = std::getenv("HOMEDRIVE");
                    const char* path  = std::getenv("HOMEPATH");
                    if (drive && path) return fs::path(std::string(drive) + path) / in.substr(2);
                } else {
                    return fs::path(home) / in.substr(2);
                }
            }
        #else
            if (in[0] == '~' && (in.size() == 1 || in[1] == '/')) {
                const char* home = std::getenv("HOME");
                if (home && *home) {
                    if (in.size() == 1) return fs::path(home);
                    return fs::path(home) / in.substr(2);
                }
            }
        #endif

            return fs::path(in);
    }
int App::runECMMarinTwistedEdwards()
{
    using namespace std;
    using namespace std::chrono;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const uint64_t B1 = options.B1 ? options.B1 : 1000000ULL;
    const uint64_t B2 = options.B2 ? options.B2 : 0ULL;
    uint64_t curves = options.nmax ? options.nmax : (options.K ? options.K : 250);
    const bool verbose = true;
    const bool forceCurve = (options.curve_seed != 0ULL);
    const bool forceSigma = !options.sigma.empty();
    if (forceCurve) curves =1ULL;
    const uint32_t progress_interval_ms = (options.ecm_progress_interval_ms > 0) ? options.ecm_progress_interval_ms : 2000;

    const bool stage2_debug_checks = true;

    auto u64_bits = [](uint64_t x)->size_t{ if(!x) return 1; size_t n=0; while(x){ ++n; x>>=1; } return n; };

    auto splitmix64_step = [](uint64_t& x)->uint64_t{
        x += 0x9E3779B97f4A7C15ULL;
        uint64_t z=x;
        z^=z>>30; z*=0xBF58476D1CE4E5B9ULL;
        z^=z>>27; z*=0x94D049BB133111EBULL;
        z^=z>>31;
        return z;
    };
    auto splitmix64_u64 = [&](uint64_t seed0)->uint64_t{
        uint64_t s=seed0;
        return splitmix64_step(s);
    };
    auto mix64 = [&](uint64_t seed, uint64_t idx)->uint64_t{
        uint64_t x = seed ^ 0x9E3779B97f4A7C15ULL;
        x ^= (idx+1) * 0xBF58476D1CE4E5B9ULL;
        x ^= (idx+0x100) * 0x94D049BB133111EBULL;
        return splitmix64_u64(x);
    };
    auto rnd_mpz_bits = [&](const mpz_class& N, uint64_t seed0, unsigned bits)->mpz_class{
        mpz_class z = 0;
        uint64_t s = seed0;
        for (unsigned i=0;i<bits;i+=64){
            z <<= 64;
            z += (unsigned long)splitmix64_step(s);
        }
        z %= N;
        if (z <= 2) z += 3;
        return z;
    };
    auto fmt_hms = [&](double s)->string{
        uint64_t u=(uint64_t)(s+0.5);
        uint64_t h=u/3600,m=(u%3600)/60,se=u%60;
        ostringstream ss; ss<<h<<"h "<<m<<"m "<<se<<"s";
        return ss.str();
    };

    mpz_class N = (mpz_class(1) << p) - 1;

#ifdef SIGINT
    std::signal(SIGINT, handle_sigint);
#endif
#ifdef SIGTERM
    std::signal(SIGTERM, handle_sigint);
#endif
#ifdef SIGQUIT
    std::signal(SIGQUIT, handle_sigint);
#endif
    interrupted.store(false, std::memory_order_relaxed);

    auto run_start = high_resolution_clock::now();
    uint32_t bits_B1 = u64_bits(B1);
    uint32_t bits_B2 = B2 ? u64_bits(B2) : 0;
    uint64_t mersenne_digits = (uint64_t)mpz_sizeinbase(N.get_mpz_t(),10);

    bool wrote_result = false;
    string mode_name = "twisted_edwards";
    const bool te_use_torsion16 = (!options.notorsion && options.torsion16) && !forceSigma;
    const bool te_use_family_iv_163 = (!options.notorsion && !options.torsion16 && options.family_iv_163) && !forceSigma;
    const uint8_t current_te_family_mode = te_use_torsion16 ? 1 : (te_use_family_iv_163 ? 2 : 0);
    string torsion_name = te_use_torsion16 ? string("16") : (te_use_family_iv_163 ? string("family_iv_163") : string("none"));
    string result_status = "NF";
    mpz_class result_factor = 0;
    uint64_t curves_tested_for_found = 0;
    size_t transform_size_once = 0;

    auto write_result = [&](){
        if (wrote_result) return;
        double tot = duration<double>(high_resolution_clock::now() - run_start).count();
        uint64_t tested = curves_tested_for_found ?
                          curves_tested_for_found :
                          ((result_status=="found") ? curves_tested_for_found : curves);
        ofstream jf("ecm_result.json", ios::app);
        ostringstream js;
        js<<"{";
        js<<"\"exponent\":"<<p<<",";
        js<<"\"mode\":\""<<mode_name<<"\",";
        js<<"\"torsion\":\""<<torsion_name<<"\",";
        js<<"\"B1\":"<<B1<<",";
        js<<"\"B2\":"<<B2<<",";
        js<<"\"bits_B1\":"<<bits_B1<<",";
        js<<"\"bits_B2\":"<<bits_B2<<",";
        js<<"\"factor\":\""<<(result_factor>0? result_factor.get_str() : string("0"))<<"\",";
        js<<"\"factor_digits\":"<<(result_factor>0? (int)mpz_sizeinbase(result_factor.get_mpz_t(),10) : 0)<<",";
        js<<"\"mersenne_digits\":"<<mersenne_digits<<",";
        js<<fixed<<setprecision(6);
        js<<"\"elapsed_total_s\":"<<tot<<",";
        js<<"\"elapsed_avg_per_curve_s\":"<<(tested? (tot/double(tested)) : 0.0)<<",";
        js<<"\"curves_tested\":"<<tested<<",";
        js<<"\"status\":\""<<result_status<<"\"";
        js<<"}";
        jf<<js.str()<<"\n";
        wrote_result = true;
    };

    auto is_known = [&](const mpz_class& g)->bool{
        for (auto &s: options.knownFactors){
            if (s.empty()) continue;
            mpz_class f;
            if (mpz_set_str(f.get_mpz_t(), s.c_str(), 0) != 0) continue;
            if (f < 0) f = -f;
            if (f > 1 && g == f) return true;
        }
        return false;
    };

    auto publish_json = [&](){
        vector<string> saved = options.knownFactors;
        if (result_factor > 0) {
            options.knownFactors.clear();
            options.knownFactors.push_back(result_factor.get_str());
        }
        string json_out = io::JsonBuilder::generate(options, static_cast<int>(transform_size_once), false, string(), string());
        ofstream jf2("ecm_result.json", ios::app);
        jf2 << json_out << "\n";
        cout << "[ECM] json for manual submit to primenet:\n" << json_out << endl;
        options.knownFactors = saved;
        io::WorktodoManager wm(options);
        wm.appendToResultsTxt(json_out);

        if (hasWorktodoEntry_) {
            if (worktodoParser_->removeFirstProcessed()) {
                std::cout << "Entry removed from " << options.worktodo_path
                          << " and saved to worktodo_save.txt\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss  << "Entry removed from " << options.worktodo_path
                         << " and saved to worktodo_save.txt\n";
                    guiServer_->appendLog(oss.str());
                }
                std::ifstream f(options.worktodo_path);
                std::string    l;
                bool           more = false;
                while (std::getline(f, l)) {
                    if (!l.empty() && l[0] != '#') {
                        more = true;
                        break;
                    }
                }
                f.close();

                if (more) {
                    std::cout << "Restarting for next entry in worktodo.txt\n";
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss  << "Entry removed from " << options.worktodo_path
                             << " and saved to worktodo_save.txt\n";
                        guiServer_->appendLog(oss.str());
                    }
                    restart_self(argc_, argv_);
                } else {
                    std::cout << "No more entries in worktodo.txt, exiting.\n";
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss  << "No more entries in worktodo.txt, exiting.\n";
                        guiServer_->appendLog(oss.str());
                    }
                    if (!options.gui) {
                        std::exit(0);
                    }
                }
            } else {
                std::cerr << "Failed to update " << options.worktodo_path << "\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss  << "Failed to update " << options.worktodo_path << "\n";
                    guiServer_->appendLog(oss.str());
                }
                if (!options.gui) {
                    std::exit(-1);
                }
            }
        }
    };




    const bool want_p95_stage2 = (options.p95stage2 && !options.p95path.empty() && B2 > B1);
    bool p95_stage2_enabled = false;
    fs::path p95_dir;
    fs::path p95_exe;
    std::deque<Prime95Stage2Task> p95_pending_tasks;
    std::optional<Prime95Stage2Task> p95_active_task;
    std::future<Prime95Stage2TaskResult> p95_future;
    bool p95_future_active = false;
    std::string p95_background_error;

    auto p95_log = [&](const std::string& msg)->void {
        std::cout << msg << std::endl;
        if (guiServer_) guiServer_->appendLog(msg);
    };

    auto p95_finalize_background_stop = [&](engine* eng_or_null)->int {
        if (!p95_background_error.empty()) {
            p95_log(std::string("[ECM] Prime95 Stage2 background error: ") + p95_background_error);
            if (eng_or_null) delete eng_or_null;
            return 1;
        }
        if (result_status == "found" && result_factor > 0) {
            std::ostringstream oss;
            oss << "[ECM] Prime95 Stage2 background found factor " << result_factor.get_str();
            p95_log(oss.str());
            write_result();
            publish_json();
            if (eng_or_null) delete eng_or_null;
            return 0;
        }
        if (eng_or_null) delete eng_or_null;
        return 0;
    };

    auto p95_start_next_task = [&]() -> void {
        if (!p95_stage2_enabled || p95_future_active || p95_pending_tasks.empty()) return;
        p95_active_task = p95_pending_tasks.front();
        p95_pending_tasks.pop_front();
        Prime95Stage2Task task = *p95_active_task;
        std::ostringstream oss;
        oss << "[ECM] Prime95 Stage2 background start | curve=" << (task.curve_idx + 1)
            << " | queue_remaining=" << p95_pending_tasks.size()
            << " | log=" << (p95_dir / task.log_filename).string()
            << " | resume=" << (p95_dir / task.resume_filename).string();
        p95_log(oss.str());
        p95_future = std::async(std::launch::async, [p95_dir, p95_exe, task]() {
            return p95_run_stage2_task(p95_dir, p95_exe, task);
        });
        p95_future_active = true;
    };

    auto p95_handle_finished_task = [&](const Prime95Stage2TaskResult& rr)->bool {
        if (!rr.error.empty()) {
            p95_background_error = rr.error;
            interrupted.store(true, std::memory_order_relaxed);
            return true;
        }

        std::ostringstream oss;
        oss << "[ECM] Prime95 Stage2 background done | curve=" << (rr.curve_idx + 1)
            << " | exit_code=" << rr.exit_code;
        if (!rr.json_line.empty()) oss << " | result=" << rr.json_line;
        p95_log(oss.str());

        if (rr.factor_found && !rr.factor.empty()) {
            if (rr.known_factor) {
                std::ostringstream ok;
                ok << "[ECM] Prime95 Stage2 background curve " << (rr.curve_idx + 1)
                   << " found known factor " << rr.factor << ", continuing";
                p95_log(ok.str());
                return false;
            }

            if (!p95_active_task.has_value()) {
                p95_background_error = "Prime95 factor result received without active task metadata";
                interrupted.store(true, std::memory_order_relaxed);
                return true;
            }

            const Prime95Stage2Task& found_task = *p95_active_task;

            if (mpz_set_str(result_factor.get_mpz_t(), rr.factor.c_str(), 10) != 0) {
                p95_background_error = std::string("Failed to parse Prime95 factor: ") + rr.factor;
                interrupted.store(true, std::memory_order_relaxed);
                return true;
            }

            options.sigma_hex = found_task.sigma_hex;
            options.curve_seed = found_task.curve_seed;
            options.base_seed = found_task.base_seed;

            result_status = "found";
            curves_tested_for_found = rr.curve_idx + 1;
            options.curves_tested_for_found = (uint32_t)(rr.curve_idx + 1);
            interrupted.store(true, std::memory_order_relaxed);
            return true;
        }
        return false;
    };

    auto p95_poll_background = [&](bool wait_for_active)->bool {
        if (!p95_stage2_enabled) return false;
        if (p95_future_active) {
            if (wait_for_active) {
                Prime95Stage2TaskResult rr = p95_future.get();
                p95_future_active = false;
                bool stop = p95_handle_finished_task(rr);
                p95_active_task.reset();
                if (!stop) p95_start_next_task();
                return stop;
            }
            if (p95_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                Prime95Stage2TaskResult rr = p95_future.get();
                p95_future_active = false;
                bool stop = p95_handle_finished_task(rr);
                p95_active_task.reset();
                if (!stop) p95_start_next_task();
                return stop;
            }
        }
        if (!p95_future_active) p95_start_next_task();
        return false;
    };

    auto p95_enqueue_curve = [&](uint64_t curve_idx,
                                const std::string& curve_resume_path,
                                const std::string& curve_sigma_hex,
                                uint64_t curve_seed_for_task,
                                uint64_t base_seed_for_task)->bool {
        if (!p95_stage2_enabled || curve_resume_path.empty()) return false;
        const std::string resume_filename = fs::path(curve_resume_path).filename().string();
        const std::string known_csv = p95_join_known_factors_csv(options.knownFactors);

        std::ostringstream wt;
        wt << "ECMSTAGE2=N/A,1,2," << p << ",-1,\"" << resume_filename << "\"," << B2;
        if (!known_csv.empty()) {
            wt << ",\"" << known_csv << "\"";
        }

        Prime95Stage2Task task;
        task.curve_idx = curve_idx;
        task.resume_src_path = curve_resume_path;
        task.resume_filename = resume_filename;
        task.worktodo_line = wt.str();
        task.known_factors = options.knownFactors;
        task.sigma_hex = curve_sigma_hex;
        task.curve_seed = curve_seed_for_task;
        task.base_seed = base_seed_for_task;
        {
            std::ostringstream logn;
            logn << "prmers_p95stage2_curve_" << std::setw(6) << std::setfill('0') << (curve_idx + 1) << ".log";
            task.log_filename = logn.str();
        }
        p95_pending_tasks.push_back(task);
        std::ostringstream oss;
        oss << "[ECM] Curve " << (curve_idx + 1) << "/" << curves
            << " | Stage2 queued for Prime95 background"
            << " | pending=" << p95_pending_tasks.size();
        p95_log(oss.str());
        p95_start_next_task();
        return true;
    };

    if (want_p95_stage2) {
        
        //p95_dir = fs::path(options.p95path);
        p95_dir = p95_expand_user_path(options.p95path);
        std::error_code ec;
        if (!fs::exists(p95_dir, ec) || !fs::is_directory(p95_dir, ec)) {
            //p95_log(std::string("[ECM] Prime95 Stage2 disabled: invalid directory '") + options.p95path + "'");
            p95_log(std::string("[ECM] Prime95 Stage2 disabled: invalid directory '") + options.p95path + "' -> resolved to '" + p95_dir.string() + "'");
        } else {
#ifdef _WIN32
            const std::vector<std::string> exe_candidates = {"prime95.exe", "mprime.exe"};
#else
            const std::vector<std::string> exe_candidates = {"mprime", "prime95"};
#endif
            for (const std::string& cand : exe_candidates) {
                fs::path test = p95_dir / cand;
                if (fs::exists(test, ec)) {
                    p95_exe = fs::absolute(test, ec);
                    if (ec) p95_exe = test;
                    break;
                }
            }
            if (p95_exe.empty()) {
                p95_log(std::string("[ECM] Prime95 Stage2 disabled: no Prime95/mprime executable found in '") + options.p95path + "'");
            } else {
                std::string moved_worktodo;
                std::string moved_results;
                bool ok_worktodo = p95_backup_existing_file(p95_dir / "worktodo.txt", &moved_worktodo);
                bool ok_results = p95_backup_existing_file(p95_dir / "results.json.txt", &moved_results);
                if (!ok_worktodo || !ok_results) {
                    p95_log(std::string("[ECM] Prime95 Stage2 disabled: failed to backup worktodo/results in '") + options.p95path + "'");
                } else {
                    if (!moved_worktodo.empty()) p95_log(std::string("[ECM] Prime95 worktodo backed up to ") + moved_worktodo);
                    if (!moved_results.empty()) p95_log(std::string("[ECM] Prime95 results.json.txt backed up to ") + moved_results);
                    p95_stage2_enabled = true;
                    p95_log(std::string("[ECM] Prime95 Stage2 background enabled using ") + p95_exe.string());
                }
            }
        }
    }

    {
        vector<string> kf = options.knownFactors;
        vector<pair<mpz_class,unsigned>> acc;
        mpz_class C = N;
        for (auto& ss : kf) {
            if (ss.empty()) continue;
            mpz_class f;
            if (mpz_set_str(f.get_mpz_t(), ss.c_str(), 0) != 0) continue;
            if (f < 0) f = -f;
            if (f <= 1) continue;
            mpz_class gk;
            mpz_gcd(gk.get_mpz_t(), f.get_mpz_t(), N.get_mpz_t());
            if (gk > 1) {
                unsigned m = 0;
                while (mpz_divisible_p(C.get_mpz_t(), gk.get_mpz_t())) { C /= gk; ++m; }
                if (m) acc.push_back({gk, m});
            }
        }
        if (!acc.empty()) {
            for (auto& kv : acc) while (mpz_divisible_p(N.get_mpz_t(), kv.first.get_mpz_t())) N /= kv.first;
            if (N == 1) {
                std::cout << "[ECM] Trivial after removing known factors." << std::endl;
                write_result();
                publish_json();
                return 0;
            }
        }
    }

    const std::string ecm_stage1_resume_save_file =
        "resume_p" + std::to_string(p) + "_ECM_TE_B1_" + std::to_string(B1) + ".save";
    const std::string ecm_stage1_resume_p95_file =
        "resume_p" + std::to_string(p) + "_ECM_TE_B1_" + std::to_string(B1) + ".p95";

    auto ecm_now_string = [&]() -> std::string {
        std::time_t t = std::time(nullptr);
        std::tm tmv{};
    #if defined(_WIN32)
        localtime_s(&tmv, &t);
    #else
        localtime_r(&t, &tmv);
    #endif
        std::ostringstream oss;
        oss << std::put_time(&tmv, "%a %b %d %H:%M:%S %Y");
        return oss.str();
    };

    auto append_ecm_stage1_resume_line = [&](uint64_t curve_idx, const mpz_class& Aresume, const mpz_class& xAff, const mpz_class* sigmaForP95)->std::string {
        std::string curve_p95_resume_path;
        mpz_class Ared = Aresume;
        mpz_mod(Ared.get_mpz_t(), Ared.get_mpz_t(), N.get_mpz_t());
        if (Ared < 0) Ared += N;

        mpz_class xred = xAff;
        mpz_mod(xred.get_mpz_t(), xred.get_mpz_t(), N.get_mpz_t());
        if (xred < 0) xred += N;

        const std::string ecmResumeProgram = std::string("PrMers ") + core::PRMERS_VERSION;
        const std::string ecmResumeTime = ecm_now_string();
        const std::string ecmResumeWho = options.user;

        mpz_class chk;
        mpz_set_ui(chk.get_mpz_t(), (unsigned long)B1);
        chk *= mpz_class(mpz_fdiv_ui(Ared.get_mpz_t(), CHKSUMMOD));
        chk *= mpz_class(mpz_fdiv_ui(N.get_mpz_t(), CHKSUMMOD));
        chk *= mpz_class(mpz_fdiv_ui(xred.get_mpz_t(), CHKSUMMOD));
        const uint32_t chk_u = (uint32_t)mpz_fdiv_ui(chk.get_mpz_t(), CHKSUMMOD);

        {
            std::ofstream out(ecm_stage1_resume_save_file, std::ios::out | std::ios::app);
            const std::string nField = ("2^" + std::to_string(p) + "-1");

            if (!out) {
                std::ostringstream oss;
                oss << "[ECM] Warning: cannot append Stage1 resume to '" << ecm_stage1_resume_save_file << "'";
                std::cerr << oss.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(oss.str());
            } else {
                out << "METHOD=ECM; B1=" << B1
                    << "; N=" << nField
                    << "; X=0x" << xred.get_str(16)
                    << "; A=" << Ared.get_str()
                    << "; CHECKSUM=" << chk_u
                    << "; PROGRAM=" << ecmResumeProgram
                    << "; X0=0x0; Y0=0x0;";
                if (!ecmResumeWho.empty()) out << " WHO=" << ecmResumeWho << ";";
                out << " TIME=" << ecmResumeTime << ";\n";
                out.flush();
            }
        }

        if (sigmaForP95 != nullptr) {
            const mpz_class& sigma = *sigmaForP95;

            mpz_class chk2;
            mpz_set_ui(chk2.get_mpz_t(), (unsigned long)B1);
            chk2 *= mpz_class(mpz_fdiv_ui(sigma.get_mpz_t(), CHKSUMMOD));
            chk2 *= mpz_class(mpz_fdiv_ui(N.get_mpz_t(), CHKSUMMOD));
            chk2 *= mpz_class(mpz_fdiv_ui(xred.get_mpz_t(), CHKSUMMOD));
            const uint32_t chk_u2 = (uint32_t)mpz_fdiv_ui(chk2.get_mpz_t(), CHKSUMMOD);

            //const std::string nField = N.get_str();
            const std::string nField = ("2^" + std::to_string(p) + "-1");

            std::ofstream outp(ecm_stage1_resume_p95_file, std::ios::out | std::ios::app);
            if (!outp) {
                std::ostringstream oss;
                oss << "[ECM] Warning: cannot append Prime95 Stage2 resume to '" << ecm_stage1_resume_p95_file << "'";
                std::cerr << oss.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(oss.str());
            } else {
                outp << "METHOD=ECM; SIGMA=" << sigma.get_str()
                     << "; B1=" << B1
                     << "; N=" << nField
                     << "; X=0x" << xred.get_str(16)
                     << "; CHECKSUM=" << chk_u2
                     << "; PROGRAM=" << ecmResumeProgram
                     << "; X0=0x0; Y0=0x0;";
                if (!ecmResumeWho.empty()) outp << " WHO=" << ecmResumeWho << ";";
                outp << " TIME=" << ecmResumeTime << ";\n";
                outp.flush();
            }

            std::ostringstream one_name;
            one_name << "resume_p" << p << "_ECM_TE_B1_" << B1
                     << "_c" << std::setw(6) << std::setfill('0') << (curve_idx + 1) << ".p95";
            curve_p95_resume_path = one_name.str();

            std::ofstream outpc(curve_p95_resume_path, std::ios::out | std::ios::trunc);
            if (!outpc) {
                std::ostringstream oss;
                oss << "[ECM] Warning: cannot write per-curve Prime95 Stage2 resume to '" << curve_p95_resume_path << "'";
                std::cerr << oss.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(oss.str());
                curve_p95_resume_path.clear();
            } else {
                outpc << "METHOD=ECM; SIGMA=" << sigma.get_str()
                      << "; B1=" << B1
                      << "; N=" << nField
                      << "; X=0x" << xred.get_str(16)
                      << "; CHECKSUM=" << chk_u2
                      << "; PROGRAM=" << ecmResumeProgram
                      << "; X0=0x0; Y0=0x0;";
                if (!ecmResumeWho.empty()) outpc << " WHO=" << ecmResumeWho << ";";
                outpc << " TIME=" << ecmResumeTime << ";\n";
                outpc.flush();
            }
        }

        std::ostringstream oss;
        oss << "[ECM] Curve " << (curve_idx + 1) << "/" << curves
            << " | Stage1 resume appended: " << ecm_stage1_resume_save_file;
        if (sigmaForP95 != nullptr) oss << " + " << ecm_stage1_resume_p95_file;
        else oss << " (Prime95 export skipped: TE curve is not a GMP-ECM SIGMA family)";
        std::cout << oss.str() << std::endl;
        if (guiServer_) guiServer_->appendLog(oss.str());
        return curve_p95_resume_path;
    };

    vector<uint64_t> primesB1_v, primesS2_v;
    {
        const uint64_t Pmax = (B2 > B1) ? B2 : B1;
        const uint64_t root = ecm_isqrt_u64(Pmax);
        const std::vector<uint32_t> basePrimes = ecm_sieve_base_primes((uint32_t)root);

        const uint64_t SEG = 1000000ULL;
        std::vector<uint64_t> segPrimes;

        for (uint64_t lo = 2; lo <= B1; ) {
            uint64_t hi = lo + SEG - 1;
            if (hi < lo || hi > B1) hi = B1;
            ecm_segmented_primes_odd(lo, hi, basePrimes, segPrimes);
            primesB1_v.insert(primesB1_v.end(), segPrimes.begin(), segPrimes.end());
            if (hi == B1) break;
            lo = hi + 1;
        }

        if (B2 > B1) {
            uint64_t lo = B1 + 1;
            while (lo <= B2) {
                uint64_t hi = lo + SEG - 1;
                if (hi < lo || hi > B2) hi = B2;
                ecm_segmented_primes_odd(lo, hi, basePrimes, segPrimes);
                primesS2_v.insert(primesS2_v.end(), segPrimes.begin(), segPrimes.end());
                if (hi == B2) break;
                lo = hi + 1;
            }
        }

        std::cout << "[ECM] Prime counts: B1=" << primesB1_v.size()
                  << ", S2=" << primesS2_v.size() << std::endl;
    }

    std::vector<uint32_t> s2_chunk_ends;
    std::vector<uint64_t> s2_chunk_prefix_iters;
    uint64_t total_s2_iters = 0;

    mpz_class K(1);
    {
        std::ostringstream msg;
        msg << "[ECM] Building Stage1 exponent K from B1=" << B1 << " ...";
        std::cout << msg.str() << std::endl;
        if (guiServer_) guiServer_->appendLog(msg.str());

        using clock = std::chrono::steady_clock;
        auto t0k = clock::now();
        auto tlast = t0k;
        size_t last_done = 0;
        double ema_pps = 0.0;

        const size_t total = primesB1_v.size();
        const uint32_t BLOCK_BITS = 32768u;
        const double L_est_bits = 1.4426950408889634 * (double)B1;
        std::vector<mpz_class> chunks;
        chunks.reserve((size_t)std::ceil(L_est_bits / (double)BLOCK_BITS) + 8);

        mpz_class chunk(1);
        double approx_bits = 0.0;

        for (size_t i = 0; i < total; ++i) {
            const uint64_t q = primesB1_v[i];
            uint64_t mm = q;
            while (mm <= B1 / q) mm *= q;

            ecm_mpz_mul_u64(chunk, mm);
            approx_bits += std::log2((long double)mm);

            if (approx_bits >= (double)BLOCK_BITS) {
                chunks.push_back(chunk);
                chunk = 1;
                approx_bits = 0.0;
            }

            auto now = clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - tlast).count() >= 400 || i + 1 == total) {
                const size_t done = i + 1;
                const double elapsed = std::chrono::duration<double>(now - t0k).count();
                const double dt = std::chrono::duration<double>(now - tlast).count();
                const double dd = (double)(done - last_done);
                const double inst = (dt > 1e-9 && dd >= 0.0) ? (dd / dt) : (done / std::max(1e-9, elapsed));
                if (ema_pps <= 0.0) ema_pps = inst;
                else ema_pps = 0.75 * ema_pps + 0.25 * inst;
                const double eta = (ema_pps > 0.0 && done < total) ? ((double)(total - done) / ema_pps) : 0.0;

                std::ostringstream line;
                line << "[ECM] Building K " << done << "/" << total
                     << " (" << std::fixed << std::setprecision(1) << (100.0 * (double)done / (double)total) << "%)"
                     << " | primes/s " << std::fixed << std::setprecision(1) << ema_pps
                     << " | ETA " << fmt_hms(eta);
                ecm_print_progress_line(line.str());

                last_done = done;
                tlast = now;
            }
        }

        if (chunk != 1) chunks.push_back(chunk);
        std::cout << std::endl;

        if (chunks.empty()) {
            K = 1;
        } else {
            std::ostringstream m2;
            m2 << "[ECM] Reducing " << chunks.size() << " chunks...";
            std::cout << m2.str() << std::endl;
            if (guiServer_) guiServer_->appendLog(m2.str());

            auto tr0 = clock::now();
            auto trl = tr0;
            const size_t initial = chunks.size();

            while (chunks.size() > 1) {
                std::vector<mpz_class> next;
                next.reserve((chunks.size() + 1) / 2);
                for (size_t j = 0; j + 1 < chunks.size(); j += 2) next.push_back(chunks[j] * chunks[j + 1]);
                if (chunks.size() & 1) next.push_back(chunks.back());
                chunks.swap(next);

                auto now = clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - trl).count() >= 400 || chunks.size() == 1) {
                    const double elapsed = std::chrono::duration<double>(now - tr0).count();
                    const double pct = 100.0 * (1.0 - (double)chunks.size() / (double)initial);
                    std::ostringstream line;
                    line << "[ECM] Reducing K " << std::fixed << std::setprecision(1) << pct
                         << "% | remaining " << chunks.size()
                         << " | elapsed " << std::fixed << std::setprecision(1) << elapsed << "s";
                    ecm_print_progress_line(line.str());
                    trl = now;
                }
            }
            std::cout << std::endl;
            K = chunks[0];
        }

        std::ostringstream done;
        done << "[ECM] K built (" << mpz_sizeinbase(K.get_mpz_t(), 2) << " bits)";
        std::cout << done.str() << std::endl;
        if (guiServer_) guiServer_->appendLog(done.str());
    }
    size_t Kbits = mpz_sizeinbase(K.get_mpz_t(),2);

    {
        std::ostringstream hdr;
        hdr<<"[ECM] N=M_"<<p<<"  B1="<<B1<<"  B2="<<B2<<"  curves="<<curves<<"\n";
        hdr<<"[ECM] Stage1: prime powers up to B1="<<primesB1_v.size()<<", K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<"\n";
        if (!primesS2_v.empty()) hdr<<"[ECM] Stage2 primes ("<<B1<<","<<B2<<"] count="<<primesS2_v.size()<<"\n"; else hdr<<"[ECM] Stage2: disabled\n";
        std::cout<<hdr.str(); if (guiServer_) guiServer_->appendLog(hdr.str());
    }



    const int backup_period = options.backup_interval > 0 ? options.backup_interval : 10;

    string torsion_last = torsion_name;

    uint64_t resume_curve_idx  = 0;
    uint64_t resume_curve_seed = 0;
    bool     have_resume_seed  = false;
    bool     have_resume_stage2 = false;

    auto try_probe_te_ckpt = [&](const std::string& file, uint64_t& out_seed)->bool {
        File f(file);
        if (!f.exists()) return false;

        int version = 0;
        if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return false;
        if (version != 1) return false; // checkpoints S1 Twisted-Edwards

        uint32_t rp = 0;
        if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return false;
        if (rp != p) return false;

        uint32_t i   = 0;
        uint32_t nbb = 0;
        uint64_t rB1 = 0;
        double   et  = 0.0;
        if (!f.read(reinterpret_cast<char*>(&i),   sizeof(i)))   return false;
        if (!f.read(reinterpret_cast<char*>(&nbb), sizeof(nbb))) return false;
        if (!f.read(reinterpret_cast<char*>(&rB1), sizeof(rB1))) return false;
        if (!f.read(reinterpret_cast<char*>(&et),  sizeof(et)))  return false;
        if (rB1 != B1) return false;

        uint64_t saved_curve_seed = 0;
        uint8_t  saved_curve_family = 0;
        if (!f.read(reinterpret_cast<char*>(&saved_curve_seed), sizeof(saved_curve_seed))) return false;
        if (!f.read(reinterpret_cast<char*>(&saved_curve_family),  sizeof(saved_curve_family)))  return false;
        uint8_t current_curve_family = current_te_family_mode;
        if (saved_curve_family != current_curve_family) return false;

        out_seed = saved_curve_seed;
        return true;
    };

    auto try_probe_te_ckpt2 = [&](const std::string& file, uint64_t& out_seed, bool& out_has_seed)->bool {
        File f(file);
        if (!f.exists()) return false;

        int version = 0;
        if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return false;
        if (version != 2 && version != 3 && version != 4 && version != 5 && version != 6) return false;

        uint32_t rp = 0;
        if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return false;
        if (rp != p) return false;

        uint32_t idx = 0;
        uint32_t cnt_bits = 0;
        if (!f.read(reinterpret_cast<char*>(&idx), sizeof(idx))) return false;
        if (!f.read(reinterpret_cast<char*>(&cnt_bits), sizeof(cnt_bits))) return false;

        uint64_t b1s = 0, b2s = 0;
        if (!f.read(reinterpret_cast<char*>(&b1s), sizeof(b1s))) return false;
        if (!f.read(reinterpret_cast<char*>(&b2s), sizeof(b2s))) return false;
        if (b1s != B1 || b2s != B2) return false;

        out_seed = 0;
        out_has_seed = false;
        if (version == 6 || version == 5) {
            uint64_t saved_seed = 0;
            uint8_t saved_tor = 0;
            double et = 0.0;
            uint8_t in_chunk = 0;
            uint32_t chunk_start_idx = 0, chunk_end_idx = 0, chunk_bits_saved = 0, chunk_steps_done_saved = 0;
            if (!f.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed))) return false;
            if (!f.read(reinterpret_cast<char*>(&saved_tor), sizeof(saved_tor))) return false;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return false;
            if (!f.read(reinterpret_cast<char*>(&in_chunk), sizeof(in_chunk))) return false;
            if (!f.read(reinterpret_cast<char*>(&chunk_start_idx), sizeof(chunk_start_idx))) return false;
            if (!f.read(reinterpret_cast<char*>(&chunk_end_idx), sizeof(chunk_end_idx))) return false;
            if (!f.read(reinterpret_cast<char*>(&chunk_bits_saved), sizeof(chunk_bits_saved))) return false;
            if (!f.read(reinterpret_cast<char*>(&chunk_steps_done_saved), sizeof(chunk_steps_done_saved))) return false;
            uint8_t current_tor = current_te_family_mode;
            if (saved_tor != current_tor) return false;
            out_seed = saved_seed;
            out_has_seed = true;
            return true;
        }
        if (version == 4) {
            uint64_t saved_seed = 0;
            uint8_t saved_tor = 0;
            double et = 0.0;
            if (!f.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed))) return false;
            if (!f.read(reinterpret_cast<char*>(&saved_tor), sizeof(saved_tor))) return false;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return false;
            uint8_t current_tor = current_te_family_mode;
            if (saved_tor != current_tor) return false;
            out_seed = saved_seed;
            out_has_seed = true;
            return true;
        }
        if (version == 3) {
            double et = 0.0;
            uint64_t saved_seed = 0;
            uint8_t saved_tor = 0;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return false;
            if (!f.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed))) return false;
            if (!f.read(reinterpret_cast<char*>(&saved_tor), sizeof(saved_tor))) return false;
            uint8_t current_tor = current_te_family_mode;
            if (saved_tor != current_tor) return false;
            out_seed = saved_seed;
            out_has_seed = true;
            return true;
        }

        // version 2: no seed or torsion information recorded.
        // Safe only when the curve is externally fixed and can be rebuilt identically.
        if (forceCurve || forceSigma) {
            out_seed = forceCurve ? options.curve_seed : 0ULL;
            out_has_seed = forceCurve;
            return true;
        }
        return false;
    };

    if (!options.seed) {
        for (uint64_t c = 0; c < curves; ++c) {
            const std::string ckpt2_file      = "ecm2_te_m_" + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
            const std::string ckpt2_file_o    = ckpt2_file + ".old";
            const std::string ckpt2_legacy    = "ecm2_m_"    + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
            const std::string ckpt2_legacy_o  = ckpt2_legacy + ".old";

            uint64_t s = 0;
            bool has_seed = false;
            if (try_probe_te_ckpt2(ckpt2_file, s, has_seed) ||
                try_probe_te_ckpt2(ckpt2_file_o, s, has_seed) ||
                try_probe_te_ckpt2(ckpt2_legacy, s, has_seed) ||
                try_probe_te_ckpt2(ckpt2_legacy_o, s, has_seed)) {
                resume_curve_idx   = c;
                resume_curve_seed  = s;
                have_resume_seed   = has_seed;
                have_resume_stage2 = true;
                break;
            }
        }

        if (!have_resume_stage2) {
            for (uint64_t c = 0; c < curves; ++c) {
                const std::string ckpt_file   = "ecm_te_m_"  + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
                const std::string ckpt_file_o = ckpt_file + ".old";

                uint64_t s = 0;
                if (try_probe_te_ckpt(ckpt_file, s) || try_probe_te_ckpt(ckpt_file_o, s)) {
                    resume_curve_idx  = c;
                    resume_curve_seed = s;
                    have_resume_seed  = true;
                    break;
                }
            }
        }
    }

    auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();

    uint64_t base_seed;
    if (options.seed) {
        base_seed = options.seed;
    } else if (have_resume_seed) {
        base_seed = resume_curve_seed;
    } else {
        base_seed = (now_ns ^ ((uint64_t)p<<32) ^ B1);
    }

    std::cout << "[ECM] seed=" << base_seed;
    if (!options.seed && (have_resume_seed || have_resume_stage2)) {
        std::cout << " (resumed from " << (have_resume_stage2 ? "Stage2" : "Stage1")
                  << " checkpoint, curve index " << (resume_curve_idx + 1) << ")";
    }
    std::cout << std::endl;

    uint64_t start_curve = 0;
    if (!options.seed && (have_resume_seed || have_resume_stage2)) {
        start_curve = resume_curve_idx;
    }

    for (uint64_t c = start_curve; c < curves; ++c)

    {
        if (p95_stage2_enabled) {
            if (p95_poll_background(false)) {
                return p95_finalize_background_stop(nullptr);
            }
        }
        if (interrupted.load(std::memory_order_relaxed)) {
            if (result_status == "found" && result_factor > 0) {
                return p95_finalize_background_stop(nullptr);
            }
            if (!p95_background_error.empty()) {
                return p95_finalize_background_stop(nullptr);
            }
            std::cout << "[ECM] Interrupt received — exiting without publishing a final result." << std::endl;
            return 0;
        }

        engine* eng = engine::create_gpu(p, static_cast<size_t>(51), static_cast<size_t>(options.device_id), verbose);
        if (!eng) {
            std::cout<<"[ECM] GPU engine unavailable\n";
            result_status = "error";
            write_result();
            return 1;
        }
        if (transform_size_once == 0) {
            transform_size_once = eng->get_size();
            std::ostringstream os;
            os<<"[ECM] Transform size="<<transform_size_once<<" words, device_id="<<options.device_id;
            std::cout<<os.str()<<std::endl;
            if (guiServer_) guiServer_->appendLog(os.str());
        }
        if (transform_size_once != 0 && B2 > B1 && s2_chunk_ends.empty()) {
            const uint32_t max_s2_chunk_bits_plan = compute_s2_chunk_bits(transform_size_once);
            mpz_class EchunkPre(1);
            uint32_t idxPre = 0;
            while (idxPre < (uint32_t)primesS2_v.size()) {
                EchunkPre = 1;
                const uint32_t chunkStartPre = idxPre;
                while (idxPre < (uint32_t)primesS2_v.size()) {
                    mpz_mul_ui(EchunkPre.get_mpz_t(), EchunkPre.get_mpz_t(), (unsigned long)primesS2_v[idxPre]);
                    ++idxPre;
                    if (mpz_sizeinbase(EchunkPre.get_mpz_t(), 2) >= max_s2_chunk_bits_plan && idxPre > chunkStartPre) break;
                }
                const uint32_t cb = (uint32_t)mpz_sizeinbase(EchunkPre.get_mpz_t(), 2);
                total_s2_iters += cb;
                s2_chunk_ends.push_back(idxPre);
                s2_chunk_prefix_iters.push_back(total_s2_iters);
            }
            if (stage2_debug_checks) {
                std::ostringstream oss;
                oss << "[ECM] Stage2 layout: chunk_bits=" << max_s2_chunk_bits_plan
                    << " | chunks=" << s2_chunk_ends.size()
                    << " | total_bits=" << total_s2_iters;
                std::cout << oss.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(oss.str());
            }
        }

        uint32_t s2_idx = 0, s2_cnt = 0; 
        double   s2_et  = 0.0;
        bool     resume_stage2 = false; 
        bool     resume_stage2_in_chunk = false;
        uint32_t resume_s2_chunk_start = 0;
        uint32_t resume_s2_chunk_end = 0;
        uint32_t resume_s2_chunk_bits = 0;
        uint32_t resume_s2_steps_done = 0;
        (void) resume_stage2;

        uint64_t curve_seed;
        if (!options.seed && have_resume_seed && c == resume_curve_idx) {
            curve_seed = resume_curve_seed;
            base_seed  = curve_seed;
        } else {
            curve_seed = mix64(base_seed, c);
            if (forceCurve){
                curve_seed = options.curve_seed;
                base_seed  = curve_seed;
            }
        }

        std::cout << "[ECM] curve_seed=" << curve_seed << std::endl;
        options.curve_seed = curve_seed;
        options.base_seed  = base_seed;
        


        const std::string ckpt_file      = "ecm_te_m_"  + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
        const std::string ckpt2_file     = "ecm2_te_m_" + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
        const std::string ckpt_legacy    = "ecm_m_"     + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
        const std::string ckpt2_legacy   = "ecm2_m_"    + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";

        mpz_class s2_base_Xpos, s2_base_Ypos, s2_base_Tpos;
        mpz_class s2_base_Xneg, s2_base_Yneg, s2_base_Tneg;
        bool have_s2_base_cache = false;

        auto write_mpz_blob = [&](File& f, const mpz_class& z)->bool {
            std::string hx = z.get_str(16);
            uint32_t len = (uint32_t)hx.size();
            if (!f.write(reinterpret_cast<const char*>(&len), sizeof(len))) return false;
            if (len != 0 && !f.write(hx.data(), len)) return false;
            return true;
        };
        auto read_mpz_blob = [&](File& f, mpz_class& z)->bool {
            uint32_t len = 0;
            if (!f.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
            std::string hx(len, '\0');
            if (len != 0 && !f.read(&hx[0], len)) return false;
            if (len == 0) { z = 0; return true; }
            return mpz_set_str(z.get_mpz_t(), hx.c_str(), 16) == 0;
        };
        auto restore_te_stage2_base_from_cache = [&]()->void {
            if (!have_s2_base_cache) return;
            mpz_t zXpos; mpz_init_set(zXpos, s2_base_Xpos.get_mpz_t()); eng->set_mpz((engine::Reg)6,  zXpos); mpz_clear(zXpos);
            mpz_t zYpos; mpz_init_set(zYpos, s2_base_Ypos.get_mpz_t()); eng->set_mpz((engine::Reg)7,  zYpos); mpz_clear(zYpos);
            mpz_t zTpos; mpz_init_set(zTpos, s2_base_Tpos.get_mpz_t()); eng->set_mpz((engine::Reg)9,  zTpos); mpz_clear(zTpos);
            mpz_t zXneg; mpz_init_set(zXneg, s2_base_Xneg.get_mpz_t()); eng->set_mpz((engine::Reg)47, zXneg); mpz_clear(zXneg);
            mpz_t zYneg; mpz_init_set(zYneg, s2_base_Yneg.get_mpz_t()); eng->set_mpz((engine::Reg)48, zYneg); mpz_clear(zYneg);
            mpz_t zTneg; mpz_init_set(zTneg, s2_base_Tneg.get_mpz_t()); eng->set_mpz((engine::Reg)49, zTneg); mpz_clear(zTneg);
            eng->set_multiplicand((engine::Reg)43,(engine::Reg)16);
            eng->set_multiplicand((engine::Reg)44,(engine::Reg)8);
            eng->set_multiplicand((engine::Reg)45,(engine::Reg)29);
            eng->set_multiplicand((engine::Reg)46,(engine::Reg)9);
            eng->set_multiplicand((engine::Reg)50,(engine::Reg)49);
        };

        auto save_ckpt = [&](uint32_t i, double et){
            const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
            {
                File f(newf, "wb");
                int version = 1;
                if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
                if (!f.write(reinterpret_cast<const char*>(&p),       sizeof(p)))       return;
                if (!f.write(reinterpret_cast<const char*>(&i),       sizeof(i)))       return;
                uint32_t nbb = (uint32_t)mpz_sizeinbase(K.get_mpz_t(),2);
                if (!f.write(reinterpret_cast<const char*>(&nbb),     sizeof(nbb)))     return;
                if (!f.write(reinterpret_cast<const char*>(&B1),      sizeof(B1)))      return;
                if (!f.write(reinterpret_cast<const char*>(&et),      sizeof(et)))      return;

                if (!f.write(reinterpret_cast<const char*>(&curve_seed), sizeof(curve_seed))) return;
                uint8_t curve_family_mode = current_te_family_mode;
                if (!f.write(reinterpret_cast<const char*>(&curve_family_mode), sizeof(curve_family_mode))) return;

                const size_t cksz = eng->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!eng->get_checkpoint(data)) return;
                if (!f.write(data.data(), cksz)) return;

                f.write_crc32();
            }
            std::error_code ec;
            fs::remove(ckpt_file + ".old", ec);
            fs::rename(ckpt_file,         ckpt_file + ".old", ec);
            fs::rename(ckpt_file + ".new",ckpt_file,          ec);
            fs::remove(ckpt_file + ".old", ec);
        };

        auto read_ckpt_one = [&](const std::string& file, uint32_t& ri, uint32_t& rnb, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;

            int version = 0;
            if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 1) return -2;

            uint32_t rp = 0;
            if (!f.read(reinterpret_cast<char*>(&rp),  sizeof(rp)))  return -2;
            if (rp != p) return -2;

            if (!f.read(reinterpret_cast<char*>(&ri),  sizeof(ri)))  return -2;
            if (!f.read(reinterpret_cast<char*>(&rnb), sizeof(rnb))) return -2;

            uint64_t rB1 = 0;
            if (!f.read(reinterpret_cast<char*>(&rB1), sizeof(rB1))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et),  sizeof(et)))  return -2;

            uint64_t saved_curve_seed = 0;
            uint8_t  saved_curve_family = 0;
            if (!f.read(reinterpret_cast<char*>(&saved_curve_seed), sizeof(saved_curve_seed))) return -2;
            if (!f.read(reinterpret_cast<char*>(&saved_curve_family),  sizeof(saved_curve_family)))  return -2;
            uint8_t current_curve_family = current_te_family_mode;
            if ((!forceSigma && saved_curve_seed != curve_seed) || saved_curve_family != current_curve_family) return -2;

            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data))  return -2;
            if (!f.check_crc32())            return -2;

            if (rnb != mpz_sizeinbase(K.get_mpz_t(),2) || rB1 != B1) return -2;
            return 0;
        };
        auto read_ckpt = [&](uint32_t& ri, uint32_t& rnb, double& et)->int{
            int rr = read_ckpt_one(ckpt_file, ri, rnb, et);
            if (rr < 0) rr = read_ckpt_one(ckpt_file + ".old", ri, rnb, et);
            if (rr < 0) rr = read_ckpt_one(ckpt_legacy, ri, rnb, et);         
            if (rr < 0) rr = read_ckpt_one(ckpt_legacy + ".old", ri, rnb, et);
            return rr;
        };

        auto save_ckpt2_ex = [&](uint32_t idx, double et, uint32_t cnt_bits,
                                 uint8_t in_chunk,
                                 uint32_t chunk_start_idx,
                                 uint32_t chunk_end_idx,
                                 uint32_t chunk_bits_saved,
                                 uint32_t chunk_steps_done_saved){
            const std::string oldf = ckpt2_file + ".old", newf = ckpt2_file + ".new";
            {
                File f(newf, "wb");
                int version = 6;
                if (!f.write(reinterpret_cast<const char*>(&version),  sizeof(version)))  return;
                if (!f.write(reinterpret_cast<const char*>(&p),        sizeof(p)))        return;
                if (!f.write(reinterpret_cast<const char*>(&idx),      sizeof(idx)))      return;
                if (!f.write(reinterpret_cast<const char*>(&cnt_bits), sizeof(cnt_bits))) return;
                if (!f.write(reinterpret_cast<const char*>(&B1),       sizeof(B1)))       return;
                if (!f.write(reinterpret_cast<const char*>(&B2),       sizeof(B2)))       return;
                uint64_t seed64 = (uint64_t)curve_seed;
                if (!f.write(reinterpret_cast<const char*>(&seed64),   sizeof(seed64)))   return;
                uint8_t curve_family_mode = current_te_family_mode;
                if (!f.write(reinterpret_cast<const char*>(&curve_family_mode), sizeof(curve_family_mode))) return;
                if (!f.write(reinterpret_cast<const char*>(&et),       sizeof(et)))       return;
                if (!f.write(reinterpret_cast<const char*>(&in_chunk), sizeof(in_chunk))) return;
                if (!f.write(reinterpret_cast<const char*>(&chunk_start_idx), sizeof(chunk_start_idx))) return;
                if (!f.write(reinterpret_cast<const char*>(&chunk_end_idx), sizeof(chunk_end_idx))) return;
                if (!f.write(reinterpret_cast<const char*>(&chunk_bits_saved), sizeof(chunk_bits_saved))) return;
                if (!f.write(reinterpret_cast<const char*>(&chunk_steps_done_saved), sizeof(chunk_steps_done_saved))) return;
                uint8_t has_stage2_base = have_s2_base_cache ? 1u : 0u;
                if (!f.write(reinterpret_cast<const char*>(&has_stage2_base), sizeof(has_stage2_base))) return;
                if (has_stage2_base) {
                    if (!write_mpz_blob(f, s2_base_Xpos)) return;
                    if (!write_mpz_blob(f, s2_base_Ypos)) return;
                    if (!write_mpz_blob(f, s2_base_Tpos)) return;
                    if (!write_mpz_blob(f, s2_base_Xneg)) return;
                    if (!write_mpz_blob(f, s2_base_Yneg)) return;
                    if (!write_mpz_blob(f, s2_base_Tneg)) return;
                }

                const size_t cksz = eng->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!eng->get_checkpoint(data)) return;
                if (!f.write(data.data(), cksz)) return;

                f.write_crc32();
            }
            std::error_code ec;
            fs::remove(ckpt2_file + ".old", ec);
            if (fs::exists(ckpt2_file)) fs::rename(ckpt2_file, ckpt2_file + ".old", ec);
            fs::rename(ckpt2_file + ".new", ckpt2_file, ec);
            fs::remove(ckpt2_file + ".new", ec);
        };

        auto save_ckpt2 = [&](uint32_t idx, double et, uint32_t cnt_bits){
            save_ckpt2_ex(idx, et, cnt_bits, 0, 0, 0, 0, 0);
        };

        auto read_ckpt2_one = [&](const std::string& file, uint32_t& idx, uint32_t& cnt_bits, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;

            int version = 0;
            if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 2 && version != 3 && version != 4 && version != 5 && version != 6) return -2;

            uint32_t rp = 0;
            if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != p) return -2;

            if (!f.read(reinterpret_cast<char*>(&idx),      sizeof(idx)))      return -2;
            if (!f.read(reinterpret_cast<char*>(&cnt_bits), sizeof(cnt_bits))) return -2;

            uint64_t b1s = 0, b2s = 0;
            if (!f.read(reinterpret_cast<char*>(&b1s), sizeof(b1s))) return -2;
            if (!f.read(reinterpret_cast<char*>(&b2s), sizeof(b2s))) return -2;

            have_s2_base_cache = false;
            uint64_t saved_seed = 0;
            uint8_t  saved_tor  = 0;
            if (version == 6 || version == 5) {
                if (!f.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed))) return -2;
                if (!f.read(reinterpret_cast<char*>(&saved_tor),  sizeof(saved_tor)))  return -2;
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
                uint8_t in_chunk = 0;
                if (!f.read(reinterpret_cast<char*>(&in_chunk), sizeof(in_chunk))) return -2;
                if (!f.read(reinterpret_cast<char*>(&resume_s2_chunk_start), sizeof(resume_s2_chunk_start))) return -2;
                if (!f.read(reinterpret_cast<char*>(&resume_s2_chunk_end), sizeof(resume_s2_chunk_end))) return -2;
                if (!f.read(reinterpret_cast<char*>(&resume_s2_chunk_bits), sizeof(resume_s2_chunk_bits))) return -2;
                if (!f.read(reinterpret_cast<char*>(&resume_s2_steps_done), sizeof(resume_s2_steps_done))) return -2;
                resume_stage2_in_chunk = (in_chunk != 0);
                if (version == 6) {
                    uint8_t has_stage2_base = 0;
                    if (!f.read(reinterpret_cast<char*>(&has_stage2_base), sizeof(has_stage2_base))) return -2;
                    if (has_stage2_base) {
                        if (!read_mpz_blob(f, s2_base_Xpos)) return -2;
                        if (!read_mpz_blob(f, s2_base_Ypos)) return -2;
                        if (!read_mpz_blob(f, s2_base_Tpos)) return -2;
                        if (!read_mpz_blob(f, s2_base_Xneg)) return -2;
                        if (!read_mpz_blob(f, s2_base_Yneg)) return -2;
                        if (!read_mpz_blob(f, s2_base_Tneg)) return -2;
                        have_s2_base_cache = true;
                    }
                } else if (resume_stage2_in_chunk) {
                    return -2;
                }
            } else if (version == 4) {
                if (!f.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed))) return -2;
                if (!f.read(reinterpret_cast<char*>(&saved_tor),  sizeof(saved_tor)))  return -2;
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            } else if (version == 3) {
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
                if (!f.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed))) return -2;
                if (!f.read(reinterpret_cast<char*>(&saved_tor),  sizeof(saved_tor)))  return -2;
            } else {
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            }

            if (version >= 3) {
                uint8_t current_tor = current_te_family_mode;
                if ((!forceSigma && saved_seed != (uint64_t)curve_seed) || saved_tor != current_tor) return -2;
            }

            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32())           return -2;

            if (b1s != B1 || b2s != B2) return -2;
            return 0;
        };
        auto read_ckpt2 = [&](uint32_t& idx, uint32_t& cnt_bits, double& et)->int{
            resume_stage2_in_chunk = false;
            resume_s2_chunk_start = 0;
            resume_s2_chunk_end = 0;
            resume_s2_chunk_bits = 0;
            resume_s2_steps_done = 0;
            int rr = read_ckpt2_one(ckpt2_file, idx, cnt_bits, et);
            if (rr < 0) rr = read_ckpt2_one(ckpt2_file + ".old", idx, cnt_bits, et);
            if (rr < 0) rr = read_ckpt2_one(ckpt2_legacy, idx, cnt_bits, et);
            if (rr < 0) rr = read_ckpt2_one(ckpt2_legacy + ".old", idx, cnt_bits, et);
            return rr;
        };

        auto addm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
            mpz_class r=a+b;
            mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
            if (r<0) r+=N;
            return r;
        };
        auto subm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
            mpz_class r=a-b;
            mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
            if (r<0) r+=N;
            return r;
        };
        auto mulm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
            mpz_class r;
            mpz_mul(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
            mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
            if (r<0) r+=N;
            return r;
        };
        auto sqrm = [&](const mpz_class& a)->mpz_class{
            mpz_class r;
            mpz_mul(r.get_mpz_t(), a.get_mpz_t(), a.get_mpz_t());
            mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
            if (r<0) r+=N;
            return r;
        };
        auto invm = [&](const mpz_class& a, mpz_class& inv)->int{
            if (mpz_sgn(a.get_mpz_t())==0) return -1;
            if (mpz_invert(inv.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t())) return 0;
            mpz_class g;
            mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t());
            if (g > 1 && g < N) {
                result_factor = g;
                result_status = "found";
                return 1;
            }
            return -1;
        };
        
       // auto sqrQ = [](const mpq_class& z)->mpq_class { return z*z; };
        /*auto pow4Q = [&](const mpq_class& z)->mpq_class { mpq_class t=sqrQ(z); return t*t; };

        auto mpq_to_mod = [&](const mpq_class& q, mpz_class& out)->int{
            mpz_class num(q.get_num()), den(q.get_den());
            mpz_class inv;
            int r = invm(den, inv);
            if (r==1) return 1;    
            if (r<0)  return -1;   
            out = mulm(num, inv);
            return 0;
        };*/

        struct QPt { mpq_class x, y; bool inf=false; };
        /*auto q_add = [&](const QPt& P, const QPt& Q)->QPt{
            if (P.inf) {return Q;} if (Q.inf) {return P;}
            if (P.x==Q.x && P.y==-Q.y) return QPt{{}, {}, true};
            mpq_class lambda;
            if (P.x==Q.x && P.y==Q.y) {
                lambda = (3*sqrQ(P.x) + mpq_class(4)) / (mpq_class(2)*P.y);
            } else {
                lambda = (Q.y - P.y) / (Q.x - P.x);
            }
            mpq_class x3 = sqrQ(lambda) - P.x - Q.x;
            mpq_class y3 = lambda*(P.x - x3) - P.y;
            return QPt{x3,y3,false};
        };*/
        /*auto q_mul = [&](QPt P, uint32_t k)->QPt{
            QPt R{{}, {}, true};
            while (k){
                if (k&1u) R = q_add(R,P);
                P = q_add(P,P);
                k >>= 1u;
            }
            return R;
        };*/

        mpz_class aE, dE, X0, Y0;
        mpz_class sigma_resume;
        bool have_sigma_resume = false;
        
        string torsion_used = te_use_torsion16 ? string("16") : (te_use_family_iv_163 ? string("family_iv_163") : string("none"));
                             // (options.torsion16 ? string("16") : string("8"));
        bool built = false;

        // In-memory checkpoint of the last known good state for fast recovery on errors
        std::vector<char> last_good_state;
        bool have_last_good_state = false;
        uint32_t last_good_iter = 0;
        uint32_t current_iter_for_invariant = 0;
        const size_t ctx_ckpt_size = eng->get_checkpoint_size();

        auto force_on_curve = [&](mpz_class& aE_ref,
                                  mpz_class& dE_ref,
                                  const mpz_class& X,
                                  const mpz_class& Y)->bool
        {
            mpz_class X2 = sqrm(X);
            mpz_class Y2 = sqrm(Y);
            mpz_class XY2 = mulm(X2, Y2);
            mpz_class invXY2;
            int r = invm(XY2, invXY2);
            if (r==1) return false;
            if (r<0) return false;
            dE_ref = mulm(subm(addm(mulm(aE_ref, X2), Y2), mpz_class(1)), invXY2);
            return true;
        };
        auto sqrQ = [](const mpq_class& z)->mpq_class { return z * z; };
        auto pow4Q = [&](const mpq_class& z)->mpq_class { mpq_class t = sqrQ(z); return t * t; };
        auto mpq_to_mod = [&](const mpq_class& q, mpz_class& out)->int{
            mpz_class num(q.get_num()), den(q.get_den());
            mpz_class inv;
            int r = invm(den, inv);
            if (r==1) return 1;
            if (r<0)  return -1;
            out = mulm(num, inv);
            return 0;
        };

        //struct QPt { mpq_class x, y; bool inf=false; };
        auto q_add_family_iv = [&](const QPt& P, const QPt& Q)->QPt{
            if (P.inf) return Q;
            if (Q.inf) return P;
            if (P.x == Q.x) {
                if (P.y == -Q.y) return QPt{{}, {}, true};
                if (P.y == 0) return QPt{{}, {}, true};
            }
            mpq_class lambda;
            if (P.x == Q.x && P.y == Q.y) {
                lambda = (mpq_class(3) * sqrQ(P.x) - mpq_class(2) * P.x - mpq_class(9)) / (mpq_class(2) * P.y);
            } else {
                lambda = (Q.y - P.y) / (Q.x - P.x);
            }
            mpq_class x3 = sqrQ(lambda) + mpq_class(1) - P.x - Q.x;
            mpq_class y3 = -P.y - lambda * (x3 - P.x);
            return QPt{x3, y3, false};
        };
        auto q_mul_family_iv = [&](QPt P, uint32_t k)->QPt{
            QPt R{{}, {}, true};
            while (k) {
                if (k & 1u) R = q_add_family_iv(R, P);
                P = q_add_family_iv(P, P);
                k >>= 1u;
            }
            return R;
        };
        // ---- torsion 16 (Gallot / Theorem 2.5) : a = 1 ----
        if (te_use_torsion16) {
            bool ok = false;
            using clock = std::chrono::steady_clock;
            auto t0_t16 = clock::now();
            auto tlast_t16 = t0_t16;
            double ema_tps_t16 = 0.0;
            constexpr uint32_t torsion16_init_tries = 128;
            for (uint32_t tries = 0; tries < torsion16_init_tries && !ok; ++tries) {
                uint64_t m = (mix64(curve_seed, 0x544F523136ULL + uint64_t(tries)) | 1ULL);
                if (m < 3) m += 2;

                {
                    auto now = clock::now();
                    const uint32_t done = tries;
                    const double elapsed = std::chrono::duration<double>(now - t0_t16).count();
                    const double dt = std::chrono::duration<double>(now - tlast_t16).count();
                    const double inst = (dt > 1e-9 && tries > 0) ? (1.0 / dt) : 0.0;
                    if (inst > 0.0) {
                        if (ema_tps_t16 <= 0.0) ema_tps_t16 = inst;
                        else ema_tps_t16 = 0.75 * ema_tps_t16 + 0.25 * inst;
                    }
                    const double rate = (ema_tps_t16 > 0.0) ? ema_tps_t16 : ((elapsed > 1e-9 && done > 0) ? (double)done / elapsed : 0.0);
                    const double eta = (rate > 0.0) ? ((double)(torsion16_init_tries - done) / rate) : 0.0;
                    std::ostringstream line;
                    line << "[ECM] torsion16 init " << done << "/" << torsion16_init_tries
                         << " (" << std::fixed << std::setprecision(1) << (100.0 * (double)done / (double)torsion16_init_tries) << "%)"
                         << " | tries/s " << std::fixed << std::setprecision(1) << rate
                         << " | ETA " << fmt_hms(eta)
                         << " | m=" << m;
                    ecm_print_progress_line(line.str());
                    tlast_t16 = now;
                }

                mpz_class s, t;
                ecm_local::EC_mod4::get(m, 4, 8, N, s, t);

                aE = mpz_class(1);

                auto inv_or_gcd = [&](const mpz_class& den, mpz_class& inv)->int {
                    int rr = invm(den, inv);
                    if (rr == 1) return 1;
                    if (rr == 0) return 0;
                    return -1;
                };

                mpz_class inv, alpha, alpha2, r;
                mpz_class den = subm(s, mpz_class(4));
                int inv_status = inv_or_gcd(den, inv);
                if (inv_status == 1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (inv_status < 0) continue;
                alpha  = mulm(addm(t, mpz_class(8)), inv);
                alpha2 = sqrm(alpha);

                den = subm(mpz_class(8), alpha2);
                inv_status = inv_or_gcd(den, inv);
                if (inv_status == 1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (inv_status < 0) continue;
                r = mulm(addm(mpz_class(8), mulm(mpz_class(2), alpha)), inv);

                mpz_class two_r_minus1 = subm(mulm(mpz_class(2), r), mpz_class(1));
                mpz_class t1 = sqrm(two_r_minus1);

                mpz_class numD = addm(subm(mulm(mpz_class(8), sqrm(r)), mulm(mpz_class(8), r)), mpz_class(1));
                mpz_class t1sq = sqrm(t1);
                inv_status = inv_or_gcd(t1sq, inv);
                if (inv_status == 1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (inv_status < 0) continue;
                dE = mulm(numD, inv);
                if (dE == 0 || dE == 1 || dE == subm(N, mpz_class(1))) continue;

                mpz_class numX = mulm(subm(mpz_class(8), alpha2), subm(mulm(mpz_class(2), sqrm(r)), mpz_class(1)));
                mpz_class denX = addm(subm(mulm(mpz_class(2), s), alpha2), mpz_class(4));
                inv_status = inv_or_gcd(denX, inv);
                if (inv_status == 1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (inv_status < 0) continue;
                X0 = mulm(numX, inv);

                mpz_class denY = subm(mulm(mpz_class(4), r), mpz_class(3));
                inv_status = inv_or_gcd(denY, inv);
                if (inv_status == 1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (inv_status < 0) continue;
                Y0 = mulm(t1, inv);

                if (X0 == 0 || Y0 == 0) continue;

                auto X2 = sqrm(X0), Y2 = sqrm(Y0);
                auto L  = addm(X2, Y2);
                auto R  = addm(mpz_class(1), mulm(dE, mulm(X2, Y2)));
                if (subm(L, R) != 0) continue;

                built = ok = true;
                torsion_used = "16";
            }
            std::cout << std::endl;

        }

        if (!built && te_use_family_iv_163) {
            bool family_built = false;
            for (uint32_t tries = 0; tries < 128 && !family_built; ++tries) {
                uint32_t m = (uint32_t)(1 + (mix64(curve_seed, 0x163163ULL + uint64_t(tries)) % 100ULL));
                QPt Q0{mpq_class(5), mpq_class(8), false};
                QPt Pq = q_mul_family_iv(Q0, m);
                if (Pq.inf) continue;
                if (Pq.y == mpq_class(4)) continue;

                mpq_class tq = (mpq_class(4) * Pq.x + mpq_class(4)) / (Pq.y - mpq_class(4));
                mpq_class t2q = sqrQ(tq);
                if (t2q == mpq_class(4)) continue;
                mpq_class eq = (t2q + mpq_class(4) * tq) / (t2q - mpq_class(4));
                if (eq == 0) continue;

                mpq_class t4q = pow4Q(tq);
                mpq_class t6q = t4q * t2q;
                mpq_class denXq = t4q + mpq_class(6) * tq * t2q + mpq_class(12) * t2q + mpq_class(16) * tq;
                if (denXq == 0) continue;
                mpq_class Xq = (mpq_class(2) * tq * t2q + mpq_class(2) * t2q - mpq_class(8) * tq - mpq_class(8)) / denXq;
                mpq_class denYq = t6q + mpq_class(6) * t4q * tq + mpq_class(10) * t4q + mpq_class(16) * tq * t2q + mpq_class(48) * t2q + mpq_class(64) * tq;
                if (denYq == 0) continue;
                mpq_class Yq = (t6q + mpq_class(6) * t4q * tq + mpq_class(10) * t4q - mpq_class(16) * tq * t2q - mpq_class(48) * t2q - mpq_class(32) * tq - mpq_class(32)) / denYq;
                mpq_class dQ = -pow4Q(eq);

                aE = subm(N, mpz_class(1));
                int r = mpq_to_mod(dQ, dE);
                if (r == 1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (r < 0) continue;
                r = mpq_to_mod(Xq, X0);
                if (r == 1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (r < 0) continue;
                r = mpq_to_mod(Yq, Y0);
                if (r == 1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (r < 0) continue;

                if (dE == 0 || dE == 1 || dE == subm(N, mpz_class(1))) continue;
                if (X0 == 0 || Y0 == 0) continue;

                auto X2 = sqrm(X0), Y2 = sqrm(Y0);
                auto L  = addm(mulm(aE, X2), Y2);
                auto R  = addm(mpz_class(1), mulm(dE, mulm(X2, Y2)));
                if (subm(L, R) != 0) continue;

                built = true;
                family_built = true;
                torsion_used = "family_iv_163";
            }
        }

        if (!built && te_use_family_iv_163) {
            std::cout << "[ECM] Could not build a family_iv_163 Twisted Edwards curve for this seed, retrying curve\n";
            delete eng;
            continue;
        }

        if (!built && !te_use_torsion16 && !te_use_family_iv_163) {
            bool sigma_built = false;
            for (uint32_t tries = 0; tries < 256 && !sigma_built; ++tries) {
                mpz_class sigma_mpz;
                if (forceSigma) {
                    if (tries > 0) break;
                    if (mpz_set_str(sigma_mpz.get_mpz_t(), options.sigma.c_str(), 0) != 0) {
                        std::cerr << "[ECM] Invalid -sigma value: " << options.sigma << std::endl;
                        delete eng;
                        return 0;
                    }
                } else {
                    const uint64_t sigma_seed = (tries == 0) ? curve_seed : mix64(curve_seed, 0x5EED5EEDULL + uint64_t(tries));
                    sigma_mpz = rnd_mpz_bits(N, sigma_seed, 64);
                }

                sigma_mpz %= N;
                if (sigma_mpz < 0) sigma_mpz += N;
                if (sigma_mpz == 0) sigma_mpz = 1;

                auto inv_or_finish = [&](const mpz_class& den, mpz_class& inv)->int {
                    int r = invm(den, inv);
                    if (r == 1) {
                        curves_tested_for_found = c+1;
                        options.curves_tested_for_found = (uint32_t)(c+1);
                        write_result();
                        publish_json();
                        delete eng;
                        return 1;
                    }
                    return r;
                };

                mpz_class sigma2 = sqrm(sigma_mpz);
                mpz_class su = subm(sigma2, mpz_class(5));
                mpz_class sv = mulm(mpz_class(4), sigma_mpz);

                mpz_class inv_sv;
                {
                    int r = inv_or_finish(sv, inv_sv);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }

                mpz_class denA = mulm(mpz_class(4), mulm(mulm(sqrm(su), su), sv));
                mpz_class inv_denA;
                {
                    int r = inv_or_finish(denA, inv_denA);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }

                mpz_class tnum = mulm(sqrm(subm(sv, su)), subm(sv, su));
                mpz_class Aplus2 = mulm(mulm(tnum, addm(mulm(mpz_class(3), su), sv)), inv_denA);
                mpz_class Bmont = mulm(su, inv_sv);
                mpz_class invB;
                {
                    int r = inv_or_finish(Bmont, invB);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }

                aE = mulm(Aplus2, invB);
                dE = mulm(subm(Aplus2, mpz_class(4)), invB);

                mpz_class sp = addm(sigma2, mpz_class(5));
                mpz_class denX = mulm(mulm(subm(su, sv), addm(su, sv)), sp);
                mpz_class inv_denX;
                {
                    int r = inv_or_finish(denX, inv_denX);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }
                X0 = mulm(mulm(sqrm(su), sv), inv_denX);

                mpz_class su3 = mulm(sqrm(su), su);
                mpz_class sv3 = mulm(sqrm(sv), sv);
                mpz_class denY = addm(su3, sv3);
                mpz_class inv_denY;
                {
                    int r = inv_or_finish(denY, inv_denY);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }
                Y0 = mulm(subm(su3, sv3), inv_denY);

                if (X0 == 0 || Y0 == 0) {
                    if (forceSigma) {
                        std::cerr << "[ECM] Invalid/singular -sigma generated a degenerate point" << std::endl;
                        delete eng;
                        return 0;
                    }
                    continue;
                }

                auto X2 = sqrm(X0), Y2 = sqrm(Y0);
                auto L  = addm(mulm(aE, X2), Y2);
                auto R  = addm(mpz_class(1), mulm(dE, mulm(X2, Y2)));
                if (subm(L, R) != 0) {
                    if (forceSigma) {
                        std::cerr << "[ECM] Internal curve conversion failure for -sigma" << std::endl;
                        delete eng;
                        return 0;
                    }
                    continue;
                }

                options.sigma_hex = sigma_mpz.get_str(16);
                sigma_resume = sigma_mpz;
                have_sigma_resume = true;
                built = true;
                sigma_built = true;
                torsion_used = "none";
            }
        }

        if (!built && !te_use_torsion16) {
            std::cout << "[ECM] Could not build a SIGMA-based Twisted Edwards curve for this seed, retrying curve\n";
            delete eng;
            continue;
        }

        if(!built){
            aE = subm(N, mpz_class(2));
            for (int tries=0; tries<256 && !built; ++tries){
                X0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xC0FFEEull) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x9E37u)), 256);
                Y0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xFACEFEEDull) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x94D0u)), 256);
                if (X0==0 || Y0==0) continue;
                if (!force_on_curve(aE, dE, X0, Y0)) {
                    if (result_factor>0){
                        curves_tested_for_found = c+1;
                        options.curves_tested_for_found = (uint32_t)(c+1);
                        write_result();
                        publish_json();
                        delete eng;
                        return 0;
                    }
                    continue;
                }
                built = true;
            }
        }
        {
            std::ostringstream head;
            head << "[ECM] Curve " << (c+1) << "/" << curves
                 << " | twisted_edwards"
                 << " | torsion=" << torsion_used
                 << " | K_bits=" << mpz_sizeinbase(K.get_mpz_t(),2)
                 << " | seed=" << curve_seed;
            if (have_sigma_resume) head << " | sigma";
            std::cout << head.str() << std::endl;
            if (guiServer_) guiServer_->appendLog(head.str());
        }
        
        auto check_invariant = [&](){
            auto Xv = compute_X_with_dots(eng,(engine::Reg)3,N);
            auto Yv = compute_X_with_dots(eng,(engine::Reg)4,N);
            auto Zv = compute_X_with_dots(eng,(engine::Reg)1,N);
            auto Tv = compute_X_with_dots(eng,(engine::Reg)5,N);
            auto lhs = addm(mulm(aE, sqrm(Xv)), sqrm(Yv));
            auto rhs = addm(sqrm(Zv), mulm(dE, sqrm(Tv)));
            auto rel = subm(lhs, rhs);
            if (rel != 0){
                std::cout << "[ECM] invariant FAIL " //(a="
                          /*<< aE*/ << ")\n";
                return false;
            }
            else{
                std::cout << "[ECM] check invariant OK " // (a="
                          /*<< aE*/ << ")\n";
                // Save an in-memory checkpoint of this good state
                if (ctx_ckpt_size > 0) {
                    last_good_state.resize(ctx_ckpt_size);
                    if (eng->get_checkpoint(last_good_state)) {
                        have_last_good_state = true;
                        last_good_iter = current_iter_for_invariant;
                    }
                }
                return true;
            }
        };

        torsion_last = torsion_used;
        torsion_name = torsion_used;

        std::cout<<"[ECM] torsion="<<torsion_used<<std::endl;

        if (!built){
            delete eng;
            continue;
        }

        {
            mpz_class X2 = sqrm(X0);
            mpz_class Y2 = sqrm(Y0);
            mpz_class L = addm(mulm(aE, X2), Y2);
            mpz_class R = addm(mpz_class(1), mulm(dE, mulm(X2, Y2)));
            std::cout<<"[ECM] check_on_curve="<<(subm(L,R)==0? "OK":"FAIL")<<std::endl;
            std::cout<<"[ECM] Compute in Twisted Edwards Mode"<<std::endl;
            if (subm(L,R)!=0){
                delete eng;
                continue;
            }
        }

        {
            std::ostringstream head;
            head<<"[ECM] Curve "<<(c+1)<<"/"<<curves
                <<" | twisted_edwards | torsion="<<torsion_used
                <<" | K_bits="<<Kbits<<
                "| curve="<<curve_seed;
            std::cout<<head.str()<<std::endl;
            if (guiServer_) guiServer_->appendLog(head.str());
        }

        {
            mpz_class dummyA24(0);
            mpz_class x0_ref = X0;
        }


        auto hadamard = [&](size_t a, size_t b, size_t s, size_t d){
            eng->addsub((engine::Reg)s, (engine::Reg)d, (engine::Reg)a, (engine::Reg)b); // s=a+b, d=a-b
        };
        /*auto hadamard_copy = [&](size_t a, size_t b, size_t s, size_t d, size_t s_copy, size_t d_copy){
            eng->addsub_copy((engine::Reg)s,(engine::Reg)d,(engine::Reg)s_copy,(engine::Reg)d_copy,
                            (engine::Reg)a,(engine::Reg)b);
        };*/


        // Inputs/outputs mapping :
        // X1=R3, Y1=R4, Z1=R1, T1=R5
        // X2=R6, Y2=R7, Z2=1 (affine), T2=R9
        // constants: a=R16, d=R29

        // eADD_RP: Twisted Edwards extended coordinates (X,Y,Z,T)
        // Formulas:
        // E = X1*Y2 + Y1*X2
        // H = Y1*Y2 - a*X1*X2
        // C = d*T1*T2
        // D = Z1*Z2 (here Z2 = 1 so D = Z1)
        // X3 = E*(D - C)
        // Y3 = H*(D + C)
        // T3 = E*H
        // Z3 = (D - C)*(D + C)
        auto eADD_RP = [&](){
            // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
            eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3); // 34=S1, 35=D1
            eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)7,(engine::Reg)6); // 36=S2, 37=D2

            // 30 = X1*X2
            eng->copy((engine::Reg)30,(engine::Reg)3);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)6);   // multiplicand <- X2
            eng->mul_copy((engine::Reg)30,(engine::Reg)11, (engine::Reg)39);               // 30 = X1*X2

            // 31 = Y1*Y2
            eng->copy((engine::Reg)31,(engine::Reg)4);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);   // multiplicand <- Y2
            eng->mul((engine::Reg)31,(engine::Reg)11);               // 31 = Y1*Y2

            // 32 = C = d*T1*T2
            eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)9);   // multiplicand <- T2
            eng->mul((engine::Reg)32,(engine::Reg)46);               // 32 = T1*T2
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)29);  // multiplicand <- d
            eng->mul((engine::Reg)32,(engine::Reg)45);               // 32 = d*T1*T2 = C

            // 42 = D + C, 41 = D - C  (D = Z1)
            hadamard((engine::Reg)1,(engine::Reg)32, (engine::Reg)42,(engine::Reg)41);

            // 38 = E = S1*S2 - X1X2 - Y1Y2
            eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
            eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)30);           // 38 -= X1*X2
            eng->sub_reg((engine::Reg)38,(engine::Reg)31);           // 38 -= Y1*Y2  => E

            // 39 = a*X1*X2 ; 40 = H = Y1*Y2 - a*X1*X2
            //eng->copy((engine::Reg)39,(engine::Reg)30);              // 39 <- X1*X2
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)16);  // multiplicand <- a   
            eng->mul((engine::Reg)39,(engine::Reg)43);               // 39 = a*X1*X2
            eng->copy((engine::Reg)40,(engine::Reg)31);              // 40 <- Y1*Y2
            eng->sub_reg((engine::Reg)40,(engine::Reg)39);           // 40 = H

            // X3 = E*(D - C)
            eng->copy((engine::Reg)3,(engine::Reg)38);               // X3 <- E
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);  // multiplicand <- (D - C)
            eng->mul((engine::Reg)3,(engine::Reg)11);                // X3 = E*(D - C)

            // Z3 = (D - C)*(D + C)
            eng->copy((engine::Reg)1,(engine::Reg)42);               // Z3 <- (D + C)
            eng->mul((engine::Reg)1,(engine::Reg)11);                // Z3 = (D + C)*(D - C)

            // Y3 = H*(D + C)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);  // multiplicand <- H
            eng->copy((engine::Reg)4,(engine::Reg)42);               // Y3 <- (D + C)
            eng->mul((engine::Reg)4,(engine::Reg)11);                // Y3 = (D + C)*H

            // T3 = E*H
            eng->copy((engine::Reg)5,(engine::Reg)38);               // T3 <- E
            eng->mul((engine::Reg)5,(engine::Reg)11);                // T3 = E*H
        };


        
//Swap 6,7,9 by 47,48,49

        // Inputs/outputs mapping :
        // X1=R3, Y1=R4, Z1=R1, T1=R5
        // X2=R6, Y2=R7, Z2=1 (affine), T2=R9
        // constants: a=R16, d=R29

        // eADD_RP: Twisted Edwards extended coordinates (X,Y,Z,T)
        // Formulas:
        // E = X1*Y2 + Y1*X2
        // H = Y1*Y2 - a*X1*X2
        // C = d*T1*T2
        // D = Z1*Z2 (here Z2 = 1 so D = Z1)
        // X3 = E*(D - C)
        // Y3 = H*(D + C)
        // T3 = E*H
        // Z3 = (D - C)*(D + C)
        auto eADD_RP_2 = [&](){
            // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
            eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3); // 34=S1, 35=D1
            eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)48,(engine::Reg)47); // 36=S2, 37=D2

            // 30 = X1*X2
            eng->copy((engine::Reg)30,(engine::Reg)3);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)47);   // multiplicand <- X2
            eng->mul_copy((engine::Reg)30,(engine::Reg)11, (engine::Reg)39);               // 30 = X1*X2

            // 31 = Y1*Y2
            eng->copy((engine::Reg)31,(engine::Reg)4);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)48);   // multiplicand <- Y2
            eng->mul((engine::Reg)31,(engine::Reg)11);               // 31 = Y1*Y2

            // 32 = C = d*T1*T2
            eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)49);  // multiplicand <- T2
            eng->mul((engine::Reg)32,(engine::Reg)50);               // 32 = T1*T2
            eng->mul((engine::Reg)32,(engine::Reg)45);               // 32 = d*T1*T2 = C

            // 42 = D + C, 41 = D - C  (D = Z1)
            hadamard((engine::Reg)1,(engine::Reg)32, (engine::Reg)42,(engine::Reg)41);

            // 38 = E = S1*S2 - X1X2 - Y1Y2
            eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
            eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)30);           // 38 -= X1*X2
            eng->sub_reg((engine::Reg)38,(engine::Reg)31);           // 38 -= Y1*Y2  => E

            // 39 = a*X1*X2 ; 40 = H = Y1*Y2 - a*X1*X2
            //eng->copy((engine::Reg)39,(engine::Reg)30);              // 39 <- X1*X2
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)16);  // multiplicand <- a   
            eng->mul((engine::Reg)39,(engine::Reg)43);               // 39 = a*X1*X2
            eng->copy((engine::Reg)40,(engine::Reg)31);              // 40 <- Y1*Y2
            eng->sub_reg((engine::Reg)40,(engine::Reg)39);           // 40 = H

            // X3 = E*(D - C)
            eng->copy((engine::Reg)3,(engine::Reg)38);               // X3 <- E
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);  // multiplicand <- (D - C)
            eng->mul((engine::Reg)3,(engine::Reg)11);                // X3 = E*(D - C)

            // Z3 = (D - C)*(D + C)
            eng->copy((engine::Reg)1,(engine::Reg)42);               // Z3 <- (D + C)
            eng->mul((engine::Reg)1,(engine::Reg)11);                // Z3 = (D + C)*(D - C)

            // Y3 = H*(D + C)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);  // multiplicand <- H
            eng->copy((engine::Reg)4,(engine::Reg)42);               // Y3 <- (D + C)
            eng->mul((engine::Reg)4,(engine::Reg)11);                // Y3 = (D + C)*H

            // T3 = E*H
            eng->copy((engine::Reg)5,(engine::Reg)38);               // T3 <- E
            eng->mul((engine::Reg)5,(engine::Reg)11);                // T3 = E*H
        };



        // eDBL_XYTZ: Twisted Edwards doubling (X,Y,Z,T) with general a (a in R43)
        // Formulas:
        // A = X^2
        // B = Y^2
        // C = 2*Z^2
        // D = a*A
        // E = 2*T*Z
        // G = D + B
        // H = D - B
        // F = G - C
        // X3 = E*F
        // Y3 = G*H
        // T3 = E*H
        // Z3 = F*G
        auto eDBL_XYTZ = [&](size_t RX,size_t RY,size_t RZ,size_t RT){
            // E = 2*T*Z  (compute first while Z is intact)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)RZ); // 11 = Z
            eng->mul((engine::Reg)RT,(engine::Reg)11);              // RT = T*Z
            eng->add((engine::Reg)RT,(engine::Reg)RT);              // RT = 2*T*Z = E

            // C = 2*Z^2 (no const-mul: square then add)
            eng->square_mul((engine::Reg)RZ);                       // RZ = Z^2
            eng->add((engine::Reg)RZ,(engine::Reg)RZ);              // RZ = 2*Z^2 = C

            // A = X^2 ; B = Y^2  (in-place)
            eng->square_mul((engine::Reg)RX);                       // RX = A
            eng->square_mul((engine::Reg)RY);                       // RY = B

            // D = a*A
            eng->mul((engine::Reg)RX,(engine::Reg)43);              // RX = D = a*A

            // G = D + B ; H = D - B  (Hadamard on RX=D and RY=B)
            hadamard((engine::Reg)RX,(engine::Reg)RY,
                    (engine::Reg)23,(engine::Reg)25);              // 23=G, 25=H

            // F = G - C
            eng->copy((engine::Reg)24,(engine::Reg)23);             // 24 = G
            eng->sub_reg((engine::Reg)24,(engine::Reg)RZ);          // 24 = F = G - C

            // X3 = E*F ; Z3 = F*G   (first and only set_multiplicand for F)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)24); // 11 = F
            eng->copy((engine::Reg)RX,(engine::Reg)RT);             // RX <- E
            eng->mul((engine::Reg)RX,(engine::Reg)11);              // X3 = E*F
            eng->copy((engine::Reg)RZ,(engine::Reg)23);             // RZ <- G
            eng->mul((engine::Reg)RZ,(engine::Reg)11);              // Z3 = G*F

            // Y3 = G*H ; T3 = E*H   (second and last set_multiplicand for H)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)25); // 11 = H
            eng->copy((engine::Reg)RY,(engine::Reg)23);             // RY <- G
            eng->mul((engine::Reg)RY,(engine::Reg)11);              // Y3 = G*H
            eng->mul((engine::Reg)RT,(engine::Reg)11);              // T3 = E*H
        };


        // eDBL_XYTZ_notwist (a = 1)
        // A = X^2, B = Y^2, C = 2*Z^2, E = 2*T*Z
        // G = A + B, H = A - B, F = G - C
        // X3 = E*F, Z3 = F*G, Y3 = G*H, T3 = E*H
        auto eDBL_XYTZ_notwist = [&](size_t RX,size_t RY,size_t RZ,size_t RT){
            // --- E = 2*T*Z  (avant d’écraser Z)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)RZ); // 11 = Z
            eng->mul((engine::Reg)RT,(engine::Reg)11);              // RT = T*Z
            eng->add((engine::Reg)RT,(engine::Reg)RT);              // RT = 2*T*Z = E

            // --- C = 2*Z^2
            eng->square_mul((engine::Reg)RZ);                       // RZ = Z^2
            eng->add((engine::Reg)RZ,(engine::Reg)RZ);              // RZ = 2*Z^2 = C

            // --- A = X^2, B = Y^2  (in-place)
            eng->square_mul((engine::Reg)RX);                       // RX = A
            eng->square_mul((engine::Reg)RY);                       // RY = B

            // --- G = A + B ; H = A - B  + copie de G en 24 (économise un copy)
            // 23=G, 25=H, 24=G (copie), 22=d_copy (scratch, ignoré)
            eng->addsub_copy((engine::Reg)23,(engine::Reg)25,(engine::Reg)24,(engine::Reg)22,
                            (engine::Reg)RX,(engine::Reg)RY);

            // --- F = G - C  (F en 24, G reste en 23)
            eng->sub_reg((engine::Reg)24,(engine::Reg)RZ);          // 24 = F = G - C

            // --- X3 = E*F ; Z3 = F*G
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)24); // 11 = F
            eng->copy((engine::Reg)RX,(engine::Reg)RT);             // RX <- E
            eng->mul((engine::Reg)RX,(engine::Reg)11);              // X3 = E*F
            eng->copy((engine::Reg)RZ,(engine::Reg)23);             // RZ <- G
            eng->mul((engine::Reg)RZ,(engine::Reg)11);              // Z3 = G*F

            // --- Y3 = G*H ; T3 = E*H
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)25); // 11 = H
            eng->copy((engine::Reg)RY,(engine::Reg)23);             // RY <- G
            eng->mul((engine::Reg)RY,(engine::Reg)11);              // Y3 = G*H
            eng->mul((engine::Reg)RT,(engine::Reg)11);              // T3 = E*H
        };

        // eADD_RP_notwist : Twisted Edwards addition (a = 1, Z2 = 1)
        // Formulas:
        // E = X1*Y2 + Y1*X2
        // H = Y1*Y2 - a*X1*X2 (ici a=1)
        // C = d*T1*T2
        // D = Z1
        // X3 = E*(D - C)
        // Y3 = H*(D + C)
        // T3 = E*H
        // Z3 = (D - C)*(D + C)
        auto eADD_RP_notwist = [&](){
            // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
            eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3); // 34=S1=Y1+X1, 35=D1=Y1-X1
            eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)7,(engine::Reg)6); // 36=S2=Y2+X2, 37=D2=Y2-X2

            // 30 = X1*X2
            eng->copy((engine::Reg)30,(engine::Reg)3);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)6);   // multiplicand <- X2
            eng->mul((engine::Reg)30,(engine::Reg)11);               // 30 = X1*X2

            // 31 = Y1*Y2
            eng->copy((engine::Reg)31,(engine::Reg)4);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);   // multiplicand <- Y2
            eng->mul((engine::Reg)31,(engine::Reg)11);               // 31 = Y1*Y2

            // sum & H d’un coup
            eng->addsub((engine::Reg)20,(engine::Reg)40, (engine::Reg)31,(engine::Reg)30); // 20 = Y1Y2+X1X2, 40 = H

            // 32 = C = d*T1*T2   (T2 via 46, d via 45)
            eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
            eng->mul((engine::Reg)32,(engine::Reg)46);               // 32 = T1*T2
            eng->mul((engine::Reg)32,(engine::Reg)45);               // 32 = d*T1*T2 = C

            // 42 = D + C, 41 = D - C  (D = Z1)
            eng->addsub((engine::Reg)42,(engine::Reg)41, (engine::Reg)1,(engine::Reg)32);

            // 38 = E = S1*S2 - sum
            eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
            eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)20);           // 38 -= sum  => E

            // --- profiter de 11 = H : calculer Y3 et T3 avant de changer 11 ---
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);  // multiplicand <- H
            eng->copy((engine::Reg)4,(engine::Reg)42);               // Y3 <- (D + C)
            eng->mul((engine::Reg)4,(engine::Reg)11);                // Y3 = (D + C)*H
            eng->copy((engine::Reg)5,(engine::Reg)38);               // T3 <- E
            eng->mul((engine::Reg)5,(engine::Reg)11);                // T3 = E*H

            // --- bascule unique vers (D - C) pour X3 et Z3 ---
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);  // multiplicand <- (D - C)
            eng->copy((engine::Reg)3,(engine::Reg)38);               // X3 <- E
            eng->mul((engine::Reg)3,(engine::Reg)11);                // X3 = E*(D - C)
            eng->copy((engine::Reg)1,(engine::Reg)42);               // Z3 <- (D + C)
            eng->mul((engine::Reg)1,(engine::Reg)11);                // Z3 = (D + C)*(D - C)
        };
        // eADD_RP_notwist : Twisted Edwards addition (a = 1, Z2 = 1)
        // Formulas:
        // E = X1*Y2 + Y1*X2
        // H = Y1*Y2 - a*X1*X2 (ici a=1)
        // C = d*T1*T2
        // D = Z1
        // X3 = E*(D - C)
        // Y3 = H*(D + C)
        // T3 = E*H
        // Z3 = (D - C)*(D + C)
        auto eADD_RP_notwist_2 = [&](){
            // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
            eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3);          // 34=S1=Y1+X1, 35=D1=Y1-X1
            eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)48,(engine::Reg)47);        // 36=S2=Y2+X2, 37=D2=Y2-X2

            // 30 = X1*X2  (X2-)
            eng->copy((engine::Reg)30,(engine::Reg)3);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)47);   // multiplicand <- X2-
            eng->mul((engine::Reg)30,(engine::Reg)11);                // 30 = X1*X2

            // 31 = Y1*Y2  (Y2-)
            eng->copy((engine::Reg)31,(engine::Reg)4);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)48);   // multiplicand <- Y2-
            eng->mul((engine::Reg)31,(engine::Reg)11);                // 31 = Y1*Y2

            // sum & H d’un coup
            eng->addsub((engine::Reg)20,(engine::Reg)40, (engine::Reg)31,(engine::Reg)30); // 20 = Y1Y2+X1X2, 40 = H

            // 32 = C = d*T1*T2  (T2- via 49, puis d)
            eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)49);  // multiplicand <- T2-
            eng->mul((engine::Reg)32,(engine::Reg)50);               // 32 = T1*T2
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)29);  // multiplicand <- d
            eng->mul((engine::Reg)32,(engine::Reg)45);               // 32 = d*T1*T2 = C

            // 42 = D + C, 41 = D - C  (D = Z1)
            eng->addsub((engine::Reg)42,(engine::Reg)41, (engine::Reg)1,(engine::Reg)32);

            // 38 = E = S1*S2 - sum
            eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
            eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)20);           // 38 -= sum => E

            // --- profiter de 11 = H : calculer Y3 et T3 avant de changer 11 ---
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);  // multiplicand <- H
            eng->copy((engine::Reg)4,(engine::Reg)42);               // Y3 <- (D + C)
            eng->mul((engine::Reg)4,(engine::Reg)11);                // Y3 = (D + C)*H
            eng->copy((engine::Reg)5,(engine::Reg)38);               // T3 <- E
            eng->mul((engine::Reg)5,(engine::Reg)11);                // T3 = E*H

            // --- bascule unique vers (D - C) pour X3 et Z3 ---
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);  // multiplicand <- (D - C)
            eng->copy((engine::Reg)3,(engine::Reg)38);               // X3 <- E
            eng->mul((engine::Reg)3,(engine::Reg)11);                // X3 = E*(D - C)
            eng->copy((engine::Reg)1,(engine::Reg)42);               // Z3 <- (D + C)
            eng->mul((engine::Reg)1,(engine::Reg)11);                // Z3 = (D + C)*(D - C)
        };


        mpz_t za; mpz_init(za); mpz_set(za, aE.get_mpz_t()); eng->set_mpz((engine::Reg)16, za); mpz_clear(za);
        static const mpz_class TWO = 2;
        mpz_class two_dE = mulm(dE, TWO);
        mpz_t tmp; mpz_init_set(tmp, two_dE.get_mpz_t());
        eng->set_mpz((engine::Reg)17, tmp);
        mpz_clear(tmp);
        mpz_t zd; mpz_init(zd); mpz_set(zd, dE.get_mpz_t());
        eng->set_mpz((engine::Reg)29, zd);
        mpz_clear(zd);
        eng->set((engine::Reg)0, 0u);
        eng->set((engine::Reg)1, 1u);
        mpz_t zx; mpz_init(zx); mpz_set(zx, X0.get_mpz_t()); eng->set_mpz((engine::Reg)6, zx); mpz_clear(zx);
        mpz_t zy; mpz_init(zy); mpz_set(zy, Y0.get_mpz_t()); eng->set_mpz((engine::Reg)7, zy); mpz_clear(zy);
        eng->set((engine::Reg)8, 2u);
        eng->copy((engine::Reg)9,(engine::Reg)6);
        eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);
        eng->mul((engine::Reg)9,(engine::Reg)11);
        eng->set((engine::Reg)1, 1u);
        eng->copy((engine::Reg)3,(engine::Reg)6);
        eng->copy((engine::Reg)4,(engine::Reg)7);
        eng->copy((engine::Reg)5,(engine::Reg)9);
        eng->set_multiplicand((engine::Reg)43,(engine::Reg)16); // a
        eng->set_multiplicand((engine::Reg)44,(engine::Reg)8);  // 2
        eng->set_multiplicand((engine::Reg)45,(engine::Reg)29); // d
        eng->set_multiplicand((engine::Reg)46,(engine::Reg)9);  // T2 (will be T_pos)
        uint32_t start_i = 0, nb_ck = 0;
        double   saved_et = 0.0;
        int rr = read_ckpt(start_i, nb_ck, saved_et);
        
        int rr2 = read_ckpt2(s2_idx, s2_cnt, s2_et); 
        resume_stage2 = (rr2 == 0);
        
        

        bool resumed = (rr == 0 && start_i > 0);
        if (!resumed) {
            saved_et = 0.0;
            nb_ck = 0;
        } else {
            // Treat the resumed state as a valid last-good checkpoint
            if (ctx_ckpt_size > 0) {
                last_good_state.resize(ctx_ckpt_size);
                if (eng->get_checkpoint(last_good_state)) {
                    have_last_good_state = true;
                    last_good_iter = start_i;
                    current_iter_for_invariant = start_i;
                }
            }
        }

        auto t0 = high_resolution_clock::now();
        auto last_save = t0;
        auto last_ui   = t0;
        auto last_check = t0;
        size_t last_ui_done = start_i;
        double ema_ips_stage1 = 0.0;

        std::cout<<"[ECM] stage1_begin Kbits="<<Kbits<<std::endl;
        std::cout<<"[ECM] Stage1 init: building NAF path..."<<std::endl;
        std::vector<short> naf_vec; naf_vec.reserve((size_t)Kbits + 2);
        {
            mpz_class ec = K;
            mpz_ptr e = ec.get_mpz_t();
            auto naf_t0 = high_resolution_clock::now();
            auto naf_last = naf_t0;
            size_t naf_done = 0;
            size_t naf_last_done = 0;
            double ema_naf_bits_per_s = 0.0;
            for (; mpz_size(e) != 0; )
            {
                short di = 0;
                if (mpz_odd_p(e))
                {
                    unsigned long limb0 = (mpz_size(e) > 0) ? mpz_getlimbn(e, 0) : 0ul;
                    short mod4 = short(limb0 & 3u);
                    di = (mod4 == 1) ? 1 : -1;
                    if (di > 0) mpz_sub_ui(e, e, 1u); else mpz_add_ui(e, e, 1u);
                }
                naf_vec.push_back(di);
                mpz_fdiv_q_2exp(e, e, 1);
                ++naf_done;

                auto now = high_resolution_clock::now();
                if (duration_cast<milliseconds>(now - naf_last).count() >= progress_interval_ms) {
                    const double elapsed = duration<double>(now - naf_t0).count();
                    const double dt = duration<double>(now - naf_last).count();
                    const double dd = double(naf_done - naf_last_done);
                    const double inst = (dt > 1e-9) ? (dd / dt) : 0.0;
                    const double avg = (elapsed > 1e-9) ? (double)naf_done / elapsed : 0.0;
                    const double speed = (inst > 0.0) ? inst : avg;
                    if (ema_naf_bits_per_s <= 0.0) ema_naf_bits_per_s = speed;
                    else ema_naf_bits_per_s = 0.75 * ema_naf_bits_per_s + 0.25 * speed;
                    const double pct = (Kbits > 0) ? (100.0 * double(naf_done) / double(Kbits)) : 100.0;
                    const double eta = (ema_naf_bits_per_s > 0.0 && naf_done < Kbits)
                                     ? (double(Kbits - naf_done) / ema_naf_bits_per_s)
                                     : 0.0;

                    std::ostringstream line;
                    line << "[ECM] Stage1 init: NAF "
                         << naf_done << "/" << Kbits
                         << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                         << " | bits/s " << std::fixed << std::setprecision(1) << ema_naf_bits_per_s
                         << " | ETA " << fmt_hms(eta);
                    ecm_print_progress_line(line.str());
                    naf_last = now;
                    naf_last_done = naf_done;
                }
            }
            while (!naf_vec.empty() && naf_vec.back()==0) naf_vec.pop_back();
            std::ostringstream line;
            line << "[ECM] Stage1 init: NAF "
                 << naf_done << "/" << Kbits
                 << " (100.0%)"
                 << " | elapsed " << std::fixed << std::setprecision(1)
                 << duration<double>(high_resolution_clock::now() - naf_t0).count() << "s";
            ecm_print_progress_line(line.str());
        }
        std::cout<<std::endl;
        size_t naf_len = naf_vec.size();
        if (naf_len == 0) { std::cout<<std::endl; }
        if (naf_len == 0) { }
        std::cout<<"[ECM] Stage1 init: loading cached base point registers..."<<std::endl;
        mpz_class T0_mpz = mulm(X0, Y0);
        mpz_class X0_neg = subm(N, X0);
        mpz_class T0_neg = subm(N, T0_mpz);
        mpz_t zXpos; mpz_init_set(zXpos, X0.get_mpz_t());
        mpz_t zYpos; mpz_init_set(zYpos, Y0.get_mpz_t());
        mpz_t zTpos; mpz_init_set(zTpos, T0_mpz.get_mpz_t());
        mpz_t zXneg; mpz_init_set(zXneg, X0_neg.get_mpz_t());
        mpz_t zYneg; mpz_init_set(zYneg, Y0.get_mpz_t());
        mpz_t zTneg; mpz_init_set(zTneg, T0_neg.get_mpz_t());

        size_t total_steps = (naf_len>=1 ? naf_len-1 : 0);

        eng->set_mpz((engine::Reg)6,  zXpos);   // X2 =  +P.x
        eng->set_mpz((engine::Reg)7,  zYpos);   // Y2 =  +P.y
        eng->set_mpz((engine::Reg)9,  zTpos);   // T2 =  +P.t
        eng->set_mpz((engine::Reg)47, zXneg);   // cache −P.x
        eng->set_mpz((engine::Reg)48, zYneg);   // cache −P.y
        eng->set_mpz((engine::Reg)49, zTneg);   // cache −P.t
        eng->set_multiplicand((engine::Reg)43,(engine::Reg)16); // a
        eng->set_multiplicand((engine::Reg)44,(engine::Reg)8);  // 2
        eng->set_multiplicand((engine::Reg)45,(engine::Reg)29); // d
        eng->set_multiplicand((engine::Reg)46,(engine::Reg)9);  // T2 = +P.t
        eng->set_multiplicand((engine::Reg)50,(engine::Reg)49); // T2neg = −P.t
        std::cout<<"[ECM] Stage1 init: base point registers ready"<<std::endl;

        if (!resumed && !resume_stage2 && naf_len) {
            short top = naf_vec[naf_len - 1];
            if (top < 0) {
                eng->set_mpz((engine::Reg)3, zXneg);
                eng->set_mpz((engine::Reg)4, zYneg);
                eng->set_mpz((engine::Reg)5, zTneg);
            } else {
                eng->set_mpz((engine::Reg)3, zXpos);
                eng->set_mpz((engine::Reg)4, zYpos);
                eng->set_mpz((engine::Reg)5, zTpos);
            }
        }
        bool errordone = false;
        bool fatal_error = false;
        if(resume_stage2){
            start_i = total_steps;
        }
        for (uint32_t i = start_i; i < total_steps; ++i) {
            if (interrupted.load(std::memory_order_relaxed)) {
                if (result_status == "found" && result_factor > 0) {
                    return p95_finalize_background_stop(eng);
                }
                if (!p95_background_error.empty()) {
                    double elapsed = duration<double>(high_resolution_clock::now() - t0).count() + saved_et;
                    save_ckpt((uint32_t)i, elapsed);
                    return p95_finalize_background_stop(eng);
                }
                double elapsed = duration<double>(high_resolution_clock::now() - t0).count() + saved_et;
                save_ckpt((uint32_t)i, elapsed);
                std::cout << "[ECM] Interrupted at curve " << (c+1) << ", iter " << i << "/" << total_steps << "";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "[ECM] Interrupted at curve " << (c+1) << ", iter " << i << "/" << total_steps;
                    guiServer_->appendLog(oss.str());
                }
                delete eng;
                return 0;
            }

            
            if(te_use_torsion16){
                eDBL_XYTZ_notwist(3,4,1,5);
            }
            else{
                eDBL_XYTZ(3,4,1,5);
            }
            short di = naf_vec[naf_len - 2 - i];
            if (di != 0){
                if (di > 0){
                    /*eng->set_mpz((engine::Reg)6, zXpos);
                    eng->set_mpz((engine::Reg)7, zYpos);
                    eng->set_mpz((engine::Reg)9, zTpos);*/
                    if(te_use_torsion16) eADD_RP_notwist(); else eADD_RP();
                } else {
                    //eng->set_mpz((engine::Reg)6, zXneg);
                    //eng->set_mpz((engine::Reg)7, zYneg);
                    //eng->set_mpz((engine::Reg)9, zTneg);
                    if(te_use_torsion16) eADD_RP_notwist_2(); else eADD_RP_2();
                }
            }
            if (options.erroriter > 0 && (i + 1) == options.erroriter && !errordone) {
                errordone = true;
                //eng->error();
                eng->sub(1, 2);
                std::cout << "Injected error at iteration " << (i + 1) << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Injected error at iteration " << (i + 1);
                    guiServer_->appendLog(oss.str());
                }
            }
            auto now = high_resolution_clock::now();
            if ((options.ecm_check_interval <= 0 && i + 1 == total_steps) ||
                (options.ecm_check_interval > 0 &&
                (duration_cast<seconds>(now - last_check).count() >= options.ecm_check_interval || i + 1 == total_steps))) {
                std::cout << "\n[ECM] Error check ...." << std::endl;
                current_iter_for_invariant = i + 1;
                if (check_invariant()) {
                    std::cout << "[ECM] Error check Done ! ...." << std::endl;
                } else {
                    std::cout << "[ECM] Error detected!!!!!!!! ...." << std::endl;
                    if (have_last_good_state) {
                        options.invarianterror += 1;
                        std::cout << "[ECM] Restoring last known good state at iteration "
                                << last_good_iter << " and retrying from there." << std::endl;
                        if (eng->set_checkpoint(last_good_state)) {
                            i = (last_good_iter > 0) ? (last_good_iter - 1) : 0;
                            last_check = now;
                            last_save  = now;
                            last_ui = now;
                            last_ui_done = (last_good_iter > 0) ? size_t(last_good_iter) : size_t(0);
                            ema_ips_stage1 = 0.0;
                            continue;
                        } else {
                            std::cout << "[ECM] Failed to restore last good state, aborting curve." << std::endl;
                        }
                    } else {
                        std::cout << "[ECM] No saved good state available, aborting curve." << std::endl;
                    }
                    fatal_error = true;
                    break;
                }
                last_check = now;
            }
            if (duration_cast<milliseconds>(now - last_ui).count() >= progress_interval_ms || i+1 == total_steps) {
                if (p95_stage2_enabled && p95_poll_background(false)) {
                    interrupted.store(true, std::memory_order_relaxed);
                }
                const size_t done_u = i + 1;
                const double done = double(done_u), total = double(total_steps ? total_steps : 1);
                const double elapsed = duration<double>(now - t0).count() + saved_et;
                const double avg_ips = done / std::max(1e-9, elapsed);
                const double dt_ui = duration<double>(now - last_ui).count();
                const double dd_ui = double(done_u - last_ui_done);
                const double inst_ips = (dt_ui > 1e-9 && dd_ui >= 0.0) ? (dd_ui / dt_ui) : avg_ips;
                if (ema_ips_stage1 <= 0.0) ema_ips_stage1 = inst_ips;
                else ema_ips_stage1 = 0.75 * ema_ips_stage1 + 0.25 * inst_ips;
                const double eta_ips = (ema_ips_stage1 > 0.0) ? ema_ips_stage1 : avg_ips;
                const double eta = (total > done && eta_ips > 0.0) ? (total - done) / eta_ips : 0.0;
                std::ostringstream line;
                line << "[ECM] Curve " << (c+1) << "/" << curves << " | Stage1 " << (i+1) << "/" << total_steps
                     << " (" << std::fixed << std::setprecision(2) << (done * 100.0 / total) << "%)"
                     << " | it/s " << std::fixed << std::setprecision(1) << eta_ips
                     << " | ETA " << fmt_hms(eta);
                ecm_print_progress_line(line.str());
                last_ui_done = done_u;
                last_ui = now;
            }
            if (duration_cast<seconds>(now - last_save).count() >= backup_period) {
                double elapsed = duration<double>(now - t0).count() + saved_et;
                save_ckpt((uint32_t)(i + 1), elapsed);
                last_save = now;
            }
        }
        std::cout << std::endl;
        mpz_clear(zXpos); mpz_clear(zYpos); mpz_clear(zTpos);
        mpz_clear(zXneg); mpz_clear(zYneg); mpz_clear(zTneg);

        if (fatal_error) {
            std::cout << "[ECM] Curve " << (c+1) << " aborted after unrecoverable error, skipping\n";
            delete eng;
            continue;
        }

        if (resume_stage2 && (s2_cnt != (uint32_t)primesS2_v.size() || s2_idx > (uint32_t)primesS2_v.size())) {
            resume_stage2 = false;
            s2_idx = 0;
            s2_cnt = 0;
            s2_et = 0.0;
        }

        std::string curve_p95_resume_path;
        if (!resume_stage2) {
            mpz_class Zacc = compute_X_with_dots(eng, (engine::Reg)5, N);
            mpz_class g = gcd_with_dots(Zacc, N);
            if (g == N) {
                std::cout<<"[ECM] Curve "<<(c+1)<<": singular or failure, retrying\n";
                delete eng;
                continue;
            }

            bool found = (g > 1 && g < N);

            double elapsed_stage1 = duration<double>(high_resolution_clock::now() - t0).count() + saved_et;
            {
                std::ostringstream s1;
                s1<<"[ECM] Curve "<<(c+1)<<"/"<<curves
                   <<" | Stage1 elapsed="<<fixed<<setprecision(2)<<elapsed_stage1<<" s";
                std::cout<<s1.str()<<std::endl;
                if (guiServer_) guiServer_->appendLog(s1.str());
            }
            std::cout<<"\n[ECM] Error check ...."<<std::endl;

            mpz_class uAff = 0;
            mpz_class Aresume = 0;
            bool have_u = false;
            bool have_A = false;

            if (g == 1) {
                mpz_class Zv = compute_X_with_dots(eng, (engine::Reg)1, N);
                mpz_class Yv = compute_X_with_dots(eng, (engine::Reg)4, N);

                mpz_class den_u = subm(Zv, Yv), inv_den_u;
                int r_u = invm(den_u, inv_den_u);
                if (r_u == 1) found = true;
                if (r_u == 0) {
                    uAff = mulm(addm(Zv, Yv), inv_den_u);
                    have_u = true;

                    mpz_class numA = mulm(mpz_class(2), addm(aE, dE));
                    mpz_class denA = subm(aE, dE), invDenA;
                    int r_a = invm(denA, invDenA);
                    if (r_a == 1) found = true;
                    if (r_a == 0) {
                        Aresume = mulm(numA, invDenA);
                        have_A = true;
                    }
                }
            }

            if ((have_u && have_A) || found) {
                curve_p95_resume_path = append_ecm_stage1_resume_line(
                    c,
                    have_A ? Aresume : mpz_class(0),
                    have_u ? uAff   : mpz_class(0),
                    have_sigma_resume ? &sigma_resume : nullptr
                );
            }

            if (found) {
                bool known = is_known(result_factor > 1 ? result_factor : g);
                const mpz_class& gf = (result_factor > 1 ? result_factor : g);

                std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves
                         <<(known?" | known factor=":" | factor=")<<gf.get_str()<<std::endl;
                std::cout << "[ECM] This result has been added to ecm_result.json" << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<gf.get_str();
                    guiServer_->appendLog(oss.str());
                }

                std::error_code ec0;
                fs::remove(ckpt_file, ec0); fs::remove(ckpt_file + ".old", ec0); fs::remove(ckpt_file + ".new", ec0);
                fs::remove(ckpt2_file, ec0); fs::remove(ckpt2_file + ".old", ec0); fs::remove(ckpt2_file + ".new", ec0);

                if (!known) {
                    if (!(result_factor > 1)) result_factor = gf;
                    result_status = "found";
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                }

                delete eng;
                continue;
            }
        }


        if (p95_stage2_enabled && B2 > B1 && !resume_stage2 && !curve_p95_resume_path.empty()) {
            if (p95_poll_background(false)) {
                return p95_finalize_background_stop(eng);
            }
            std::error_code ecq;
            fs::remove(ckpt_file, ecq); fs::remove(ckpt_file + ".old", ecq); fs::remove(ckpt_file + ".new", ecq);
            if (!p95_enqueue_curve(c, curve_p95_resume_path, options.sigma_hex, options.curve_seed, options.base_seed)) {
                p95_log(std::string("[ECM] Curve ") + std::to_string(c + 1) + "/" + std::to_string(curves) + " | Prime95 Stage2 enqueue failed, falling back to internal Stage2");
            } else {
                std::ostringstream fin;
                fin << "[ECM] Curve " << (c+1) << "/" << curves << " | Stage2 delegated to Prime95 background";
                std::cout << fin.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(fin.str());
                delete eng;
                continue;
            }
        }

        if (B2 > B1 /*&& !primesS2_v.empty()*/) {
            auto t2_0 = high_resolution_clock::now();
            auto last2_save = t2_0, last2_ui = t2_0;
            double saved_et2 = resume_stage2 ? s2_et : 0.0;

            if (stage2_debug_checks) {
                std::ostringstream oss;
                oss << "[ECM] Stage2 debug ON | chunk_bits=" << compute_s2_chunk_bits(transform_size_once)
                    << " | primes=" << primesS2_v.size()
                    << " | resume_idx=" << s2_idx
                    << " | total_bits=" << total_s2_iters;
                std::cout << oss.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(oss.str());
            }

            uint64_t done_bits_est = 0;
            for (size_t ci = 0; ci < s2_chunk_ends.size(); ++ci) {
                if (s2_chunk_ends[ci] <= s2_idx) done_bits_est = s2_chunk_prefix_iters[ci];
                else break;
            }
            uint64_t last2_done_bits = done_bits_est;
            double ema_ips_stage2 = 0.0;

            auto publish_stage2_factor = [&](const mpz_class& gg)->int {
                bool known = is_known(gg);
                std::cout << "[ECM] Curve " << (c+1) << "/" << curves << (known ? " | known factor=" : " | factor=") << gg.get_str() << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "[ECM] " << (known ? "Known " : "") << "factor: " << gg.get_str();
                    guiServer_->appendLog(oss.str());
                }

                std::error_code ec;
                fs::remove(ckpt_file, ec);  fs::remove(ckpt_file + ".old", ec);  fs::remove(ckpt_file + ".new", ec);
                fs::remove(ckpt2_file, ec); fs::remove(ckpt2_file + ".old", ec); fs::remove(ckpt2_file + ".new", ec);

                if (!known) {
                    result_factor = gg;
                    result_status = "found";
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = c+1;
                    write_result();
                    publish_json();
                }

                delete eng;
                return known ? 1 : 2;
            };

            auto setup_te_stage2_base = [&]() -> int {
                mpz_class Xv = compute_X_with_dots(eng, (engine::Reg)3, N);
                mpz_class Yv = compute_X_with_dots(eng, (engine::Reg)4, N);
                mpz_class Zv = compute_X_with_dots(eng, (engine::Reg)1, N);
                mpz_class invZv;
                int r = invm(Zv, invZv);
                if (r != 0) return r;

                mpz_class Xaff = mulm(Xv, invZv);
                mpz_class Yaff = mulm(Yv, invZv);
                mpz_class Taff = mulm(Xaff, Yaff);
                mpz_class Xneg = subm(N, Xaff);
                mpz_class Tneg = subm(N, Taff);

                s2_base_Xpos = Xaff;
                s2_base_Ypos = Yaff;
                s2_base_Tpos = Taff;
                s2_base_Xneg = Xneg;
                s2_base_Yneg = Yaff;
                s2_base_Tneg = Tneg;
                have_s2_base_cache = true;

                restore_te_stage2_base_from_cache();
                return 0;
            };

            bool interrupted_in_chunk = false;
            auto run_te_stage2_chunk = [&](const mpz_class& Echunk, uint32_t chunk_start, uint32_t chunk_end, uint32_t chunk_bits, uint32_t resume_steps_done) -> bool {
                std::vector<short> naf2;
                naf2.reserve((size_t)mpz_sizeinbase(Echunk.get_mpz_t(), 2) + 2);
                mpz_class ec = Echunk;
                mpz_ptr e = ec.get_mpz_t();
                for (; mpz_size(e) != 0; ) {
                    short di = 0;
                    if (mpz_odd_p(e)) {
                        unsigned long limb0 = (mpz_size(e) > 0) ? mpz_getlimbn(e, 0) : 0ul;
                        short mod4 = short(limb0 & 3u);
                        di = (mod4 == 1) ? 1 : -1;
                        if (di > 0) mpz_sub_ui(e, e, 1u); else mpz_add_ui(e, e, 1u);
                    }
                    naf2.push_back(di);
                    mpz_fdiv_q_2exp(e, e, 1);
                }
                while (!naf2.empty() && naf2.back() == 0) naf2.pop_back();
                if (naf2.empty()) return true;

                short top = naf2.back();
                const size_t total_steps2 = (naf2.size() >= 1 ? naf2.size() - 1 : 0);
                if (resume_steps_done == 0) {
                    eng->set((engine::Reg)1, 1u);
                    if (top < 0) {
                        eng->copy((engine::Reg)3, (engine::Reg)47);
                        eng->copy((engine::Reg)4, (engine::Reg)48);
                        eng->copy((engine::Reg)5, (engine::Reg)49);
                    } else {
                        eng->copy((engine::Reg)3, (engine::Reg)6);
                        eng->copy((engine::Reg)4, (engine::Reg)7);
                        eng->copy((engine::Reg)5, (engine::Reg)9);
                    }
                } else if ((size_t)resume_steps_done >= total_steps2) {
                    return true;
                }

                for (size_t j = (size_t)resume_steps_done; j < total_steps2; ++j) {
                    if (te_use_torsion16) eDBL_XYTZ_notwist(3,4,1,5);
                    else                  eDBL_XYTZ(3,4,1,5);

                    short di = naf2[naf2.size() - 2 - j];
                    if (di != 0) {
                        if (di > 0) {
                            if (te_use_torsion16) eADD_RP_notwist(); else eADD_RP();
                        } else {
                            if (te_use_torsion16) eADD_RP_notwist_2(); else eADD_RP_2();
                        }
                    }

                    if (((j + 1) & 1023u) == 0u || (j + 1) == total_steps2) {
                        auto now2 = high_resolution_clock::now();
                        if (duration_cast<milliseconds>(now2 - last2_ui).count() >= progress_interval_ms || (j + 1) == total_steps2) {
                            if (p95_stage2_enabled && p95_poll_background(false)) {
                                interrupted.store(true, std::memory_order_relaxed);
                            }
                            uint64_t inside_done = done_bits_est + std::min<uint64_t>((uint64_t)(j + 1), (uint64_t)chunk_bits);
                            const double done = double(inside_done);
                            const double total = double(std::max<uint64_t>(1ULL, total_s2_iters));
                            const double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                            const double avg_ips = done / std::max(1e-9, elapsed);
                            const double dt_ui = duration<double>(now2 - last2_ui).count();
                            const double dd_ui = double(inside_done - last2_done_bits);
                            const double inst_ips = (dt_ui > 1e-9 && dd_ui >= 0.0) ? (dd_ui / dt_ui) : avg_ips;
                            if (ema_ips_stage2 <= 0.0) ema_ips_stage2 = inst_ips;
                            else ema_ips_stage2 = 0.75 * ema_ips_stage2 + 0.25 * inst_ips;
                            const double eta_ips = (ema_ips_stage2 > 0.0) ? ema_ips_stage2 : avg_ips;
                            const double eta = (total > done && eta_ips > 0.0) ? (total - done) / eta_ips : 0.0;
                            std::ostringstream line;
                            line << "[ECM] Curve " << (c+1) << "/" << curves
                                 << " | Stage2 " << std::fixed << std::setprecision(1) << (100.0 * done / total) << "%"
                                 << " | primes " << (chunk_start + 1) << "-" << chunk_end << "/" << primesS2_v.size()
                                 << " | it/s " << std::fixed << std::setprecision(1) << eta_ips
                                 << " | ETA " << fmt_hms(eta);
                            ecm_print_progress_line(line.str());
                            last2_done_bits = inside_done;
                            last2_ui = now2;
                        }
                        if (interrupted.load(std::memory_order_relaxed) && (j + 1) != total_steps2) {
                            if (result_status == "found" && result_factor > 0) {
                                interrupted_in_chunk = true;
                                return false;
                            }
                            if (!p95_background_error.empty()) {
                                interrupted_in_chunk = true;
                                return false;
                            }
                            const double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                            save_ckpt2_ex(chunk_start, elapsed, (uint32_t)primesS2_v.size(),
                                          1u, chunk_start, chunk_end, chunk_bits, (uint32_t)(j + 1));
                            interrupted_in_chunk = true;
                            return false;
                        }
                    }
                }
                return true;
            };

            bool abort_curve = false;
            bool stop_after_chunk = false;
            bool stop_msg_printed = false;
            bool handled_known_factor = false;

            while (s2_idx < (uint32_t)primesS2_v.size()) {
                mpz_class Echunk(1);
                uint32_t chunk_start = s2_idx;
                uint32_t chunk_end = s2_idx;
                uint32_t chunk_bits = 0;
                uint32_t chunk_resume_steps_done = 0;
                bool resume_this_chunk = false;
                const uint32_t max_s2_chunk_bits = compute_s2_chunk_bits(transform_size_once);
                if (resume_stage2_in_chunk && s2_idx == resume_s2_chunk_start && resume_s2_chunk_end > resume_s2_chunk_start) {
                    chunk_start = resume_s2_chunk_start;
                    chunk_end = resume_s2_chunk_end;
                    for (uint32_t qi = chunk_start; qi < chunk_end; ++qi) {
                        mpz_mul_ui(Echunk.get_mpz_t(), Echunk.get_mpz_t(), primesS2_v[qi]);
                    }
                    chunk_bits = resume_s2_chunk_bits ? resume_s2_chunk_bits : (uint32_t)mpz_sizeinbase(Echunk.get_mpz_t(), 2);
                    chunk_resume_steps_done = resume_s2_steps_done;
                    resume_this_chunk = true;
                } else {
                    while (chunk_end < (uint32_t)primesS2_v.size()) {
                        mpz_mul_ui(Echunk.get_mpz_t(), Echunk.get_mpz_t(), primesS2_v[chunk_end]);
                        ++chunk_end;
                        if (mpz_sizeinbase(Echunk.get_mpz_t(), 2) >= max_s2_chunk_bits && chunk_end > chunk_start) break;
                    }
                    chunk_bits = (uint32_t)mpz_sizeinbase(Echunk.get_mpz_t(), 2);
                }

                if (stage2_debug_checks) {
                    std::ostringstream oss;
                    oss << "[ECM] Stage2 chunk begin | curve=" << (c+1)
                        << " | primes=" << (chunk_start + 1) << "-" << chunk_end << "/" << primesS2_v.size()
                        << " | bits=" << chunk_bits;
                    std::cout << oss.str() << std::endl;
                    if (guiServer_) guiServer_->appendLog(oss.str());
                }

                if (!resume_this_chunk) {
                    int setup_rc = setup_te_stage2_base();
                    if (setup_rc == 1) {
                        int factor_rc = publish_stage2_factor(result_factor);
                        if (factor_rc == 2) return 0;
                        handled_known_factor = true;
                        break;
                    }
                    if (setup_rc < 0) {
                        abort_curve = true;
                        break;
                    }
                }

                if (resume_this_chunk) {
                    restore_te_stage2_base_from_cache();
                    last2_done_bits = done_bits_est + std::min<uint64_t>((uint64_t)chunk_resume_steps_done, (uint64_t)chunk_bits);
                    last2_ui = high_resolution_clock::now();
                }

                bool chunk_completed = run_te_stage2_chunk(Echunk, chunk_start, chunk_end, chunk_bits, chunk_resume_steps_done);
                if (!chunk_completed) {
                    stop_after_chunk = true;
                    if (!stop_msg_printed) {
                        stop_msg_printed = true;
                        std::cout << "[ECM] Interrupt received — checkpoint saved inside current Stage2 chunk...\n";
                    }
                    break;
                }

                resume_stage2_in_chunk = false;
                resume_s2_chunk_start = 0;
                resume_s2_chunk_end = 0;
                resume_s2_chunk_bits = 0;
                resume_s2_steps_done = 0;

                done_bits_est += chunk_bits;
                s2_idx = chunk_end;

                auto now2 = high_resolution_clock::now();
                const double done = double(done_bits_est);
                const double total = double(std::max<uint64_t>(1ULL, total_s2_iters));
                const double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                const double avg_ips = done / std::max(1e-9, elapsed);
                const double dt_ui = duration<double>(now2 - last2_ui).count();
                const double dd_ui = double(done_bits_est - last2_done_bits);
                const double inst_ips = (dt_ui > 1e-9 && dd_ui >= 0.0) ? (dd_ui / dt_ui) : avg_ips;
                if (ema_ips_stage2 <= 0.0) ema_ips_stage2 = inst_ips;
                else ema_ips_stage2 = 0.75 * ema_ips_stage2 + 0.25 * inst_ips;
                const double eta_ips = (ema_ips_stage2 > 0.0) ? ema_ips_stage2 : avg_ips;
                const double eta = (total > done && eta_ips > 0.0) ? (total - done) / eta_ips : 0.0;
                std::ostringstream line;
                line << "[ECM] Curve " << (c+1) << "/" << curves
                     << " | Stage2 " << std::fixed << std::setprecision(1) << (100.0 * done / total) << "%"
                     << " | primes " << (chunk_start + 1) << "-" << chunk_end << "/" << primesS2_v.size()
                     << " | it/s " << std::fixed << std::setprecision(1) << eta_ips
                     << " | ETA " << fmt_hms(eta);
                ecm_print_progress_line(line.str());
                last2_done_bits = done_bits_est;
                last2_ui = now2;

                mpz_class Tchunk = compute_X_with_dots(eng, (engine::Reg)5, N);
                mpz_class gz = gcd_with_dots(Tchunk, N);
                if (stage2_debug_checks) {
                    std::ostringstream oss;
                    oss << "[ECM] Stage2 chunk end | curve=" << (c+1)
                        << " | next_prime_idx=" << s2_idx
                        << " | gcd(T,N)=" << gz.get_str();
                    std::cout << oss.str() << std::endl;
                    if (guiServer_) guiServer_->appendLog(oss.str());
                }
                if (gz == N) {
                    std::cout << "[ECM] Curve " << (c+1) << ": Stage2 gcd=N, retrying\n";
                    abort_curve = true;
                    break;
                }
                if (gz > 1 && gz < N) {
                    std::cout << std::endl;
                    int factor_rc = publish_stage2_factor(gz);
                    if (factor_rc == 2) return 0;
                    handled_known_factor = true;
                    break;
                }

                if (duration_cast<seconds>(now2 - last2_save).count() >= backup_period || s2_idx == (uint32_t)primesS2_v.size()) {
                    save_ckpt2(s2_idx, elapsed, (uint32_t)primesS2_v.size());
                    last2_save = now2;
                }

                if (interrupted.load(std::memory_order_relaxed)) {
                    stop_after_chunk = true;
                    if (!stop_msg_printed) {
                        stop_msg_printed = true;
                        std::cout << "[ECM] Interrupt received — finishing current Stage2 chunk before stopping...\n";
                    }
                }
                if (stop_after_chunk) {
                    std::cout << "\n[ECM] Interrupted at Stage2 curve " << (c+1)
                              << " prime-index " << s2_idx << "/" << primesS2_v.size() << "\n";
                    if (guiServer_) {
                        std::ostringstream oss;
                        uint32_t stop_idx2 = resume_stage2_in_chunk ? resume_s2_chunk_start : s2_idx;
                        oss << "[ECM] Interrupted at Stage2 curve " << (c+1) << " prime-index " << stop_idx2 << "/" << primesS2_v.size();
                        guiServer_->appendLog(oss.str());
                    }
                    save_ckpt2(s2_idx, elapsed, (uint32_t)primesS2_v.size());
                    delete eng;
                    return 0;
                }
            }
            std::cout << std::endl;
            if (stop_after_chunk) {
                if (result_status == "found" && result_factor > 0) {
                    return p95_finalize_background_stop(eng);
                }
                if (!p95_background_error.empty()) {
                    return p95_finalize_background_stop(eng);
                }
                delete eng;
                return 0;
            }
            if (handled_known_factor) continue;
            if (abort_curve) {
                std::error_code ec;
                fs::remove(ckpt_file, ec);  fs::remove(ckpt_file + ".old", ec);  fs::remove(ckpt_file + ".new", ec);
                fs::remove(ckpt2_file, ec); fs::remove(ckpt2_file + ".old", ec); fs::remove(ckpt2_file + ".new", ec);
                delete eng;
                continue;
            }

            std::error_code ec2;
            fs::remove(ckpt2_file, ec2);
            fs::remove(ckpt2_file + ".old", ec2);
            fs::remove(ckpt2_file + ".new", ec2);

            double elapsed2 = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2;
            { std::ostringstream s2s; s2s<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 elapsed="<<std::fixed<<std::setprecision(2)<<elapsed2<<" s"; std::cout<<s2s.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2s.str()); }
        }

        if (p95_stage2_enabled && p95_poll_background(false)) {
            return p95_finalize_background_stop(eng);
        }
        std::error_code ec; fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec);
        { std::ostringstream fin; fin<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" done"; std::cout<<fin.str()<<std::endl; if (guiServer_) guiServer_->appendLog(fin.str()); }
        delete eng;
    }

    if (p95_stage2_enabled) {
        if (p95_future_active || !p95_pending_tasks.empty()) {
            std::ostringstream oss;
            oss << "[ECM] Waiting for Prime95 Stage2 background jobs to finish"
                << " | active=" << (p95_future_active ? 1 : 0)
                << " | pending=" << p95_pending_tasks.size();
            p95_log(oss.str());
        }
        while (p95_future_active || !p95_pending_tasks.empty()) {
            if (p95_poll_background(true)) {
                return p95_finalize_background_stop(nullptr);
            }
        }
        if (!p95_background_error.empty() || (result_status == "found" && result_factor > 0)) {
            return p95_finalize_background_stop(nullptr);
        }
    }

    if (result_status == "found") return 0;

    std::cout<<"[ECM] No factor found"<<std::endl;
    curves_tested_for_found = curves;
    options.curves_tested_for_found = (uint32_t)curves;
    write_result();
    publish_json();
    return 1;
}

} // namespace core