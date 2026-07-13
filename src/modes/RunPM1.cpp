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
#include "marin/ibdwt.h"
#include "marin/file.h"
#include "ui/WebGuiServer.hpp"
#include "core/Version.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <map>
#include <future>
#include <cinttypes>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#ifndef CL_DEVICE_MAX_MEM_ALLOC_SIZE
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#endif
#ifdef _WIN32
# include <windows.h>
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
#include <unordered_map>
#include <limits>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <numeric>
#include <climits>
#include <gmpxx.h>
#include <cstdlib>

using namespace core;
using namespace std::chrono;
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
using core::algo::evenGapBound2;
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

static inline void mpz_set_u64(mpz_t z, uint64_t x) {
#if ULONG_MAX >= UINT64_MAX
    mpz_set_ui(z, (unsigned long)x);
#else
    mpz_import(z, 1, -1, sizeof(x), 0, 0, &x);
#endif
}

static inline uint64_t mpz_get_u64(const mpz_t z) {
#if ULONG_MAX >= UINT64_MAX
    return (uint64_t)mpz_get_ui(z);
#else
    uint64_t out = 0;
    size_t count = 0;
    mpz_export(&out, &count, -1, sizeof(out), 0, 0, z);
    if (count == 0) return 0;
    if (count > 1) {
        // z ne tient pas sur 64 bits
        throw std::runtime_error("mpz_get_u64: value doesn't fit in uint64_t");
    }
    return out;
#endif
}


static inline uint64_t mpz_get_u64(const mpz_class& z) {
    return mpz_get_u64(z.get_mpz_t());
}


static inline mpz_class mpz_from_u64(uint64_t x) {
    mpz_class r;
    mpz_set_u64(r.get_mpz_t(), x);
    return r;
}

namespace fs = std::filesystem;

static std::string pm1_checkpoint_backend_sidecar(const std::string& checkpoint_file) {
    std::string base = checkpoint_file;
    if (base.size() > 4 && base.compare(base.size() - 4, 4, ".old") == 0) {
        base.resize(base.size() - 4);
    }
    return base + ".backend";
}

static const char* pm1_checkpoint_backend_name(const engine* eng) {
    return eng && eng->is_aevum_backend() ? "aevum" : "marin";
}

static void write_pm1_checkpoint_backend(const std::string& checkpoint_file, const engine* eng) {
    const std::string path = pm1_checkpoint_backend_sidecar(checkpoint_file);
    const std::string temp = path + ".new";
    {
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << pm1_checkpoint_backend_name(eng) << '\n';
    }
    std::error_code ec;
    fs::rename(temp, path, ec);
    if (ec) {
        fs::remove(path, ec);
        ec.clear();
        fs::rename(temp, path, ec);
    }
}

static bool pm1_checkpoint_backend_matches(const std::string& checkpoint_file,
                                           const engine* eng,
                                           std::string* reason = nullptr) {
    const std::string sidecar = pm1_checkpoint_backend_sidecar(checkpoint_file);
    std::ifstream in(sidecar);
    if (!in) {
        // Checkpoints written before v99.5 did not carry a backend marker.
        // They used the Marin register layout by default and must never be
        // injected into Aevum buffers.
        if (eng && eng->is_aevum_backend()) {
            if (reason) *reason = "legacy checkpoint has no backend marker and is assumed to use Marin registers";
            return false;
        }
        return true;
    }

    std::string stored;
    in >> stored;
    const std::string expected = pm1_checkpoint_backend_name(eng);
    if (stored == expected) return true;
    if (reason) *reason = "checkpoint backend is " + stored + ", current backend is " + expected;
    return false;
}

struct PM1Prime95Stage2Result {
    bool success = false;
    bool factor_found = false;
    bool known_factor = false;
    std::string factor;
    std::string json_line;
    int exit_code = -1;
    std::string error;
};

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

static bool p95_read_text_file_local(const fs::path& file, std::string& out) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open()) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

static bool p95_write_text_file_local(const fs::path& file, const std::string& text, std::string& err) {
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

    struct ForcedSetting { const char* key; const char* value; };
    static const ForcedSetting forced[] = {
        {"AllowLowB1", "1"},
        {"Stage1GCD", "-1"},
        {"UsePrimenet", "0"},
        {"Pminus1BestB2", "0"},
        {"ExitWhenOutOfWork", "1"},
    };

    std::vector<std::string> missing_at_top;
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
        if (ch == '"') out += '\\';
        out.push_back(ch);
    }
    out += "\"";
    return out;
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

static bool p95_read_last_non_empty_lines(const fs::path& file, size_t max_lines, std::vector<std::string>& lines_out) {
    lines_out.clear();
    std::ifstream in(file, std::ios::in);
    if (!in.is_open()) return false;
    std::deque<std::string> q;
    std::string line;
    while (std::getline(in, line)) {
        //if (!line.empty() && line.back() == '') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        q.push_back(line);
        while (q.size() > max_lines) q.pop_front();
    }
    if (q.empty()) return false;
    lines_out.assign(q.begin(), q.end());
    return true;
}

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

static bool p95_extract_json_string_field(const std::string& line, const std::string& key, std::string& value) {
    value.clear();
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    size_t end = line.find('"', pos + 1);
    if (end == std::string::npos) return false;
    value = line.substr(pos + 1, end - pos - 1);
    return true;
}

static bool p95_extract_json_first_factor(const std::string& line, std::string& factor) {
    factor.clear();
    const std::string needle = "\"factors\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos = line.find('[', pos + needle.size());
    if (pos == std::string::npos) return false;
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    size_t end = line.find('"', pos + 1);
    if (end == std::string::npos) return false;
    factor = line.substr(pos + 1, end - pos - 1);
    return true;
}

static bool p95_parse_result_json_line(const std::string& line, std::string& status_out, std::string& factor_out) {
    status_out.clear();
    factor_out.clear();
    p95_extract_json_string_field(line, "status", status_out);
    if (!p95_extract_json_first_factor(line, factor_out)) {
        p95_extract_json_string_field(line, "factor", factor_out);
    }
    return !status_out.empty();
}

static PM1Prime95Stage2Result p95_run_pm1_stage2_task(const fs::path& p95_dir,
                                                       const fs::path& p95_exe,
                                                       const fs::path& state_file,
                                                       const std::string& worktodo_line,
                                                       const std::string& log_filename,
                                                       const std::vector<std::string>& known_factors) {
    PM1Prime95Stage2Result result;
    if (p95_dir.empty() || p95_exe.empty()) {
        result.error = "Prime95 path or executable is empty";
        return result;
    }
    if (state_file.empty()) {
        result.error = "Prime95 state file path is empty";
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

    std::string copy_err;
    if (fs::exists(prime_txt)) {
        had_prime_txt = true;
        if (!p95_read_text_file_local(prime_txt, prime_txt_original)) {
            result.error = "Failed to read existing Prime95 prime.txt";
            return result;
        }
        if (!p95_copy_overwrite(prime_txt, prime_txt_backup, copy_err)) {
            result.error = "Failed to backup Prime95 prime.txt: " + copy_err;
            return result;
        }
    }

    const std::string prime_txt_modified = p95_normalize_prime_txt(prime_txt_original);
    if (!p95_write_text_file_local(prime_txt, prime_txt_modified, prime_txt_write_err)) {
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
        wt << worktodo_line << "\n";
        wt.flush();
    }

    std::ostringstream shell;
#ifdef _WIN32
    shell << "cd /d " << p95_shell_quote_win(p95_dir.string())
          << " && " << p95_shell_quote_win(p95_exe.string())
          << " -d > " << p95_shell_quote_win((p95_dir / log_filename).string()) << " 2>&1";
    const std::string cmd = std::string("cmd /C ") + p95_shell_quote_win(shell.str());
#else
    shell << "cd " << p95_shell_quote_posix(p95_dir.string())
          << " && " << p95_shell_quote_posix(p95_exe.string())
          << " -d > " << p95_shell_quote_posix((p95_dir / log_filename).string()) << " 2>&1";
    const std::string cmd = std::string("sh -lc ") + p95_shell_quote_posix(shell.str());
#endif

    auto future_rc = std::async(std::launch::async, [cmd]() {
        return std::system(cmd.c_str());
    });

    const fs::path log_path = p95_dir / log_filename;
    const fs::path results_file = p95_dir / "results.json.txt";
    auto last_progress = std::chrono::steady_clock::now();

    for (;;) {
        if (future_rc.wait_for(std::chrono::milliseconds(250)) == std::future_status::ready) {
            int rc = future_rc.get();
#ifdef _WIN32
            result.exit_code = rc;
#else
            if (rc == -1) result.exit_code = -1;
            else if (WIFEXITED(rc)) result.exit_code = WEXITSTATUS(rc);
            else result.exit_code = rc;
#endif
            break;
        }

        auto now = std::chrono::steady_clock::now();
        if (now - last_progress >= std::chrono::seconds(5)) {
            std::cout << "[PM1] Prime95 Stage 2 still running... showing the last 5 lines of the log:" << std::endl;
            std::vector<std::string> tail_lines;
            if (p95_read_last_non_empty_lines(log_path, 5, tail_lines)) {
                std::cout << "[PM1] ----- log tail begin -----" << std::endl;
                for (const std::string& line : tail_lines) std::cout << line << std::endl;
                std::cout << "[PM1] ----- log tail end -----" << std::endl;
            } else {
                std::cout << "[PM1] (log not available yet)" << std::endl;
            }
            last_progress = now;
        }
    }

    for (int attempt = 0; attempt < 200; ++attempt) {
        if (p95_read_last_non_empty_line(results_file, result.json_line)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    restore_prime_txt();

    if (result.json_line.empty()) {
        std::ostringstream oss;
        oss << "Prime95 did not produce results.json.txt"
            << " (exit_code=" << result.exit_code << ")"
            << " | log=" << (p95_dir / log_filename).string()
            << " | worktodo=" << worktodo_line;
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
    result.known_factor = result.factor_found && !factor.empty() && p95_is_known_factor_string(factor, known_factors);
    result.success = (status == "NF") || (status == "F");

    if (!result.success && result.error.empty()) {
        result.error = "Prime95 returned an unsupported status: " + status;
    }
    return result;
}


int App::runPM1Stage2() {
    using namespace std::chrono;
    bool debug = false;
    //mpz_class B1(static_cast<unsigned long>(options.B1));
    mpz_class B1 = mpz_from_u64(options.B1);
    mpz_class B2 = mpz_from_u64(options.B2);
    //mpz_class B2(static_cast<unsigned long>(options.B2));
    if (guiServer_) {
        std::ostringstream oss;
        oss << "P-1 factoring stage 2";
        guiServer_->setStatus(oss.str());
    }
    if (B2 <= B1) { std::cerr << "Stage 2 error B2 < B1.\n"; 
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Stage 2 error B2 < B1.\n";
                      guiServer_->appendLog(oss.str());
    }
    return -1; }
    
    if (debug) std::cout << "[DEBUG] Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
    std::cout << "\nStart a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nStart a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
                      guiServer_->appendLog(oss.str());
    }
    unsigned long nbEven = evenGapBound(B2);
    size_t limbs = precompute.getN();
    size_t limbBytes = limbs * sizeof(uint64_t);

    cl_int err;
    buffers->Hbuf = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    clEnqueueCopyBuffer(context.getQueue(), buffers->input, buffers->Hbuf, 0, 0, limbBytes, 0, nullptr, nullptr);

    math::Carry carry(context, context.getQueue(), program->getProgram(),
                      precompute.getN(), precompute.getDigitWidth(), buffers->digitWidthMaskBuf);
    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;

    buffers->evenPow.resize(nbEven, nullptr);
    std::cout << "Stage 2: Will precompute " << nbEven << " powers of H^2, H^.." << "." << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Stage 2: Will precompute " << nbEven << " powers of H^2, H^.." << "." << std::endl;
                      guiServer_->appendLog(oss.str());
    }
    nttEngine->forward(buffers->input, 0);
    nttEngine->inverse(buffers->input, 0);
    carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);

    buffers->evenPow[0] = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    nttEngine->copy(buffers->input, buffers->evenPow[0], limbBytes);
    auto ensureEvenPow = [&](unsigned long needIdx) {
        while (buffers->evenPow.size() <= needIdx) {
            size_t kPrev = buffers->evenPow.size() - 1;
            cl_mem buf = clCreateBuffer(context.getContext(),
                                        CL_MEM_READ_WRITE,
                                        limbBytes, nullptr, &err);

            nttEngine->copy(buffers->evenPow[kPrev], buf, limbBytes);
            nttEngine->mulInPlace(buf, buffers->evenPow[0], carry, limbBytes);
            buffers->evenPow.push_back(buf);
        }
    };
    int pct = -1;
    std::cout << "Precomputing H powers: 0%" << std::flush;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Precomputing H powers..." << std::endl;
                      guiServer_->appendLog(oss.str());
    }
    for (unsigned long k = 1; k < nbEven; ++k) {
        buffers->evenPow[k] = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
        nttEngine->copy(buffers->evenPow[k - 1], buffers->evenPow[k], limbBytes);
        nttEngine->mulInPlace(buffers->evenPow[k], buffers->evenPow[0], carry, limbBytes);
        
        int newPct = int((k + 1) * 100 / nbEven);
        if (newPct > pct) { pct = newPct; std::cout << "\rPrecomputing H powers: " << pct << "%" << std::flush; }
    }
    for (unsigned long k = 0; k < nbEven; ++k) {
        
        nttEngine->copy(buffers->evenPow[k], buffers->input, limbBytes);
        nttEngine->forward_simple(buffers->input,0);
        nttEngine->copy(buffers->input, buffers->evenPow[k], limbBytes);
        
    }

    std::cout << "\rPrecomputing H powers: 100%" << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\rPrecomputing H powers: 100%" << std::endl;
                      guiServer_->appendLog(oss.str());
    }
   

    buffers->Hq = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    nttEngine->copy(buffers->Hbuf, buffers->input, limbBytes);

    mpz_class p_prev;
    mpz_class p;
    mpz_nextprime(p_prev.get_mpz_t(), B1.get_mpz_t());
    p = p_prev;

    size_t bitlen = mpz_sizeinbase(p.get_mpz_t(), 2);
    for (int64_t i = static_cast<int64_t>(bitlen) - 2; i >= 0; --i) {
        nttEngine->forward(buffers->input, 0);
        nttEngine->inverse(buffers->input, 0);
        carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);
        if (mpz_tstbit(p.get_mpz_t(), static_cast<mp_bitcnt_t>(i))) {
            nttEngine->mulInPlace(buffers->input, buffers->Hbuf, carry, limbBytes);
        }
    }
    nttEngine->copy(buffers->input, buffers->Hq, limbBytes);

    buffers->Qbuf = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    std::vector<uint64_t> one(limbs, 0ULL); one[0] = 1ULL;
    clEnqueueWriteBuffer(context.getQueue(), buffers->Qbuf, CL_TRUE, 0, limbBytes, one.data(), 0, nullptr, nullptr);

    buffers->tmp = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);

    size_t     idx     = 0;
    //uint64_t resumeIdx = 0;
    uint64_t resumeIdx = backupManager.loadStatePM1S2(buffers->Hq, buffers->Qbuf, limbBytes);

    mpz_nextprime(p_prev.get_mpz_t(), B1.get_mpz_t());
    p = p_prev;
    idx = 0;
    for (; idx < resumeIdx; ++idx) {
        p_prev = p;
        mpz_nextprime(p.get_mpz_t(), p.get_mpz_t());
    }


    std::cout << "\r[DEBUG] p_prev=" << p_prev << std::endl;
    std::cout << "\r[DEBUG] p=" << p << std::endl;
    std::cout << "\r[DEBUG] idx=" << idx << std::endl;
    
    
    size_t totalPrimes = primeCountApprox(B1, B2);

    timer.start(); timer2.start();
    auto start = high_resolution_clock::now();
    auto lastDisplay = start;
    //auto lastBackup  = start;

    for (; p <= B2; ++idx) {
        if (idx) {
            mpz_class d = p - p_prev;
            uint64_t gap = mpz_get_u64(d.get_mpz_t());
            unsigned long idxGap = (unsigned long)((gap >> 1) - 1);

            ensureEvenPow(idxGap);
            
            nttEngine->forward_simple(buffers->Hq, 0);
            nttEngine->pointwiseMul(buffers->Hq, buffers->evenPow[idxGap]);
            nttEngine->inverse_simple(buffers->Hq, 0);
            carry.carryGPU(buffers->Hq, buffers->blockCarryBuf, limbBytes);
            
        }

        nttEngine->copy(buffers->Hq, buffers->tmp, limbBytes);
        nttEngine->subOne(buffers->tmp);
        nttEngine->forward_simple(buffers->tmp, 0);
        nttEngine->forward_simple(buffers->Qbuf, 0);
        nttEngine->pointwiseMul(buffers->Qbuf, buffers->tmp);
        nttEngine->inverse_simple(buffers->Qbuf, 0);
        carry.carryGPU(buffers->Qbuf, buffers->blockCarryBuf, limbBytes);
        
        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
            double done = static_cast<double>(idx + 1);
            double doneSinceResume = static_cast<double>(idx + 1 - resumeIdx);
            double percent = totalPrimes ? done / static_cast<double>(totalPrimes) * 100.0 : 0.0;
            double elapsedSec = duration<double>(now - start).count();
            double ips = doneSinceResume > 0 ? doneSinceResume / elapsedSec : 0.0;

            double remaining = totalPrimes > done ? static_cast<double>(totalPrimes) - done : 0.0;
            double etaSec = ips > 0.0 ? remaining / ips : 0.0;
            int days = static_cast<int>(etaSec) / 86400;
            int hours = (static_cast<int>(etaSec) % 86400) / 3600;
            int minutes = (static_cast<int>(etaSec) % 3600) / 60;
            int seconds = static_cast<int>(etaSec) % 60;
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << percent << "% | "
                      << "prime: " << mpz_get_u64(p) << " | "
                      << "Iter: " << (idx + 1) << " | "
                      << "Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s | "
                      << "IPS: " << std::fixed << std::setprecision(2) << ips << " | "
                      << "ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                      << std::endl;
            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Progress: " << std::fixed << std::setprecision(2) << percent << "% | "
                      << "prime: " << mpz_get_u64(p) << " | "
                      << "Iter: " << (idx + 1) << " | "
                      << "Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s | "
                      << "IPS: " << std::fixed << std::setprecision(2) << ips << " | "
                      << "ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                      << std::endl;
                      guiServer_->appendLog(oss.str());
            }
            lastDisplay = now;
        }
        /*if (duration_cast<seconds>(now - lastBackup).count() >= 180) {
            backupManager.saveStatePM1S2(Hq, Qbuf, idx, limbBytes);
            lastBackup = now;
        }*/
        if (options.iterforce2 > 0 && (idx + 1) % options.iterforce2 == 0) {
            char dummy;
            clEnqueueReadBuffer(context.getQueue(), buffers->input, CL_TRUE, 0, sizeof(dummy), &dummy, 0, nullptr, nullptr);
        }


        p_prev = p;
        mpz_nextprime(p.get_mpz_t(), p.get_mpz_t());
        /*if (interrupted) {
            clFinish(context.getQueue());
            backupManager.saveStatePM1S2(Hq, Qbuf, idx, limbBytes);
            std::cout << "\r[DEBUG] p_prev=" << p_prev << std::endl;
            std::cout << "\r[DEBUG] p=" << p << std::endl;
             std::cout << "\r[DEBUG] idx=" << idx << std::endl;
            return 0;
        }*/
        if (interrupted) {
            clFinish(context.getQueue());
            backupManager.saveStatePM1S2(buffers->Hq, buffers->Qbuf, idx, limbBytes);
            return 0;
        }

    }

    std::vector<uint64_t> hostQ(limbs);
    clEnqueueReadBuffer(context.getQueue(), buffers->Qbuf, CL_TRUE, 0, limbBytes, hostQ.data(), 0, nullptr, nullptr);
    carry.handleFinalCarry(hostQ, precompute.getDigitWidth());
    mpz_class Q = util::vectToMpz(hostQ, precompute.getDigitWidth(), Mp);
    mpz_class g; mpz_gcd(g.get_mpz_t(), Q.get_mpz_t(), Mp.get_mpz_t());
    bool found = g != 1 && g != Mp;
    std::string filename = "stage2_result_B2_" + B2.get_str() +
                        "_p_" + std::to_string(options.exponent) + ".txt";

    if (found) {
        char* s = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "B2=" + B2.get_str() + "  factor=" + std::string(s));
        std::cout << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n';
        if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n';
                      guiServer_->appendLog(oss.str());
            }
        options.knownFactors.push_back(std::string(s));
        std::free(s);
    } else {
        writeStageResult(filename, "No factor P-1 up to B2=" + B2.get_str());
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
        if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
                      guiServer_->appendLog(oss.str());
        }
    }

    /*
    if (found) {
        char* s = mpz_get_str(nullptr, 10, g);
        std::cout << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n';
        std::free(s);
    } else {
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
    }*/


    //backupManager.clearState();
    return found ? 0 : 1;
}

int App::runPM1() {

    uint64_t B1 = options.B1;
    if (guiServer_) {
        std::ostringstream oss;
        oss << "P-1 factoring stage 1";
        guiServer_->setStatus(oss.str());
    }
    std::cout << "Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl;
                      guiServer_->appendLog(oss.str());
            }
    mpz_class E = backupManager.loadExponent();
    if(E==0){    
        if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Building E....";
                      guiServer_->appendLog(oss.str());
        }
        E = buildE(B1);
        E *= mpz_class(2) * mpz_from_u64(options.exponent);


    }
    
    //std::cout << "[DEBUG] E=" << E << std::endl;
    mp_bitcnt_t bits = mpz_sizeinbase(E.get_mpz_t(), 2);
    std::vector<uint64_t> x(precompute.getN(), 0ULL);
    uint64_t resumeIter = backupManager.loadState(x);
    if(resumeIter==0){
        x[0] = 1ULL;
        resumeIter = bits;
    }
    buffers->input = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, x.size() * sizeof(uint64_t), x.data(), nullptr);

    math::Carry carry(context, context.getQueue(), program->getProgram(), precompute.getN(), precompute.getDigitWidth(), buffers->digitWidthMaskBuf);


    timer.start();
    timer2.start();
    auto startTime  = high_resolution_clock::now();
    auto lastDisplay = startTime;
    interrupted.store(false, std::memory_order_relaxed);

    uint64_t startIter = resumeIter;
    uint64_t lastIter = resumeIter;
    backupManager.saveState(buffers->input, resumeIter,&E);
    spinner.displayProgress(
                    bits-resumeIter,
                    bits,
                    timer.elapsed(),
                    timer2.elapsed(),
                    options.exponent,
                    resumeIter,
                    resumeIter,
                    "", 
                    guiServer_ ? guiServer_.get() : nullptr
                );
    auto start_sys = std::chrono::system_clock::now();
    for (mp_bitcnt_t i = resumeIter; i > 0; --i) {
        lastIter = i;
        if (interrupted) {
            std::cout << "\nInterrupted signal received\n " << std::endl;
            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nInterrupted signal received\n " << std::endl;
                      guiServer_->appendLog(oss.str());
            }
            clFinish(context.getQueue());
            backupManager.saveState(buffers->input, lastIter-1, &E);
            //backupManager.saveState(buffers->input, lastIter);
            //std::cout << "\nInterrupted by user, state saved at iteration "
            //        << lastIter << std::endl;
            return 0;
        }
        nttEngine->forward(buffers->input, 0);
        nttEngine->inverse(buffers->input, 0);
        carry.carryGPU(buffers->input, buffers->blockCarryBuf, precompute.getN() * sizeof(uint64_t));



        if (mpz_tstbit(E.get_mpz_t(), i - 1)) {
            carry.carryGPU3(buffers->input, buffers->blockCarryBuf, precompute.getN() * sizeof(uint64_t));
            
        }
        if ((options.iterforce > 0 && (i+1)%options.iterforce == 0 && i>0) || (((i+1)%options.iterforce == 0))) { 
            
            //if((i+1)%100000 != 0){

                char dummy;
                clEnqueueReadBuffer(
                        context.getQueue(),
                        buffers->input,
                        CL_TRUE, 0,
                        sizeof(dummy),
                        &dummy,
                        0, nullptr, nullptr
                    );
                
            
            //}

            
        }
        
        auto now = high_resolution_clock::now();
        if ((((now - lastDisplay >= seconds(180)))) ) {
                    backupManager.saveState(buffers->input, lastIter-1);
        }
        if ((((now - lastDisplay >= seconds(10)))) ) {
                std::string res64_x;
                

                spinner.displayProgress(
                    bits-i-1,
                    bits,
                    timer.elapsed(),
                    timer2.elapsed(),
                    options.exponent,
                    resumeIter,
                    startIter,
                    res64_x, 
                    guiServer_ ? guiServer_.get() : nullptr
                );
                timer2.start();
                lastDisplay = now;
                resumeIter = bits-i-1;
            }
        //clFinish(context.getQueue());

    }
    //return 0;
    std::string res64_x;
    spinner.displayProgress(
        bits,
        bits,
        timer.elapsed(),
        timer2.elapsed(),
        options.exponent,
        resumeIter,
        startIter,
        res64_x, 
        guiServer_ ? guiServer_.get() : nullptr
    );
    backupManager.saveState(buffers->input, lastIter, &E);


    std::cout << "\nStart get result from GPU" << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nStart get result from GPU" << std::endl;
                      guiServer_->appendLog(oss.str());
            }
    std::vector<uint64_t> hostData(precompute.getN());

    clEnqueueReadBuffer(context.getQueue(), buffers->input, CL_TRUE, 0,
                        hostData.size() * sizeof(uint64_t), hostData.data(),
                        0, nullptr, nullptr);
    std::cout << "Handle final carry start" << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Handle final carry start" << std::endl;
                      guiServer_->appendLog(oss.str());
            }

    carry.handleFinalCarry(hostData, precompute.getDigitWidth());
    std::cout << "vectToResidue start" << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "vectToResidue start" << std::endl;
                      guiServer_->appendLog(oss.str());
            }
    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    mpz_class X = util::vectToMpz(hostData, precompute.getDigitWidth(), Mp);
    auto fmt = [](const std::chrono::system_clock::time_point& tp){
        using namespace std::chrono;
        auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
        std::time_t tt = system_clock::to_time_t(tp);
        std::tm tmv{};
        #if defined(_WIN32)
        gmtime_s(&tmv, &tt);
        #else
        std::tm* tmp = std::gmtime(&tt);
        if (tmp) tmv = *tmp;
        #endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
        std::ostringstream s;
        s << buf << '.' << std::setw(3) << std::setfill('0') << (int)(ms.count());
        return s.str();
    };
    auto end_sys = std::chrono::system_clock::now();
    
    std::string ds = fmt(start_sys);
    std::string de = fmt(end_sys);
    if(options.resume){
        
        writeEcmResumeLine("resume_p" + std::to_string(options.exponent) + "_B1_" +
                    std::to_string(options.B1) + ".save",
                    options.B1, options.exponent, X);
        convertEcmResumeToPrime95("resume_p" + std::to_string(options.exponent) + "_B1_" +
                    std::to_string(options.B1) + ".save","resume_p" + std::to_string(options.exponent) + "_B1_" +
                    std::to_string(options.B1) + ".p95", ds, de);
                    
    }
    
    //std::cout << "digitWidths = ";
    //for (int w : precompute.getDigitWidth()) std::cout << w << " ";
    //std::cout << "\n";
    //gmp_printf("X final  = %Zd\n", X);

    X -= 1;

    mpz_class g;
    mpz_gcd(g.get_mpz_t(), X.get_mpz_t(), Mp.get_mpz_t());


/*
    mpz_t Mp;  mpz_init(Mp);
    mpz_ui_pow_ui(Mp, 2, options.exponent);
    mpz_sub_ui(Mp, Mp, 1);

    mpz_t r;   mpz_init(r);
    vectToResidue(r, hostData, precompute.getDigitWidth(), Mp);
    std::cout << "\n GCD start\n" << std::endl;

    mpz_sub_ui(r, r, 1);
    mpz_t g;   mpz_init(g);
    mpz_gcd(g, r, Mp);*/

    //gmp_printf("GCD(x - 1, 2^%u - 1) = %Zd\n", options.exponent, g);

    bool factorFound = g != 1 && g != Mp;

    std::string filename = "stage1_result_B1_" + std::to_string(B1) +
                        "_p_" + std::to_string(options.exponent) + ".txt";

    if (factorFound) {
        char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "B1=" + std::to_string(B1) + "  factor=" + std::string(fstr));
        std::free(fstr);
    } else {
        writeStageResult(filename, "No factor up to B1=" + std::to_string(B1));
    }


    if (factorFound) {
        char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
        std::cout << "\nP-1 factor stage 1 found: " << fstr << std::endl;
        if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "\nP-1 factor stage 1 found: " << fstr << std::endl;
                      guiServer_->appendLog(oss.str());
            }
        options.knownFactors.push_back(std::string(fstr));
        std::free(fstr);
        std::cout << "\n";
        if(options.B2>0){
            runPM1Stage2();
        }
//        else{
            //backupManager.clearState();
//        }
        //return 0;
    }
    else{
        std::cout << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl;
        if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl;
                      guiServer_->appendLog(oss.str());
            }
        if(options.B2>0){
            runPM1Stage2();
        }
    }
/*    else{
            backupManager.clearState();
    }
    backupManager.clearState();*/
    std::string json = io::JsonBuilder::generate(
        options,
        static_cast<int>(context.getTransformSize()),
        false,
        "",
        ""
    );
    std::cout << "Manual submission JSON:\n" << json << "\n";
   /*if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "Manual submission JSON:\n" << json << "\n";
                      guiServer_->appendLog(oss.str());
            }*/
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, options.mode, json);
    wm.appendToResultsTxt(json);
    
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
    return 1;
}


static uint64_t isqrt_u64(uint64_t x) {
  long double r = std::sqrt((long double)x);
  uint64_t y = (uint64_t)r;
  while ((y+1) > 0 && (y+1) <= x / (y+1)) ++y;
  while (y > 0 && y > x / y) --y;
  return y;
}

static std::vector<uint32_t> sieve_base_primes(uint32_t limit) {
  std::vector<uint8_t> isPrime(limit + 1, 1);
  isPrime[0] = isPrime[1] = 0;
  for (uint32_t p = 2; (uint64_t)p * p <= limit; ++p) {
    if (!isPrime[p]) continue;
    for (uint64_t m = (uint64_t)p * p; m <= limit; m += p) isPrime[(size_t)m] = 0;
  }
  std::vector<uint32_t> primes;
  primes.reserve(limit / 10);
  for (uint32_t i = 2; i <= limit; ++i) if (isPrime[i]) primes.push_back(i);
  return primes;
}

static void segmented_primes_odd(uint64_t low, uint64_t high,
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

    uint64_t start = (low + p - 1) / p * p;
    if (start < p2) start = p2;
    if ((start & 1ULL) == 0) start += p;

    for (uint64_t x = start; x <= high; x += (p << 1)) {
      const uint64_t idx = (x - low) >> 1;
      isPrime[(size_t)idx] = 0;
    }
  }

  out.reserve((size_t)(nOdds / 10));
  for (uint64_t i = 0; i < nOdds; ++i) {
    if (!isPrime[(size_t)i]) continue;
    out.push_back(low + (i << 1));
  }
}

struct S2Entry {
  uint32_t k_rel;  // k - k0
  uint32_t e;      // q = k*D + e  (e < D)
};
/*
static void primes_to_ke(const std::vector<uint64_t>& primes,
                         uint32_t D,
                         uint64_t k0,
                         std::vector<S2Entry>& out) {
  out.clear();
  out.reserve(primes.size());
  for (uint64_t q : primes) {
    const uint64_t k = q / (uint64_t)D;
    const uint32_t e = (uint32_t)(q - k * (uint64_t)D);
    out.push_back(S2Entry{ (uint32_t)(k - k0), e });
  }
}*/




static bool read_pm1_resume_x_hex(const std::string& path,
                                  uint64_t expectedB1,
                                  uint32_t expectedP,
                                  mpz_class& Xout)
{
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    if (!std::getline(in, line)) return false;

    auto get_field = [&](const std::string& key)->std::string {
        const std::string pat = key + "=";
        size_t a = line.find(pat);
        if (a == std::string::npos) return {};
        a += pat.size();
        size_t b = line.find(';', a);
        if (b == std::string::npos) b = line.size();
        while (a < b && std::isspace((unsigned char)line[a])) ++a;
        while (b > a && std::isspace((unsigned char)line[b-1])) --b;
        return line.substr(a, b-a);
    };

    const std::string b1s = get_field("B1");
    if (!b1s.empty()) {
        try {
            uint64_t gotB1 = std::stoull(b1s);
            if (gotB1 != expectedB1) {
                std::cerr << "Stage 2 ultralowmem: resume B1 mismatch in " << path
                          << " (got " << gotB1 << ", expected " << expectedB1 << ").\n";
                return false;
            }
        } catch (...) { return false; }
    }

    const std::string ns = get_field("N");
    const std::string expectedN = "2^" + std::to_string(expectedP) + "-1";
    if (!ns.empty() && ns != expectedN) {
        std::cerr << "Stage 2 ultralowmem: resume N mismatch in " << path
                  << " (got " << ns << ", expected " << expectedN << ").\n";
        return false;
    }

    std::string xs = get_field("X");
    if (xs.empty()) return false;
    if (xs.size() >= 2 && xs[0] == '0' && (xs[1] == 'x' || xs[1] == 'X')) xs = xs.substr(2);
    if (xs.empty()) return false;
    if (mpz_set_str(Xout.get_mpz_t(), xs.c_str(), 16) != 0) return false;
    return true;
}


// Forward declarations for loading PM1 Stage-1 textual resume files without creating
// a temporary OpenCL engine.  Used by the ultralowmem Stage-2 GPU path.
static bool load_pm1_s1_from_save(const std::string& path, uint64_t& B1_out, uint32_t& p_out, mpz_class& X_out);
static bool load_pm1_s1_from_p95(const std::string& path, uint64_t& B1_out, uint32_t& p_out, mpz_class& X_out);
static inline void mpz_mul_u64(mpz_class& a, uint64_t x);

int App::runPM1Stage2MarinLowMem() {
    using namespace std::chrono;

    const uint64_t B1u = options.B1, B2u = options.B2;
    const uint64_t stage2Low = (options.B2Start > 0) ? options.B2Start : B1u;
    const uint32_t pexp = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;

    if (B2u <= stage2Low) {
        std::cerr << "Stage 2 error B2 <= stage2 lower bound. B1/resume=" << B1u
                  << " stage2Low=" << stage2Low << " B2=" << B2u << "\n";
        return -1;
    }
    if (options.pm1_s2_resume2reg && stage2Low < B1u) {
        std::cerr << "Stage 2 resume2reg error: -b2start must be >= -b1 resume bound. B1/resume="
                  << B1u << " stage2Low=" << stage2Low << "\n";
        return -1;
    }

    std::cout << "\nStart a P-1 factoring : Stage 2 Resume B1 = " << B1u
              << ", Stage2 range = (" << stage2Low << ", " << B2u << "]" << std::endl;
    std::cout << "[PM1] Low-memory Stage 2 enabled. ";
    if (options.pm1_ultralowmem && options.pm1_s2_resume2reg) {
        std::cout << "Using TRUE resume2reg mode: load Stage-1 H and compute H^Q with 2 GPU registers.\n";
    } else if (options.pm1_ultralowmem) {
        std::cout << "Using legacy 1-register GPU product-exponent Stage 2 (base-3 recompute, no baby-table BSGS).\n";
    } else {
        std::cout << "Using 3 GPU registers and streamed product accumulator (no baby-table BSGS).\n";
    }

    // Legacy Stage 2 path checkpoint loading.
    // Important: the new -pm1-s2-resume2reg path below is a TRUE resume path:
    // it loads H from resume_p<p>_B1_<B1>.p95/.save and computes H^Q.
    // Therefore the old one-register ultralowmem message/recompute path must not
    // run when options.pm1_s2_resume2reg is set.
    std::vector<char> hData;
    if (options.pm1_ultralowmem && options.pm1_s2_resume2reg) {
        std::cout << "[PM1] Stage 2 resume2reg selected: skipping legacy one-register recompute path.\n";
    } else if (options.pm1_ultralowmem) {
        std::cout << "[PM1] Ultra-low-memory Stage 2 legacy one-register mode: recomputes "
                  << "3^(E*Q) directly on GPU with the fast3 path.\n";
    } else {
        const size_t s1Regs = 3u;
        auto load_h_from = [&](const std::string& file)->int {
            engine* e = nullptr;
            try {
                e = engine::create_gpu(pexp, s1Regs, (size_t)options.device_id, verbose);
            } catch (const std::exception& ex) {
                std::cerr << "Stage 2 lowmem: cannot allocate temporary " << s1Regs
                          << "-register checkpoint loader: " << ex.what() << "\n";
                return -2;
            }
            File f(file);
            if (!f.exists()) { delete e; return -1; }
            int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) { delete e; return -2; }
            if (version != 3) { delete e; return -2; }
            uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) { delete e; return -2; }
            if (rp != pexp) { delete e; return -2; }
            uint32_t ri = 0; double et = 0.0;
            if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) { delete e; return -2; }
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) { delete e; return -2; }

            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) { delete e; return -2; }
            if (!e->set_checkpoint(data)) { delete e; return -2; }
            hData.resize(e->get_register_data_size());
            if (!e->get_data(hData, (engine::Reg)0)) { delete e; return -2; }
            delete e;
            return 0;
        };

        std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
        std::string ckpt_file = ck.str();
        int rr = load_h_from(ckpt_file);
        if (rr < 0) rr = load_h_from(ckpt_file + ".old");
        if (rr != 0) {
            std::cerr << "Stage 2 lowmem: cannot load PM1 Stage 1 checkpoint " << ckpt_file
                      << " using " << s1Regs << " register(s).\n";
            return -2;
        }
        std::cout << "[PM1] Low-memory Stage 2 loaded H from " << ckpt_file
                  << " through the Marin checkpoint path.\n";
    }

    const uint64_t root = isqrt_u64(B2u);
    const std::vector<uint32_t> basePrimes = sieve_base_primes((uint32_t)root);
    std::vector<uint64_t> primes;
    segmented_primes_odd(stage2Low + 1, B2u, basePrimes, primes);
    if (primes.empty()) {
        std::cout << "\nNo factor P-1 (stage 2) in range (" << stage2Low << ", " << B2u
                  << "] (no primes in range)\n";
        return 1;
    }

    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    auto t0 = high_resolution_clock::now();
    auto lastDisplay = high_resolution_clock::now();

    if (options.pm1_ultralowmem && options.pm1_s2_resume2reg) {
        // v42 RTX-safe true Stage 2 from an existing Stage-1 resume.
        // This is not the old 1-register recompute from base 3.  It loads H from
        // resume_p<exp>_B1_<B1>.p95/.save and computes:
        //      H <- H^Q, Q = prod_{stage2Low < q <= B2, q prime} q
        // using only two GPU registers.  It is slower than BSGS, but it is the
        // realistic no-baby-table path that fits 10GB GPUs such as RTX 3080.
        static constexpr size_t RSTATE = 0; // current H/product state, digits
        static constexpr size_t RBASE  = 1; // multiplicand copy of current state for each Q chunk

        auto ends_with = [](const std::string& x, const std::string& suf)->bool{
            return x.size() >= suf.size() && x.compare(x.size() - suf.size(), suf.size(), suf) == 0;
        };

        std::string basePath = options.pm1_extend_save_path;
        if (basePath.empty()) {
            basePath = "resume_p" + std::to_string(options.exponent) +
                       "_B1_" + std::to_string(B1u);
        }

        std::string resumeSave = basePath;
        std::string resumeP95  = basePath;
        if (ends_with(basePath, ".save")) {
            resumeSave = basePath;
            resumeP95  = basePath.substr(0, basePath.size() - 5) + ".p95";
        } else if (ends_with(basePath, ".p95")) {
            resumeP95  = basePath;
            resumeSave = basePath.substr(0, basePath.size() - 4) + ".save";
        } else {
            resumeSave += ".save";
            resumeP95  += ".p95";
        }

        uint64_t B1_file = 0;
        uint32_t p_file = 0;
        mpz_class H_old;
        std::string usedPath;
        if (load_pm1_s1_from_save(resumeSave, B1_file, p_file, H_old)) {
            usedPath = resumeSave;
        } else if (load_pm1_s1_from_p95(resumeP95, B1_file, p_file, H_old)) {
            usedPath = resumeP95;
        } else {
            std::cerr << "[PM1] Stage2 resume2reg: cannot load Stage-1 state from "
                      << resumeSave << " nor " << resumeP95 << "\n";
            return -2;
        }
        if (p_file != pexp) {
            std::cerr << "[PM1] Stage2 resume2reg: resume exponent mismatch: file p="
                      << p_file << ", expected p=" << pexp << "\n";
            return -2;
        }
        if (B1_file != B1u) {
            std::cerr << "[PM1] Stage2 resume2reg: resume B1 mismatch: file B1="
                      << B1_file << ", expected B1=" << B1u << "\n";
            return -2;
        }

        uint64_t chunkBitLimit = 200000ULL;
        if (const char* env = std::getenv("PRMERS_PM1_S2_CHUNK_BITS")) {
            try {
                uint64_t v = std::stoull(env);
                if (v >= 1024ULL) chunkBitLimit = v;
            } catch (...) {}
        }
        if (options.max_e_bits > 0) chunkBitLimit = std::min<uint64_t>(chunkBitLimit, options.max_e_bits);
        chunkBitLimit = std::max<uint64_t>(chunkBitLimit, 1024ULL);

        std::cout << "[PM1] Ultra-low-memory Stage 2 resume2reg TRUE mode.\n";
        std::cout << "[PM1] Loaded Stage-1 state H from " << usedPath << "\n";
        std::cout << "[PM1] Method: H <- H^prod(primes in (stage2Low,B2]) with 2 GPU registers; no base-3 recompute.\n";
        std::cout << "[PM1] Resume B1=" << B1u << " | Stage2 lower/start=" << stage2Low
                  << " | Stage2 upper B2=" << B2u << "\n";
        std::cout << "[PM1] 2-reg exponentiation detail: RBASE stores multiplicand(H); RSTATE is then reset to 1 as accumulator.\n";
        std::cout << "[PM1] Prime count=" << primes.size()
                  << " | product chunk limit=" << chunkBitLimit << " bits\n";

        engine* eng = nullptr;
        try {
            std::cout << "[PM1] Allocating Stage 2 resume2reg GPU engine with 2 registers..." << std::flush;
            eng = engine::create_gpu(pexp, 2, (size_t)options.device_id, verbose);
            std::cout << " done.\n";
        } catch (const std::exception& ex) {
            std::cout << " failed.\n";
            std::cerr << "[PM1] Stage2 resume2reg: 2-register GPU allocation failed: "
                      << ex.what() << "\n";
            return -2;
        }

        try {
            std::cout << "[PM1] Loading H into RSTATE using lowmem streamed set_mpz...\n";
            // engine::set_mpz takes a const mpz_t& (GMP array reference),
            // while mpz_class::get_mpz_t() is an mpz_ptr. Use a short-lived
            // mpz_t copy so the code compiles cleanly on GCC/Clang.
            mpz_t H_upload;
            mpz_init_set(H_upload, H_old.get_mpz_t());
            eng->set_mpz((engine::Reg)RSTATE, H_upload);
            mpz_clear(H_upload);
            // Free the host mpz payload before the long GPU loop.
            H_old = 0;
        } catch (const std::exception& ex) {
            std::cerr << "[PM1] Stage2 resume2reg: failed to upload H: " << ex.what() << "\n";
            delete eng;
            return -2;
        }

        const auto texp0 = high_resolution_clock::now();
        auto last = high_resolution_clock::now();
        size_t primeIndex = 0;
        uint64_t chunkNo = 0;
        uint64_t totalBitsDone = 0;
        bool stopAfterChunk = false;

        uint64_t progressSecs = 10;
        if (const char* env = std::getenv("PRMERS_PM1_S2_PROGRESS_SECS")) {
            try {
                uint64_t v = std::stoull(env);
                if (v >= 1ULL) progressSecs = v;
            } catch (...) {}
        }
        uint64_t progressBits = 32;
        if (const char* env = std::getenv("PRMERS_PM1_S2_PROGRESS_BITS")) {
            try {
                uint64_t v = std::stoull(env);
                if (v >= 1ULL) progressBits = v;
            } catch (...) {}
        }

        bool syncProgress = true;
        if (const char* env = std::getenv("PRMERS_PM1_S2_SYNC_PROGRESS")) {
            try {
                uint64_t v = std::stoull(env);
                syncProgress = (v != 0ULL);
            } catch (...) {}
        }
        std::cout << "[PM1] Stage2 resume2reg progress: newline report every "
                  << progressSecs << "s or " << progressBits
                  << " exponent bits. Override with PRMERS_PM1_S2_PROGRESS_SECS/BITS.\n";
        std::cout << "[PM1] Stage2 resume2reg progress timing is "
                  << (syncProgress ? "SYNCHRONIZED" : "ENQUEUE-ONLY")
                  << ". Override with PRMERS_PM1_S2_SYNC_PROGRESS=0/1.\n";

        while (primeIndex < primes.size()) {
            const size_t chunkPrimeStartIndex = primeIndex;
            const uint64_t chunkPrimeStart = primes[primeIndex];
            mpz_class Q(1);
            mp_bitcnt_t qbits = 1;

            while (primeIndex < primes.size()) {
                mpz_class cand = Q;
                mpz_mul_u64(cand, primes[primeIndex]);
                const mp_bitcnt_t cbits = mpz_sizeinbase(cand.get_mpz_t(), 2);
                if (primeIndex > chunkPrimeStartIndex && cbits > chunkBitLimit) break;
                Q = cand;
                qbits = cbits;
                ++primeIndex;
                if (cbits >= chunkBitLimit) break;
            }

            const uint64_t chunkPrimeEnd = primes[primeIndex - 1];
            ++chunkNo;
            const auto chunkStartTime = high_resolution_clock::now();
            uint64_t lastChunkBitsPrinted = 0;
            std::cout << "\n[PM1] Stage2 resume2reg chunk " << chunkNo
                      << " | primes " << chunkPrimeStart << ".." << chunkPrimeEnd
                      << " | count=" << (primeIndex - chunkPrimeStartIndex)
                      << " | Q bits=" << (unsigned long long)qbits << "\n";

            // Compute RSTATE <- RSTATE^Q with only two registers.
            // RBASE becomes an independent transformed multiplicand copy of the
            // current RSTATE (the loaded Stage-1 H for chunk 1, then H^previousQ
            // for later chunks).  RSTATE is then reset to 1 only as the normal
            // square-and-multiply accumulator.  This is not a base-3 recompute.
            std::cout << "[PM1] Preparing chunk base: copy current RSTATE/H into RBASE multiplicand, then reset RSTATE accumulator to 1.\n";
            eng->set_multiplicand((engine::Reg)RBASE, (engine::Reg)RSTATE);
            eng->set((engine::Reg)RSTATE, 1u);

            for (mp_bitcnt_t i = qbits; i > 0; --i) {
                eng->square_mul((engine::Reg)RSTATE);
                if (mpz_tstbit(Q.get_mpz_t(), i - 1)) eng->mul((engine::Reg)RSTATE, (engine::Reg)RBASE);

                ++totalBitsDone;
                auto now = high_resolution_clock::now();
                const uint64_t doneBitsInChunk = (uint64_t)(qbits - i + 1);
                const bool printByTime = duration_cast<seconds>(now - last).count() >= (long long)progressSecs;
                const bool printByBits = doneBitsInChunk == 1 ||
                                         (doneBitsInChunk - lastChunkBitsPrinted) >= progressBits ||
                                         doneBitsInChunk == (uint64_t)qbits;
                if (printByTime || printByBits) {
                    // OpenCL command queues are asynchronous.  Without a finish here,
                    // progress reports measure enqueue speed and then the process
                    // appears to “block” later at get_mpz/GCD.  Synchronize before
                    // timing so IPS/ETA are based on completed GPU work.
                    if (syncProgress) {
                        eng->sync();
                        now = high_resolution_clock::now();
                    }
                    const double pctChunk = 100.0 * double(doneBitsInChunk) / double(qbits);
                    const double pctPrime = 100.0 * double(primeIndex) / double(primes.size());
                    const double elapsedTotal = duration<double>(now - texp0).count();
                    const double elapsedChunk = duration<double>(now - chunkStartTime).count();
                    const double ipsTotal = elapsedTotal > 0.0 ? double(totalBitsDone) / elapsedTotal : 0.0;
                    const double ipsChunk = elapsedChunk > 0.0 ? double(doneBitsInChunk) / elapsedChunk : 0.0;
                    const double remainChunk = ipsChunk > 0.0 ? double((uint64_t)qbits - doneBitsInChunk) / ipsChunk : 0.0;
                    const uint64_t etaSec = remainChunk > 0.0 ? (uint64_t)(remainChunk + 0.5) : 0ULL;
                    const uint64_t etaH = etaSec / 3600ULL;
                    const uint64_t etaM = (etaSec % 3600ULL) / 60ULL;
                    const uint64_t etaS = etaSec % 60ULL;
                    std::cout << "[PM1] Stage2 chunk " << chunkNo
                              << " progress " << doneBitsInChunk << "/" << (uint64_t)qbits
                              << " bits (" << std::fixed << std::setprecision(2) << pctChunk << "%)"
                              << " | chunkIPS=" << std::fixed << std::setprecision(2) << ipsChunk
                              << " | totalIPS=" << std::fixed << std::setprecision(2) << ipsTotal
                              << " | ETA chunk=" << etaH << "h" << etaM << "m" << etaS << "s"
                              << " | primes=" << std::fixed << std::setprecision(2) << pctPrime << "%"
                              << " | totalBits=" << totalBitsDone
                              << "\n" << std::flush;
                    last = now;
                    lastChunkBitsPrinted = doneBitsInChunk;
                }

                if (interrupted && !stopAfterChunk) {
                    std::cout << "\n[PM1] Interrupt received; finishing current Stage2 product chunk before stopping.\n";
                    stopAfterChunk = true;
                }
            }

            eng->sync();
            std::cout << "\n[PM1] Stage2 resume2reg chunk " << chunkNo << " done.\n";
            if (stopAfterChunk) {
                std::cout << "[PM1] Stopped after a clean chunk boundary. Re-run the same command to restart from the original S1 resume.\n";
                interrupted = false;
                delete eng;
                return 0;
            }
        }

        std::cout << "\n[PM1] Stage2 resume2reg product exponent done. Computing GCD of (X-1, M_p)...\n";
        mpz_class X = compute_X_with_dots(eng, (engine::Reg)RSTATE, Mp);
        X -= 1;
        if (X < 0) X += Mp;
        mpz_class g = gcd_with_dots(X, Mp);

        auto gcd_mpz = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
            mpz_class r;
            mpz_gcd(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
            return r;
        };

        mpz_class gNew = g;
        for (const std::string& fs : options.knownFactors) {
            if (gNew == 1) break;
            mpz_class f;
            try { f = mpz_class(fs); } catch (...) { continue; }
            if (f <= 1) continue;
            mpz_class d = gcd_mpz(gNew, f);
            while (d != 1) { gNew /= d; d = gcd_mpz(gNew, f); }
        }

        const bool found = (gNew != 1 && gNew != Mp);
        const double elapsed = duration<double>(high_resolution_clock::now() - texp0).count();
        std::cout << "\nElapsed time (stage 2 resume2reg) = " << std::fixed << std::setprecision(2) << elapsed << " s.\n";

        std::string filename = "stage2_resume2reg_result_B1_" + std::to_string(B1u) +
                               "_from_" + std::to_string(stage2Low) +
                               "_to_" + std::to_string(B2u) +
                               "_p_" + std::to_string(options.exponent) + ".txt";
        if (found) {
            std::string f = gNew.get_str(10);
            writeStageResult(filename, "B1=" + std::to_string(B1u) + " B2Start=" + std::to_string(stage2Low) + " B2=" + std::to_string(B2u) + " factor=" + f);
            std::cout << "\n>>>  Factor P-1 (stage 2 resume2reg product exponent) found : " << f << "\n";
            options.knownFactors.push_back(f);
        } else {
            writeStageResult(filename, "No factor P-1 stage2 resume2reg in range (" + std::to_string(stage2Low) + "," + std::to_string(B2u) + "] from B1=" + std::to_string(B1u));
            std::cout << "\nNo factor P-1 (stage 2 resume2reg product exponent) in range ("
                      << stage2Low << ", " << B2u << "] from resume B1=" << B1u << "\n";
        }

        std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
        std::cout << "Manual submission JSON:\n" << json << "\n";
        io::WorktodoManager wm(options);
        wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage2_resume2reg", json);
        wm.appendToResultsTxt(json);

        delete eng;
        return found ? 0 : 1;
    }

    if (options.pm1_ultralowmem) {
        // Ultra-lowmem GPU path v21: one-register product-exponent Stage 2.
        // Compute directly:
        //     3^( E(B1) * 2*p * Q ) mod M_p,
        // where Q = product of primes in (B1, B2]. Since Stage 1 H is
        // 3^(E(B1)*2*p), this equals H^Q and is a valid P-1 Stage-2 extension.
        // It uses the Marin fast3 path only: square_mul(R,3) when a bit is set.
        static constexpr size_t RSTATE = 0;

        std::cout << "[PM1] Ultra-low-memory Stage 2: GPU product-exponent with 1 register "
                  << "(no BSGS baby-table, no CPU powm, no multicarte, no H multiplicand register).\n";
        std::cout << "[PM1] Method: compute 3^(E(B1)*2*p*prod_primes(B1,B2]) directly using fast3.\n";

        std::cout << "[PM1] Building Stage 2 product exponent E2 = E(B1)*2*p*Q..." << std::flush;
        mpz_class E2 = buildE(B1u);
        E2 *= mpz_class(2) * mpz_from_u64(options.exponent);
        for (uint64_t q : primes) mpz_mul_u64(E2, q);
        const mp_bitcnt_t bits = mpz_sizeinbase(E2.get_mpz_t(), 2);
        std::cout << " done. bits=" << bits << ", primes=" << primes.size() << "\n";

        engine* eng = nullptr;
        try {
            std::cout << "[PM1] Allocating Stage 2 GPU engine with 1 register..." << std::flush;
            eng = engine::create_gpu(pexp, 1, (size_t)options.device_id, verbose);
            std::cout << " done.\n";
        } catch (const std::exception& ex) {
            std::cout << " failed.\n";
            std::cerr << "[PM1] Ultra-low-memory one-register GPU Stage 2 allocation failed: "
                      << ex.what() << "\n";
            return -2;
        }

        eng->set((engine::Reg)RSTATE, 1u);
        auto texp0 = high_resolution_clock::now();
        auto last = high_resolution_clock::now();
        for (mp_bitcnt_t i = bits; i > 0; --i) {
            const int b = mpz_tstbit(E2.get_mpz_t(), i - 1) ? 1 : 0;
            if (b) eng->square_mul((engine::Reg)RSTATE, 3u);
            else   eng->square_mul((engine::Reg)RSTATE);

            auto now = high_resolution_clock::now();
            if (duration_cast<seconds>(now - last).count() >= 3) {
                const mp_bitcnt_t doneBits = bits - i + 1;
                const double pct = 100.0 * double(doneBits) / double(bits);
                const double elapsed = duration<double>(now - texp0).count();
                const double ips = elapsed > 0.0 ? double(doneBits) / elapsed : 0.0;
                const double eta = ips > 0.0 ? double(i - 1) / ips : 0.0;
                std::cout << "Progress: " << std::fixed << std::setprecision(2) << pct
                          << "% | Stage2 bits: " << doneBits << "/" << bits
                          << " | Elapsed: " << elapsed << "s | IPS: " << ips
                          << " | ETA: " << eta << "s\r" << std::flush;
                last = now;
            }
            if (interrupted) {
                std::cout << "\nInterrupted by user during Stage 2 ultralowmem one-register exponentiation.\n";
                delete eng;
                interrupted = false;
                return 0;
            }
        }
        std::cout << "\n[PM1] Stage 2 one-register exponentiation done. Computing GCD of (X-1, M_p)...\n";

        mpz_class X = compute_X_with_dots(eng, (engine::Reg)RSTATE, Mp);
        // P-1 test condition: if q-1 divides the exponent for a factor q of M_p,
        // then X = 3^E2 == 1 (mod q).  The factor is therefore in gcd(X-1, M_p),
        // not in gcd(X, M_p).  v21/v22 accidentally tested gcd(X, M_p).
        X -= 1;
        if (X < 0) X += Mp;
        mpz_class g = gcd_with_dots(X, Mp);
        const bool found = (g != 1 && g != Mp);
        if (found) {
            std::string f = g.get_str(10);
            std::cout << "\n>>>  Factor P-1 (stage 2 ultralowmem GPU one-register product exponent) found : " << f << "\n";
            options.knownFactors.push_back(f);
            delete eng;
            return 0;
        }
        std::cout << "\nNo factor P-1 (stage 2 ultralowmem GPU one-register product exponent) until B2 = " << B2u << "\n";
        delete eng;
        return 1;
    }

    // 3-register streamed product path: H is restored from CPU data for each prime,
    // RACC accumulates Π(H^q - 1).  One final GCD only.  This avoids the BSGS baby table.
    static constexpr size_t RH = 0;
    static constexpr size_t RACC = 1;
    static constexpr size_t RQ = 2;
    engine* eng = engine::create_gpu(pexp, 3, (size_t)options.device_id, verbose);
    eng->set((engine::Reg)RACC, 1);

    for (size_t i = 0; i < primes.size(); ++i) {
        const uint64_t q = primes[i];
        eng->set_data((engine::Reg)RH, hData);
        eng->pow((engine::Reg)RQ, (engine::Reg)RH, q);
        eng->sub((engine::Reg)RQ, 1);
        eng->set_multiplicand((engine::Reg)RQ, (engine::Reg)RQ);
        eng->mul((engine::Reg)RACC, (engine::Reg)RQ);

        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
            double percent = 100.0 * double(i + 1) / double(primes.size());
            double elapsed = duration<double>(now - t0).count();
            double ips = elapsed > 0.0 ? double(i + 1) / elapsed : 0.0;
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << percent
                      << "% | prime: " << q << " | Iter: " << (i + 1)
                      << "/" << primes.size() << " | Elapsed: " << elapsed
                      << "s | IPS: " << ips << "\r" << std::flush;
            lastDisplay = now;
        }
        if (interrupted) {
            std::cout << "\nInterrupted by user during Stage 2 lowmem streamed product.\n";
            delete eng;
            interrupted = false;
            return 0;
        }
    }
    std::cout << "\n";

    mpz_class X = compute_X_with_dots(eng, (engine::Reg)RACC, Mp);
    mpz_class g = gcd_with_dots(X, Mp);
    bool found = (g != 1 && g != Mp);
    if (found) {
        char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
        std::cout << "\n>>>  Factor P-1 (stage 2 lowmem streamed product) found : " << fstr << "\n";
        options.knownFactors.push_back(std::string(fstr));
        std::free(fstr);
    } else {
        std::cout << "\nNo factor P-1 (stage 2 lowmem streamed product) until B2 = " << B2u << "\n";
    }
    delete eng;
    return found ? 0 : 1;
}


int App::runPM1Stage2MarinVTrace() {
    using namespace std::chrono;

    if (guiServer_) guiServer_->setStatus("P-1 factoring stage 2 (V-trace BSGS)");

    const uint64_t B1u = options.B1, B2u = options.B2;
    mpz_class B1 = mpz_from_u64(B1u);
    mpz_class B2 = mpz_from_u64(B2u);

    if (B2 <= B1) {
        std::cerr << "Stage 2 V-trace error B2 < B1.\n";
        if (guiServer_) guiServer_->appendLog("Stage 2 V-trace error B2 < B1.");
        return -1;
    }

    uint64_t D = options.pm1_vtrace_D ? options.pm1_vtrace_D : 30030ULL;
    if (D < 4 || (D & 1ULL)) {
        std::cerr << "[PM1-VTRACE] D must be even and >= 4. Use e.g. -pm1-vtrace-d 210, 630 or 2310.\n";
        return -1;
    }

    const uint32_t pexp = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;
    const uint64_t SEG_SPAN = 100000000ULL;

    std::cout << "\nStart a P-1 factoring : Stage 2 V-trace Bounds: B1 = "
              << B1 << ", B2 = " << B2 << "\n";
    std::cout << "[PM1-VTRACE] Scalar trace path: V_n = H^n + H^(-n), D=" << D << "\n";
    std::cout << "[PM1-VTRACE] This scalar trace path is the default normal-memory Stage 2; use -pm1-vtrace-off for the previous classic Stage 2.\n";
    if (guiServer_) {
        std::ostringstream oss;
        oss << "Start P-1 Stage 2 V-trace: B1=" << B1 << " B2=" << B2 << " D=" << D;
        guiServer_->appendLog(oss.str());
    }

    // v65 safety contract:
    //   - the default path stays the v61 stable path: primorial-aware D=30030
    //     selection plus negative-baby/add term construction.
    //   - product-tree accumulation remains opt-in only via
    //     -pm1-vtrace-product-tree, so the normal stable regression level cannot
    //     be lost by default.  The product-tree path is also restricted to the
    //     dense-prime-map case for exact bucket construction.
    //   - auto-D is now memory-aware.  For larger exponents D=30030 can require
    //     a single register buffer above the OpenCL max-allocation limit even
    //     though total VRAM looks almost sufficient.  The default auto-D filters
    //     candidates using CL_DEVICE_GLOBAL_MEM_SIZE and CL_DEVICE_MAX_MEM_ALLOC_SIZE
    //     before creating the engine; manual -pm1-vtrace-d still means force it.
    const bool useNegBabyAdd = !options.pm1_vtrace_negadd_off;
    const bool requestedProductTree = options.pm1_vtrace_product_tree;
    bool useProductTree = false;

    struct VTraceDeviceMemInfo {
        cl_ulong global = 0;
        cl_ulong maxAlloc = 0;
        std::string name;
        std::string vendor;
        bool ok = false;
    };

    auto query_vtrace_device_mem = [](size_t wantedDevice)->VTraceDeviceMemInfo {
        VTraceDeviceMemInfo out;
        cl_uint numPlatforms = 0;
        cl_platform_id platforms[64];
        if (clGetPlatformIDs(64, platforms, &numPlatforms) != CL_SUCCESS) return out;

        auto scan = [&](cl_device_type dtype, bool& anyFound)->bool {
            size_t idx = 0;
            for (cl_uint pi = 0; pi < numPlatforms; ++pi) {
                cl_uint numDevices = 0;
                cl_device_id devices[64];
                cl_int r = clGetDeviceIDs(platforms[pi], dtype, 64, devices, &numDevices);
                if (r != CL_SUCCESS) continue;
                anyFound = true;
                for (cl_uint di = 0; di < numDevices; ++di, ++idx) {
                    if (idx != wantedDevice) continue;
                    char dname[1024] = {0};
                    char dvendor[1024] = {0};
                    (void)clGetDeviceInfo(devices[di], CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
                    (void)clGetDeviceInfo(devices[di], CL_DEVICE_VENDOR, sizeof(dvendor), dvendor, nullptr);
                    cl_ulong global = 0, maxAlloc = 0;
                    if (clGetDeviceInfo(devices[di], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global), &global, nullptr) != CL_SUCCESS) return false;
                    if (clGetDeviceInfo(devices[di], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc), &maxAlloc, nullptr) != CL_SUCCESS) maxAlloc = 0;
                    out.global = global;
                    out.maxAlloc = maxAlloc;
                    out.name = dname;
                    out.vendor = dvendor;
                    out.ok = true;
                    return true;
                }
            }
            return false;
        };

        bool anyGpu = false;
        if (scan(CL_DEVICE_TYPE_GPU, anyGpu)) return out;
        if (!anyGpu) { bool anyAll = false; (void)scan(CL_DEVICE_TYPE_ALL, anyAll); }
        return out;
    };

    const VTraceDeviceMemInfo vtraceMem = query_vtrace_device_mem((size_t)options.device_id);
    const size_t vtraceTransformN = ibdwt::transform_size(pexp);
    auto gib_vtrace = [](long double b)->long double { return b / 1073741824.0L; };
    auto vtrace_memory_plan = [&](size_t regs)->std::pair<long double,long double> {
        const long double n = (long double)vtraceTransformN;
        const long double regBytes = (long double)regs * n * (long double)sizeof(uint64_t);
        const long double totalBytes = regBytes
                                   + (n / 4.0L) * (long double)sizeof(uint64_t)
                                   + 3.0L * n * (long double)sizeof(uint64_t)
                                   + 2.0L * n * (long double)sizeof(uint64_t)
                                   + n * (long double)sizeof(uint8_t);
        return {regBytes, totalBytes};
    };

    auto vtrace_aux_bytes = [&]()->long double {
        const long double n = (long double)vtraceTransformN;
        return (n / 4.0L) * (long double)sizeof(uint64_t)
             + 3.0L * n * (long double)sizeof(uint64_t)
             + 2.0L * n * (long double)sizeof(uint64_t)
             + n * (long double)sizeof(uint8_t);
    };

    auto vtrace_segmented_reg_bytes = [&](size_t regs)->long double {
        if (!vtraceMem.ok || vtraceMem.maxAlloc == 0 || vtraceTransformN == 0) {
            return vtrace_memory_plan(regs).first;
        }
        long double segFrac = 0.94L;
        if (const char* envFrac = std::getenv("PRMERS_MARIN_SEGMENTED_MAXALLOC_FRAC")) {
            const long double f = std::strtold(envFrac, nullptr);
            if (f > 0.10L && f < 0.985L) segFrac = f;
        }
        size_t scratchRegs = 3;
        if (const char* envScratch = std::getenv("PRMERS_MARIN_SEGMENTED_SCRATCH_REGS")) {
            const unsigned long long v = std::strtoull(envScratch, nullptr, 10);
            if (v != 0) scratchRegs = (size_t)v;
        }
        const long double perReg = (long double)vtraceTransformN * (long double)sizeof(uint64_t);
        const size_t slots = (size_t)((long double)vtraceMem.maxAlloc * segFrac / perReg);
        if (slots <= scratchRegs + 1) return std::numeric_limits<long double>::infinity();
        const size_t usable = slots - scratchRegs;
        long double bytes = 0.0L;
        for (size_t first = 0; first < regs; first += usable) {
            const size_t logicalHere = std::min(usable, regs - first);
            bytes += (long double)(logicalHere + scratchRegs) * perReg;
        }
        return bytes;
    };
    const bool allowTightVTraceMem = (std::getenv("PRMERS_PM1_VTRACE_ALLOW_TIGHT_MEM") != nullptr);
    // v65: keep much more headroom below CL_DEVICE_MAX_MEM_ALLOC_SIZE.
    // On NVIDIA OpenCL (e.g. RTX 3080/T4), allocating a register slab close to
    // the reported max single allocation can fail later and leave the command
    // queue invalid.  The V-trace slab is one large buffer, so use a conservative
    // default cap and allow explicit opt-in via PRMERS_PM1_VTRACE_ALLOW_TIGHT_MEM.
    const long double maxAllocFrac = allowTightVTraceMem ? 0.985L : 0.680L;
    const long double globalFrac   = allowTightVTraceMem ? 0.940L : 0.700L;
    auto vtrace_mem_fits = [&](size_t regs)->bool {
        if (!vtraceMem.ok) return true;
        const auto [regBytes, totalBytes] = vtrace_memory_plan(regs);
        const bool segmentedDisabled = (std::getenv("PRMERS_MARIN_SEGMENTED_DISABLE") != nullptr);

        // v84: multi-cl_mem segmented regspace removes the max-single-allocation
        // limit, but it does not create more VRAM.  Score/guards must therefore
        // check the total bytes of all GPU segments plus auxiliary buffers and the
        // small companion engine used by the V-trace setup.  v83 only checked that
        // each segment was below maxAlloc, so auto-D could pick e.g. D=2310,
        // active=120 on RTX 3080: each cl_mem was legal, but the total exceeded
        // physical VRAM and later failed in OpenCL.
        if (vtraceMem.maxAlloc != 0 && regBytes > (long double)vtraceMem.maxAlloc * maxAllocFrac) {
            if (segmentedDisabled) return false;
            long double segGlobalFrac = 0.98L;
            if (const char* envFrac = std::getenv("PRMERS_MARIN_SEGMENTED_GLOBAL_FRAC")) {
                const long double f = std::strtold(envFrac, nullptr);
                if (f > 0.50L && f < 1.20L) segGlobalFrac = f;
            }
            const long double segmentedRegBytes = vtrace_segmented_reg_bytes(regs);
            const long double auxBytes = vtrace_aux_bytes();
            // V-trace constructs V1/VD using a temporary 11-register engine while
            // the main stage-2 engine is still alive.  Account for that resident
            // allocation too so the auto-plan does not overcommit RTX-class cards.
            const long double companion11 = vtrace_memory_plan(11).second;
            const long double segmentedTotal = segmentedRegBytes + auxBytes + companion11;
            if (!std::isfinite((double)segmentedTotal)) return false;
            if (vtraceMem.global != 0 && segmentedTotal > (long double)vtraceMem.global * segGlobalFrac) return false;
            return true;
        }

        if (vtraceMem.global != 0 && totalBytes > (long double)vtraceMem.global * globalFrac) return false;
        return true;
    };

    auto vtrace_flat_safe = [&](size_t regs)->bool {
        if (!vtraceMem.ok) return true;
        const auto [regBytes, totalBytes] = vtrace_memory_plan(regs);
        if (vtraceMem.maxAlloc != 0 && regBytes > (long double)vtraceMem.maxAlloc * maxAllocFrac) return false;
        if (vtraceMem.global != 0 && totalBytes > (long double)vtraceMem.global * globalFrac) return false;
        return true;
    };

    auto vtrace_paged_needed = [&](size_t regs)->bool {
        if (!vtraceMem.ok || vtraceMem.maxAlloc == 0) return false;
        const auto [regBytes, totalBytes] = vtrace_memory_plan(regs);
        (void)totalBytes;
        return regBytes > (long double)vtraceMem.maxAlloc * maxAllocFrac;
    };

    auto vtrace_tight_flat = [&](size_t regs)->bool {
        if (!vtraceMem.ok || vtraceMem.maxAlloc == 0) return false;
        const auto [regBytes, totalBytes] = vtrace_memory_plan(regs);
        (void)totalBytes;
        return regBytes > (long double)vtraceMem.maxAlloc * 0.62L;
    };

    auto gcd_u64 = [](uint64_t a, uint64_t b)->uint64_t{
        while (b) { uint64_t t = a % b; a = b; b = t; }
        return a;
    };

    const uint64_t root = isqrt_u64(B2u);
    const std::vector<uint32_t> basePrimes = sieve_base_primes((uint32_t)root);

    auto is_prime_trial = [&](uint64_t n)->bool{
        if (n < 2) return false;
        for (uint32_t p32 : basePrimes) {
            const uint64_t p = (uint64_t)p32;
            if (p * p > n) break;
            if (n % p == 0) return n == p;
        }
        return true;
    };

    mpz_class p0;
    mpz_nextprime(p0.get_mpz_t(), B1.get_mpz_t());
    if (p0 > B2) {
        std::cout << "\nNo factor P-1 (stage 2 V-trace) until B2 = " << B2 << " (no primes in range)\n";
        if (guiServer_) guiServer_->appendLog("No factor P-1 Stage 2 V-trace (no primes in range).");
        return 1;
    }
    const uint64_t p0u = mpz_get_u64(p0.get_mpz_t());

    // v53: for small/medium B2 ranges, build one dense CPU prime map once.
    // This removes many trial divisions in the duplicate-pair test and also lets
    // auto-D evaluate candidate D values cheaply.  It is CPU memory only and does
    // not alter the classic Stage 2 path.
    static constexpr uint64_t VTRACE_DENSE_PRIME_MAX = 100000000ULL;
    std::vector<uint64_t> denseStage2Primes;
    std::vector<uint8_t> densePrimeMark;
    if (B2u <= VTRACE_DENSE_PRIME_MAX) {
        denseStage2Primes.clear();
        segmented_primes_odd(B1u + 1, B2u, basePrimes, denseStage2Primes);
        densePrimeMark.assign((size_t)B2u + 1, uint8_t(0));
        for (uint64_t q : denseStage2Primes) densePrimeMark[(size_t)q] = uint8_t(1);
        std::cout << "[PM1-VTRACE] Dense CPU prime map enabled: "
                  << denseStage2Primes.size() << " primes, "
                  << (densePrimeMark.size() / (1024.0 * 1024.0)) << " MiB.\n";
    }

    useProductTree = requestedProductTree && !denseStage2Primes.empty();
    if (requestedProductTree && !useProductTree) {
        std::cout << "[PM1-VTRACE-PRODUCT-TREE] requested, but dense prime map is unavailable for this B2; "
                  << "falling back to the stable linear V-trace accumulator.\n";
    } else if (useProductTree) {
        uint32_t w = options.pm1_vtrace_product_tree_width;
        if (w < 2u) w = 2u;
        if (w > 64u) w = 64u;
        std::cout << "[PM1-VTRACE-PRODUCT-TREE] v63 experimental bucket product-tree enabled "
                  << "(opt-in, width=" << w << "). Default runs remain on the stable v61 path.\n";
    }

    auto is_prime_fast = [&](uint64_t n)->bool{
        if (!densePrimeMark.empty() && n < densePrimeMark.size()) return densePrimeMark[(size_t)n] != 0;
        return is_prime_trial(n);
    };

    auto baby_count_for_D = [&](uint64_t d)->size_t{
        size_t c = 0;
        for (uint64_t j = 1; j <= d / 2; j += 2) if (gcd_u64(j, d) == 1) ++c;
        return c;
    };

    auto trace_stats_for_D = [&](uint64_t d, uint64_t& terms, uint64_t& skips, uint64_t& maxk)->void{
        terms = 0; skips = 0; maxk = 0;
        const std::vector<uint64_t>* plist = denseStage2Primes.empty() ? nullptr : &denseStage2Primes;
        if (!plist) return;
        for (uint64_t q : *plist) {
            uint64_t k = q / d;
            uint64_t rem = q - k * d;
            uint64_t j = rem;
            if (rem > d / 2) { ++k; j = d - rem; }
            const uint64_t qminus = k * d - j;
            const uint64_t qplus  = k * d + j;
            bool process = true;
            if (j != 0 && q == qplus && qminus > B1u && is_prime_fast(qminus)) process = false;
            if (process) { ++terms; if (k > maxk) maxk = k; }
            else ++skips;
        }
    };

    const bool defaultAutoD = (options.pm1_vtrace_D == 0);
    const bool autoDEnabled = options.pm1_vtrace_auto_d || defaultAutoD;
    // v79: auto-D may select an execution plan, not only a D.  In particular,
    // when auto-batch is allowed, a large D is only accepted with
    // baby batching if the cost model predicts that it beats the full-slab or
    // paged-regspace plan.  These are applied after baby residues are built.
    bool autoDSelectedBabyBatching = false;
    bool autoDSelectedFullPaged = false;
    size_t autoDSelectedActiveBabyCount = 0;
    if (autoDEnabled) {
        const uint64_t maxRegs = options.pm1_vtrace_max_regs ? options.pm1_vtrace_max_regs :
                                 ((options.pm1_vtrace_auto_d_aggressive || options.pm1_vtrace_deep_d_auto) ? 8192ULL : 4096ULL);
        static constexpr uint64_t BASE_REGS_VTRACE_AUTO = 14ULL;
        static constexpr uint64_t VTRACE_PRIMORIAL_DEFAULT_D = 30030ULL;
        const std::vector<uint64_t> candidates = {
            // v64: ultra-small memory-safe fallbacks for 9-digit exponents on
            // 10--16 GiB GPUs.  At p~205M, even D=90 can be too close to the OpenCL
            // single-buffer allocation limit on NVIDIA OpenCL once driver
            // overhead is accounted for, so auto-D must have candidates below
            // 90 instead of crashing at D=30030.
            30, 42, 60, 70, 84, 90, 110, 120, 126, 140, 150, 154, 168, 180,
            // conservative/highly-composite small-D fallback
            210, 420, 630, 840, 1050, 1260, 1470, 1680, 1890, 2100,
            2310, 2730, 3570, 3990, 4620, 5460, 6930, 8190, 9240,
            11550, 13860, 18480, 23100,
            // v61 primorial-aware plateau scan around the measured sweet spot
            30030, 60060, 90090, 120120, 150150, 180180, 210210, 240240
        };
        auto primorial_tier = [](uint64_t cand)->int {
            if (cand == 30030ULL) return 0;
            if (cand == 60060ULL || cand == 90090ULL || cand == 120120ULL) return 1;
            if (cand == 150150ULL || cand == 180180ULL || cand == 210210ULL || cand == 240240ULL) return 2;
            if (cand == 4620ULL || cand == 13860ULL || cand == 23100ULL) return 3;
            return 4;
        };
        if (denseStage2Primes.empty()) {
            // v64: B2 can be too large for the dense CPU prime map, but memory
            // safety must still be enforced.  We cannot compute exact paired
            // term counts cheaply here, so choose the largest fitting D from
            // the candidate set under the register cap and OpenCL memory limits.
            // This preserves correctness and avoids the previous D=30030 ->
            // hundreds-of-GiB register slab failure at 9-digit exponents.
            if (vtraceMem.ok) {
                std::cout << "[PM1-VTRACE-MEM] auto-D memory guard: transform=" << vtraceTransformN
                          << " | device='" << vtraceMem.name << "'"
                          << " | global=" << std::fixed << std::setprecision(2) << gib_vtrace((long double)vtraceMem.global) << " GiB";
                if (vtraceMem.maxAlloc != 0) std::cout << " | max-alloc=" << gib_vtrace((long double)vtraceMem.maxAlloc) << " GiB";
                if (allowTightVTraceMem) std::cout << " | tight override enabled";
                std::cout << "\n";
            }

            uint64_t bestD = 0;
            size_t bestBabyCount = 0;
            size_t bestRegs = 0;
            size_t bestActiveBabies = 0;
            size_t bestBatches = 0;
            double bestBatchScore = std::numeric_limits<double>::infinity();
            uint64_t rejectedByMemory = 0;
            uint64_t rejectedByRegs = 0;
            uint64_t rejectedByBatches = 0;

            // v81: dense-prime-map-disabled path must score the plan that will
            // actually be executed.  v80 allowed vtrace_mem_fits(fullRegs) to mean
            // "full paged is ok", selected batches=1, then the executor later applied
            // an emergency flat-safe batch fallback (for example D=1260 -> active=6,
            // 24 passes).  That makes the auto-D score meaningless.  Here we compare
            // explicit plans: full-flat, full-paged, batch-flat, and batch+paged.
            bool autoBatchAllowed = options.pm1_vtrace_auto_batch;
            if (std::getenv("PRMERS_PM1_VTRACE_NO_AUTO_BATCH") != nullptr) autoBatchAllowed = false;
            const bool allowPagedBatch = autoBatchAllowed &&
                                         (std::getenv("PRMERS_PM1_VTRACE_NO_PAGED_BATCH") == nullptr);
            size_t maxAutoBatches = (size_t)std::max<uint64_t>(1, options.pm1_vtrace_max_batches);
            if (const char* envMaxB = std::getenv("PRMERS_PM1_VTRACE_MAX_BATCHES")) {
                const unsigned long long v = std::strtoull(envMaxB, nullptr, 10);
                if (v != 0) maxAutoBatches = (size_t)v;
            }

            auto max_active_babies_flat_safe = [&](size_t bc)->size_t{
                size_t best = 0;
                for (size_t active = 1; active <= bc; ++active) {
                    const size_t regs = (size_t)BASE_REGS_VTRACE_AUTO + active;
                    if (regs <= maxRegs && vtrace_flat_safe(regs)) best = active;
                }
                return best;
            };
            auto max_active_babies_paged_allowed = [&](size_t bc)->size_t{
                size_t best = 0;
                for (size_t active = 1; active <= bc; ++active) {
                    const size_t regs = (size_t)BASE_REGS_VTRACE_AUTO + active;
                    if (regs <= maxRegs && vtrace_mem_fits(regs)) best = active;
                }
                return best;
            };

            bool bestPlanUsesBatching = false;
            bool bestPlanUsesFullPaged = false;
            uint64_t consideredFullPaged = 0;
            uint64_t consideredBatchPlans = 0;
            uint64_t consideredBatchPaged = 0;

            auto consider_coarse_plan = [&](uint64_t cand, size_t bc, size_t active,
                                            size_t batches, size_t regs,
                                            bool usesBatching, bool usesPaged)->void {
                if (active == 0 || batches == 0 || regs > maxRegs) return;
                const double approxK = double(B2u > B1u ? (B2u - B1u) : B2u) / double(cand);
                const double passPenalty = 25000.0 * double(batches > 0 ? batches - 1 : 0);
                const double babyPenalty = 1200.0 * double(bc);
                const double tierPenalty = 50000.0 * double(primorial_tier(cand));
                // Full paged slabs with many logical babies are a last resort; they
                // can thrash the host-backed register layer.  Batch+paged is allowed
                // but still penalized because each active window may fault/map pages.
                const double fullPagedPenalty = (!usesBatching && usesPaged) ? (650000.0 + 2500.0 * double(bc)) : 0.0;
                const double batchPagedPenalty = (usesBatching && usesPaged) ? (55000.0 * double(batches) + 350.0 * double(active)) : 0.0;
                const double score = double(batches) * approxK + passPenalty + babyPenalty + tierPenalty
                                   + fullPagedPenalty + batchPagedPenalty;
                if (score < bestBatchScore) {
                    bestBatchScore = score;
                    bestD = cand;
                    bestBabyCount = bc;
                    bestRegs = regs;
                    bestActiveBabies = active;
                    bestBatches = batches;
                    bestPlanUsesBatching = usesBatching;
                    bestPlanUsesFullPaged = (!usesBatching && usesPaged);
                }
            };

            for (uint64_t cand : candidates) {
                if (cand < 4 || (cand & 1ULL) || cand > B2u) continue;
                const size_t bc = baby_count_for_D(cand);
                const size_t fullRegs = (size_t)BASE_REGS_VTRACE_AUTO + bc
                                      + (useProductTree ? std::min<size_t>(64, std::max<size_t>(2, options.pm1_vtrace_product_tree_width)) : 0);
                if (fullRegs > maxRegs && !autoBatchAllowed) { ++rejectedByRegs; continue; }

                const bool fullFlat = (fullRegs <= maxRegs && vtrace_flat_safe(fullRegs));
                const bool fullPaged = (fullRegs <= maxRegs && !fullFlat && vtrace_mem_fits(fullRegs));
                if (fullFlat) {
                    consider_coarse_plan(cand, bc, bc, 1, fullRegs, false, false);
                } else if (fullPaged) {
                    ++consideredFullPaged;
                    consider_coarse_plan(cand, bc, bc, 1, fullRegs, false, true);
                } else if (!autoBatchAllowed) {
                    ++rejectedByMemory;
                    continue;
                }

                if (autoBatchAllowed && bc > 1 && !useProductTree) {
                    std::vector<size_t> activeCandidates;
                    auto add_active_candidate = [&](size_t active) {
                        if (active == 0 || active >= bc) return;
                        if (std::find(activeCandidates.begin(), activeCandidates.end(), active) == activeCandidates.end())
                            activeCandidates.push_back(active);
                    };
                    const size_t activeFlat = max_active_babies_flat_safe(bc);
                    size_t activePagedMax = allowPagedBatch ? max_active_babies_paged_allowed(bc) : activeFlat;
                    // v82: keep batch+paged hot windows small enough to stay mostly resident
                    // in the physical scratch.  Paged regspace is a safety net, not a good
                    // inner-loop cache replacement.  If active babies exceed physical slots
                    // minus the V-trace base registers, the page layer evicts/reloads babies
                    // on almost every term and becomes very slow.
                    if (allowPagedBatch && vtraceMem.ok && vtraceMem.maxAlloc != 0) {
                        long double scratchFrac = 0.94L;
                        if (const char* envFrac = std::getenv("PRMERS_MARIN_PAGED_MAXALLOC_FRAC")) {
                            const long double f = std::strtold(envFrac, nullptr);
                            if (f > 0.10L && f < 0.985L) scratchFrac = f;
                        }
                        size_t hotMargin = 2;
                        if (const char* envMargin = std::getenv("PRMERS_PM1_VTRACE_PAGED_HOT_MARGIN")) {
                            const unsigned long long v = std::strtoull(envMargin, nullptr, 10);
                            if (v != 0) hotMargin = (size_t)v;
                        }
                        const size_t regBytesOne = (size_t)vtraceTransformN * sizeof(uint64_t);
                        const size_t physicalSlots = regBytesOne ? (size_t)((long double)vtraceMem.maxAlloc * scratchFrac / (long double)regBytesOne) : 0;
                        if (physicalSlots > (size_t)BASE_REGS_VTRACE_AUTO + hotMargin) {
                            const size_t hotActive = physicalSlots - (size_t)BASE_REGS_VTRACE_AUTO - hotMargin;
                            if (hotActive != 0 && activePagedMax > hotActive) activePagedMax = std::max(activeFlat, hotActive);
                        }
                    }
                    add_active_candidate(activeFlat);
                    for (size_t targetBatches = 2; targetBatches <= maxAutoBatches; ++targetBatches) {
                        add_active_candidate((bc + targetBatches - 1) / targetBatches);
                    }
                    if (activeFlat != 0) {
                        add_active_candidate(activeFlat * 2);
                        add_active_candidate(activeFlat * 3);
                        add_active_candidate(activeFlat * 4);
                        add_active_candidate(activeFlat * 6);
                        add_active_candidate(activeFlat * 8);
                    }
                    if (activePagedMax != 0) {
                        add_active_candidate(activePagedMax / 4);
                        add_active_candidate(activePagedMax / 2);
                        add_active_candidate(activePagedMax);
                    }

                    for (size_t active : activeCandidates) {
                        if (active == 0 || active >= bc) continue;
                        const size_t batches = (bc + active - 1) / active;
                        if (batches > maxAutoBatches) { ++rejectedByBatches; continue; }
                        const size_t regs = (size_t)BASE_REGS_VTRACE_AUTO + active;
                        if (regs > maxRegs || !vtrace_mem_fits(regs)) continue;
                        const bool batchFlat = vtrace_flat_safe(regs);
                        const bool batchPaged = !batchFlat;
                        if (batchPaged && !allowPagedBatch) continue;
                        ++consideredBatchPlans;
                        if (batchPaged) ++consideredBatchPaged;
                        consider_coarse_plan(cand, bc, active, batches, regs, true, batchPaged);
                    }
                }
            }

            if (rejectedByMemory != 0) {
                std::cout << "[PM1-VTRACE-MEM] rejected " << rejectedByMemory
                          << " auto-D candidate(s) that were too tight for this transform/device. "
                          << "Set PRMERS_PM1_VTRACE_ALLOW_TIGHT_MEM=1 to benchmark them anyway.\n";
            }
            if (rejectedByRegs != 0) {
                std::cout << "[PM1-VTRACE] rejected " << rejectedByRegs
                          << " auto-D candidate(s) above max-regs=" << maxRegs << ".\n";
            }
            if (rejectedByBatches != 0) {
                std::cout << "[PM1-VTRACE-BATCH] rejected " << rejectedByBatches
                          << " auto-D candidate(s) requiring more than " << maxAutoBatches
                          << " baby-window pass(es). Use -pm1-vtrace-max-batches <N> to allow more.\n";
            }
            if (consideredFullPaged != 0 || consideredBatchPlans != 0) {
                std::cout << "[PM1-VTRACE-BATCH] dense-map coarse scorer considered " << consideredFullPaged
                          << " full-paged plan(s), " << consideredBatchPlans
                          << " baby-batch plan(s) including " << consideredBatchPaged
                          << " batch+paged plan(s).\n";
            }

            if (bestD == 0) {
                std::cerr << "[PM1-VTRACE-MEM] no V-trace D candidate fits this transform/device "
                          << "under max-regs=" << maxRegs << ". Falling back to classic Stage 2 is recommended: "
                          << "rerun with -pm1-vtrace-off, or set PRMERS_PM1_VTRACE_ALLOW_TIGHT_MEM=1 to force.\n";
                return -2;
            }

            if (bestD != D) {
                std::cout << "[PM1-VTRACE-BATCH] dense prime map disabled for B2=" << B2u
                          << "; integrated D+batch auto selected D=" << bestD
                          << " under max-regs=" << maxRegs
                          << " (baby=" << bestBabyCount
                          << ", active-baby=" << bestActiveBabies
                          << ", batches=" << bestBatches
                          << ", regs=" << bestRegs
                          << ", plan=" << (bestPlanUsesBatching ? (vtrace_flat_safe((size_t)BASE_REGS_VTRACE_AUTO + bestActiveBabies) ? "batch-flat" : "batch+paged") : (bestPlanUsesFullPaged ? "full-paged" : "full-flat"))
                          << ", approx max-k=" << (B2u / bestD)
                          << ", batch-score=" << bestBatchScore << ").\n";
                D = bestD;
            } else {
                std::cout << "[PM1-VTRACE-BATCH] dense prime map disabled for B2=" << B2u
                          << "; integrated D+batch auto kept D=" << D
                          << " under max-regs=" << maxRegs
                          << " (baby=" << bestBabyCount
                          << ", active-baby=" << bestActiveBabies
                          << ", batches=" << bestBatches
                          << ", regs=" << bestRegs
                          << ", plan=" << (bestPlanUsesBatching ? (vtrace_flat_safe((size_t)BASE_REGS_VTRACE_AUTO + bestActiveBabies) ? "batch-flat" : "batch+paged") : (bestPlanUsesFullPaged ? "full-paged" : "full-flat"))
                          << ", batch-score=" << bestBatchScore << ").\n";
            }
            if (bestPlanUsesBatching && bestActiveBabies != 0 && bestActiveBabies < bestBabyCount) {
                autoDSelectedBabyBatching = true;
                autoDSelectedActiveBabyCount = bestActiveBabies;
            } else if (bestPlanUsesFullPaged) {
                autoDSelectedFullPaged = true;
            }
        } else {
            if (vtraceMem.ok) {
                std::cout << "[PM1-VTRACE-MEM] auto-D memory guard: transform=" << vtraceTransformN
                          << " | device='" << vtraceMem.name << "'"
                          << " | global=" << std::fixed << std::setprecision(2) << gib_vtrace((long double)vtraceMem.global) << " GiB";
                if (vtraceMem.maxAlloc != 0) std::cout << " | max-alloc=" << gib_vtrace((long double)vtraceMem.maxAlloc) << " GiB";
                if (allowTightVTraceMem) std::cout << " | tight override enabled";
                std::cout << "\n";
            }

            uint64_t bestD = D, bestTerms = 0, bestSkips = 0, bestMaxK = 0;
            size_t bestBabyCount = 0, bestRegs = 0;
            size_t bestActiveBabies = 0, bestBatches = 1;
            bool bestUsesBatching = false;
            double bestScore = std::numeric_limits<double>::infinity();
            uint64_t rejectedByMemory = 0, rejectedByBatches = 0;
            uint64_t consideredPaged = 0, consideredTightFlat = 0, consideredBatch = 0;

            const bool autoBatchAllowed = options.pm1_vtrace_auto_batch &&
                                          (std::getenv("PRMERS_PM1_VTRACE_NO_AUTO_BATCH") == nullptr) &&
                                          !useProductTree;
            size_t maxAutoBatches = (size_t)std::max<uint64_t>(1, options.pm1_vtrace_max_batches);
            if (const char* envMaxB = std::getenv("PRMERS_PM1_VTRACE_MAX_BATCHES")) {
                const unsigned long long v = std::strtoull(envMaxB, nullptr, 10);
                if (v != 0) maxAutoBatches = (size_t)v;
            }
            // v80: batching and paging are complementary, not mutually exclusive.
            // A baby window is allowed to be either flat-safe or paged. The scorer
            // penalizes paged windows, but it may still choose them when they reduce
            // the number of rescans enough to beat a tiny flat window or a full-slab
            // paged plan.
            const bool allowPagedBatch = autoBatchAllowed &&
                                         (std::getenv("PRMERS_PM1_VTRACE_NO_PAGED_BATCH") == nullptr);

            long double pagedHostLimitMult = 2.0L;
            if (const char* envMult = std::getenv("PRMERS_MARIN_PAGED_HOST_LIMIT_GLOBAL_MULT")) {
                char* endp = nullptr;
                const long double v = std::strtold(envMult, &endp);
                if (endp != envMult && v > 0.0L) pagedHostLimitMult = v;
            }

            auto max_active_babies_flat_safe = [&](size_t bc)->size_t{
                size_t best = 0;
                for (size_t active = 1; active <= bc; ++active) {
                    const size_t regs = (size_t)BASE_REGS_VTRACE_AUTO + active;
                    if (regs <= maxRegs && vtrace_flat_safe(regs)) best = active;
                }
                return best;
            };

            auto max_active_babies_paged_allowed = [&](size_t bc)->size_t{
                size_t best = 0;
                for (size_t active = 1; active <= bc; ++active) {
                    const size_t regs = (size_t)BASE_REGS_VTRACE_AUTO + active;
                    if (regs <= maxRegs && vtrace_mem_fits(regs)) best = active;
                }
                return best;
            };

            auto consider_plan = [&](uint64_t cand, uint64_t terms, uint64_t skips, uint64_t maxk,
                                     size_t bc, size_t regs, bool usesBatching, size_t activeBabies,
                                     size_t batches, bool pagedNeeded, bool tightFlat) {
                // Runtime is not purely proportional to terms.  Small-D runs pay more giant
                // recurrence (large max-k); baby-batched runs rescan the prime table and
                // restart giant recurrence once per baby window; paged full-slab runs can
                // thrash the host-backed register cache.  This score is deliberately
                // conservative: it allows paged/batched candidates, but only lets them win
                // when they have a large theoretical advantage.
                const double extraPasses = double(batches > 0 ? batches - 1 : 0);
                const double primeScanPenalty = usesBatching ? 0.18 * double(denseStage2Primes.size()) * extraPasses : 0.0;
                const double giantPenalty = 0.70 * double(maxk) * double(usesBatching ? batches : 1);
                const double babyPenalty = (usesBatching ? 90.0 : 55.0) * double(bc);
                const double regPenalty = 4.0 * double(regs);
                const double passPenalty = usesBatching ? 22000.0 * extraPasses : 0.0;
                const double distancePenalty = 0.0015 * double(cand > VTRACE_PRIMORIAL_DEFAULT_D
                                                             ? cand - VTRACE_PRIMORIAL_DEFAULT_D
                                                             : VTRACE_PRIMORIAL_DEFAULT_D - cand);
                const double tierPenalty = 15.0 * double(primorial_tier(cand));
                const double tightFlatPenalty = tightFlat ? 900.0 : 0.0;
                // Full paged plans with hundreds of logical babies can thrash badly.
                // Paged baby-window plans are still penalized, but much less: the
                // active working set is bounded by activeBabies instead of the whole
                // baby table, which is exactly the batch+pagination design.
                const double pagedPenalty = pagedNeeded
                    ? (usesBatching
                        ? (0.28 * double(terms) * double(batches) + 1800.0 * double(batches) + 35.0 * double(activeBabies))
                        : (double(terms) * 1.35 + 9000.0 + 260.0 * double(bc)))
                    : 0.0;
                const double score = double(terms) + primeScanPenalty + giantPenalty + babyPenalty
                                   + regPenalty + passPenalty + distancePenalty + tierPenalty
                                   + tightFlatPenalty + pagedPenalty;
                if (score < bestScore) {
                    bestScore = score;
                    bestD = cand;
                    bestTerms = terms;
                    bestSkips = skips;
                    bestMaxK = maxk;
                    bestBabyCount = bc;
                    bestRegs = regs;
                    bestActiveBabies = activeBabies;
                    bestBatches = batches;
                    bestUsesBatching = usesBatching;
                }
            };

            // v80 paged-aware + smart-batch selection.
            // Full paged plans are considered, but heavily penalized because accessing
            // hundreds of logical baby registers through a small physical scratch can
            // thrash.  When auto-batch is enabled, we consider both
            // flat-safe baby windows and paged baby windows; batching is selected only
            // if its score wins. A forced -pm1-vtrace-baby-batch still overrides this later.
            for (uint64_t cand : candidates) {
                if (cand < 4 || (cand & 1ULL) || cand > B2u) continue;
                const size_t bc = baby_count_for_D(cand);
                const size_t fullRegs = (size_t)BASE_REGS_VTRACE_AUTO + bc
                                      + (useProductTree ? std::min<size_t>(64, std::max<size_t>(2, options.pm1_vtrace_product_tree_width)) : 0);
                if (fullRegs > maxRegs) continue;

                uint64_t terms = 0, skips = 0, maxk = 0;
                trace_stats_for_D(cand, terms, skips, maxk);
                if (!terms) continue;

                const bool flatSafe = vtrace_flat_safe(fullRegs);
                const bool pagedNeeded = (!flatSafe && vtrace_paged_needed(fullRegs));
                const bool tightFlat = (flatSafe && vtrace_tight_flat(fullRegs));
                if (pagedNeeded) ++consideredPaged;
                if (tightFlat) ++consideredTightFlat;

                bool fullAllowed = vtrace_mem_fits(fullRegs);
                if (pagedNeeded && vtraceMem.ok && vtraceMem.global != 0) {
                    const auto [regBytes, totalBytes] = vtrace_memory_plan(fullRegs);
                    (void)totalBytes;
                    if (!allowTightVTraceMem && regBytes > (long double)vtraceMem.global * pagedHostLimitMult) {
                        fullAllowed = false;
                    }
                }
                if (fullAllowed) {
                    consider_plan(cand, terms, skips, maxk, bc, fullRegs, false, bc, 1, pagedNeeded, tightFlat);
                } else {
                    ++rejectedByMemory;
                }

                if (autoBatchAllowed && bc > 1) {
                    std::vector<size_t> activeCandidates;
                    auto add_active_candidate = [&](size_t active) {
                        if (active == 0 || active >= bc) return;
                        if (std::find(activeCandidates.begin(), activeCandidates.end(), active) == activeCandidates.end())
                            activeCandidates.push_back(active);
                    };

                    const size_t activeFlat = max_active_babies_flat_safe(bc);
                    add_active_candidate(activeFlat);

                    if (allowPagedBatch) {
                        size_t activePagedMax = max_active_babies_paged_allowed(bc);
                        // v82: cap paged batch windows to the physical scratch hot set.
                        // Otherwise the engine pages during the inner loop instead of
                        // merely spilling cold logical registers between windows.
                        if (vtraceMem.ok && vtraceMem.maxAlloc != 0) {
                            long double scratchFrac = 0.94L;
                            if (const char* envFrac = std::getenv("PRMERS_MARIN_PAGED_MAXALLOC_FRAC")) {
                                const long double f = std::strtold(envFrac, nullptr);
                                if (f > 0.10L && f < 0.985L) scratchFrac = f;
                            }
                            size_t hotMargin = 2;
                            if (const char* envMargin = std::getenv("PRMERS_PM1_VTRACE_PAGED_HOT_MARGIN")) {
                                const unsigned long long v = std::strtoull(envMargin, nullptr, 10);
                                if (v != 0) hotMargin = (size_t)v;
                            }
                            const size_t regBytesOne = (size_t)vtraceTransformN * sizeof(uint64_t);
                            const size_t physicalSlots = regBytesOne ? (size_t)((long double)vtraceMem.maxAlloc * scratchFrac / (long double)regBytesOne) : 0;
                            if (physicalSlots > (size_t)BASE_REGS_VTRACE_AUTO + hotMargin) {
                                const size_t hotActive = physicalSlots - (size_t)BASE_REGS_VTRACE_AUTO - hotMargin;
                                if (hotActive != 0 && activePagedMax > hotActive) activePagedMax = std::max(activeFlat, hotActive);
                            }
                        }
                        // Try the smallest windows that satisfy each pass-count budget,
                        // and a few multiples of the flat-safe window. This lets the
                        // model choose batch+pagination instead of being limited to
                        // tiny flat-only batches.
                        for (size_t targetBatches = 2; targetBatches <= maxAutoBatches; ++targetBatches) {
                            add_active_candidate((bc + targetBatches - 1) / targetBatches);
                        }
                        if (activeFlat != 0) {
                            add_active_candidate(activeFlat * 2);
                            add_active_candidate(activeFlat * 3);
                            add_active_candidate(activeFlat * 4);
                            add_active_candidate(activeFlat * 6);
                            add_active_candidate(activeFlat * 8);
                        }
                        add_active_candidate(activePagedMax / 4);
                        add_active_candidate(activePagedMax / 2);
                        add_active_candidate(activePagedMax);
                    }

                    for (size_t active : activeCandidates) {
                        if (active == 0 || active >= bc) continue;
                        const size_t batches = (bc + active - 1) / active;
                        if (batches > maxAutoBatches) { ++rejectedByBatches; continue; }
                        const size_t batchRegs = (size_t)BASE_REGS_VTRACE_AUTO + active;
                        if (batchRegs > maxRegs || !vtrace_mem_fits(batchRegs)) continue;
                        const bool batchFlatSafe = vtrace_flat_safe(batchRegs);
                        const bool batchPagedNeeded = (!batchFlatSafe && vtrace_paged_needed(batchRegs));
                        const bool batchTightFlat = (batchFlatSafe && vtrace_tight_flat(batchRegs));
                        ++consideredBatch;
                        consider_plan(cand, terms, skips, maxk, bc, batchRegs, true, active, batches, batchPagedNeeded, batchTightFlat);
                    }
                }
            }
            if (rejectedByMemory != 0) {
                std::cout << "[PM1-VTRACE-MEM] rejected " << rejectedByMemory
                          << " full-slab auto-D candidate(s) too large for the paged policy; "
                          << "set PRMERS_PM1_VTRACE_ALLOW_TIGHT_MEM=1 or increase "
                          << "PRMERS_MARIN_PAGED_HOST_LIMIT_GLOBAL_MULT to benchmark them.\n";
            }
            if (rejectedByBatches != 0 && autoBatchAllowed) {
                std::cout << "[PM1-VTRACE-BATCH] rejected " << rejectedByBatches
                          << " batch candidate(s) requiring more than " << maxAutoBatches
                          << " baby-window pass(es); use -pm1-vtrace-max-batches <N> to allow more.\n";
            }
            if (consideredPaged != 0 || consideredTightFlat != 0 || consideredBatch != 0) {
                std::cout << "[PM1-VTRACE-MEM] paged-aware auto-D considered " << consideredPaged
                          << " paged full-slab candidate(s), " << consideredTightFlat
                          << " near-maxAlloc flat candidate(s), and " << consideredBatch
                          << " baby-batch candidate(s); all are cost-penalized, not blindly selected.\n";
            }
            if (bestD != D) {
                std::cout << "[PM1-VTRACE] primorial-aware auto-D selected D=" << bestD
                          << (defaultAutoD ? " (default)" : (options.pm1_vtrace_deep_d_auto ? " (deep-auto)" : (options.pm1_vtrace_auto_d_aggressive ? " (aggressive)" : "")))
                          << " under max-regs=" << maxRegs
                          << " (terms=" << bestTerms
                          << ", paired-skip=" << bestSkips
                          << ", max-k=" << bestMaxK
                          << ", baby=" << bestBabyCount
                          << ", regs=" << bestRegs
                          << ", plan=" << (bestUsesBatching ? "baby-batch" : "full-slab")
                          << ", active-baby=" << (bestUsesBatching ? bestActiveBabies : bestBabyCount)
                          << ", batches=" << (bestUsesBatching ? bestBatches : 1)
                          << ", smart-score=" << bestScore << ").\n";
                D = bestD;
            } else {
                std::cout << "[PM1-VTRACE] primorial-aware auto-D kept D=" << D
                          << (defaultAutoD ? " (default)" : (options.pm1_vtrace_deep_d_auto ? " (deep-auto)" : ""))
                          << " under max-regs=" << maxRegs
                          << " (plan=" << (bestUsesBatching ? "baby-batch" : "full-slab")
                          << ", smart-score=" << bestScore << ").\n";
            }
            if (bestUsesBatching) {
                autoDSelectedBabyBatching = true;
                autoDSelectedActiveBabyCount = bestActiveBabies;
            } else if (bestRegs != 0 && !vtrace_flat_safe(bestRegs) && vtrace_mem_fits(bestRegs)) {
                autoDSelectedFullPaged = true;
            }
        }
    }

    struct TracePair { uint64_t k; uint64_t j; uint64_t qminus; uint64_t qplus; };
    auto make_pair_for_prime = [&](uint64_t q)->TracePair{
        uint64_t k = q / D;
        uint64_t rem = q - k * D;
        uint64_t j = rem;
        if (rem > D / 2) {
            ++k;
            j = D - rem;
        }
        return TracePair{k, j, k * D - j, k * D + j};
    };

    // v97: Pair95 is now the default Stage-2 V-trace planner when the dense
    // prime map is available.  The default can be disabled with
    // -pm1-vtrace-pair95-off or PRMERS_PM1_VTRACE_PAIRING95_DISABLE=1.  If the
    // user does not force D, run a Pair95-aware selector that scores D, L and
    // the baby-window size together.  This avoids the v95/v96 failure mode where
    // the old classic auto-D picked D=42 and the Pair95 planner was bolted on
    // afterwards.
    auto env_truthy = [](const char* name)->bool{
        const char* v = std::getenv(name);
        if (!v || !*v) return false;
        if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
            std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 ||
            std::strcmp(v, "OFF") == 0) return false;
        return true;
    };
    const bool pair95EnvOn  = env_truthy("PRMERS_PM1_VTRACE_PAIRING95") ||
                              env_truthy("PRMERS_PM1_VTRACE_PAIR95");
    const bool pair95EnvOff = env_truthy("PRMERS_PM1_VTRACE_PAIRING95_DISABLE") ||
                              env_truthy("PRMERS_PM1_VTRACE_PAIR95_DISABLE") ||
                              env_truthy("PRMERS_PM1_VTRACE_PAIRING95_OFF") ||
                              env_truthy("PRMERS_PM1_VTRACE_PAIR95_OFF");
    bool usePair95 = !denseStage2Primes.empty() && !pair95EnvOff && !options.pm1_vtrace_pair95_off;
    if (options.pm1_vtrace_pair95 || pair95EnvOn) usePair95 = !denseStage2Primes.empty();
    if ((options.pm1_vtrace_pair95 || pair95EnvOn) && denseStage2Primes.empty()) {
        std::cout << "[PM1-VTRACE-PAIR95] requested but disabled: dense prime map is required "
                  << "for the safe/default implementation.\n";
    }
    if (!usePair95 && !denseStage2Primes.empty()) {
        std::cout << "[PM1-VTRACE-PAIR95] disabled; using classic V-trace pairing.\n";
    }

    size_t pair95L = (size_t)options.pm1_vtrace_pair95_L;
    if (const char* envL = std::getenv("PRMERS_PM1_VTRACE_PAIRING95_L")) {
        const unsigned long long v = std::strtoull(envL, nullptr, 10);
        if (v != 0) pair95L = (size_t)v;
    }
    if (const char* envL = std::getenv("PRMERS_PM1_VTRACE_PAIR95_L")) {
        const unsigned long long v = std::strtoull(envL, nullptr, 10);
        if (v != 0) pair95L = (size_t)v;
    }
    const bool pair95LForced = (pair95L != 0);

    std::unordered_map<uint64_t,size_t> pair95PrimeIndex;
    if (usePair95) {
        pair95PrimeIndex.reserve(denseStage2Primes.size() * 2 + 16);
        for (size_t i = 0; i < denseStage2Primes.size(); ++i) pair95PrimeIndex[denseStage2Primes[i]] = i;
    }

    auto pair95_make_offsets = [&](uint64_t d, size_t L)->std::vector<uint64_t>{
        std::vector<uint64_t> unit;
        unit.reserve((size_t)std::max<uint64_t>(1, d / 4));
        for (uint64_t j = 1; j <= d / 2; j += 2) {
            if (gcd_u64(j, d) == 1) unit.push_back(j);
        }
        std::vector<uint64_t> out;
        out.reserve(unit.size() * std::max<size_t>(1, L));
        for (size_t level = 0; level < std::max<size_t>(1, L); ++level) {
            const uint64_t distMul = (level >= 63) ? std::numeric_limits<uint64_t>::max()
                                                   : ((uint64_t(1) << level) - 1u);
            if (distMul > 0 && d > std::numeric_limits<uint64_t>::max() / distMul) break;
            const uint64_t baseAdd = distMul * d;
            for (uint64_t u : unit) {
                if (u <= std::numeric_limits<uint64_t>::max() - baseAdd) out.push_back(u + baseAdd);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };

    struct Pair95AutoPlan {
        uint64_t D = 0;
        size_t L = 0;
        size_t babyCount = 0;
        size_t activeBaby = 0;
        size_t passes = 0;
        uint64_t terms = 0;
        uint64_t pairs = 0;
        uint64_t singles = 0;
        uint64_t maxK = 0;
        double score = std::numeric_limits<double>::infinity();
        bool ok = false;
    };

    auto pair95_estimate_plan = [&](uint64_t candD, size_t candL, size_t activeBaby,
                                    size_t maxPasses, bool enforceMaxPasses)->Pair95AutoPlan{
        Pair95AutoPlan plan;
        plan.D = candD;
        plan.L = candL;
        const std::vector<uint64_t> offsets = pair95_make_offsets(candD, candL);
        plan.babyCount = offsets.size();
        if (plan.babyCount == 0 || activeBaby == 0) return plan;
        if (activeBaby > plan.babyCount) activeBaby = plan.babyCount;
        plan.activeBaby = activeBaby;
        plan.passes = (plan.babyCount + activeBaby - 1) / activeBaby;
        if (plan.passes == 0) plan.passes = 1;
        if (enforceMaxPasses && maxPasses != 0 && plan.passes > maxPasses) return plan;

        std::vector<std::vector<uint64_t>> residueToOffset((size_t)candD);
        for (uint64_t r : offsets) {
            const uint64_t rr = r % candD;
            const uint64_t residueLower = (rr == 0) ? 0 : (candD - rr);
            if (residueLower < candD) residueToOffset[(size_t)residueLower].push_back(r);
        }
        for (auto& v : residueToOffset) std::sort(v.begin(), v.end());

        std::vector<uint8_t> covered(denseStage2Primes.size(), uint8_t(0));
        for (size_t pi = 0; pi < denseStage2Primes.size(); ++pi) {
            if (covered[pi]) continue;
            const uint64_t q = denseStage2Primes[pi];
            bool paired = false;
            const auto& candidates = residueToOffset[(size_t)(q % candD)];
            for (uint64_t r : candidates) {
                if (r > (std::numeric_limits<uint64_t>::max() - q) / 2u) continue;
                const uint64_t mate = q + 2u * r;
                if (mate > B2u) continue;
                auto itMate = pair95PrimeIndex.find(mate);
                if (itMate == pair95PrimeIndex.end()) continue;
                if (covered[itMate->second]) continue;
                const uint64_t base = q + r;
                if ((base % candD) != 0) continue;
                const uint64_t k = base / candD;
                if (k > plan.maxK) plan.maxK = k;
                covered[itMate->second] = uint8_t(1);
                covered[pi] = uint8_t(1);
                ++plan.pairs;
                ++plan.terms;
                paired = true;
                break;
            }
            if (!paired) {
                uint64_t k = q / candD;
                uint64_t rem = q - k * candD;
                if (rem > candD / 2) ++k;
                if (k > plan.maxK) plan.maxK = k;
                covered[pi] = uint8_t(1);
                ++plan.singles;
                ++plan.terms;
            }
        }

        // Cost model: terms dominate, but repeated baby windows also replay giant
        // recurrences and reduce locality.  Penalize many passes strongly enough
        // that L=3/D=1260/batch=72 does not beat a 3--4 pass plan by term count
        // alone on RTX3080-class GPUs.
        const double passPenalty = 0.20 * double(plan.passes > 0 ? plan.passes - 1 : 0);
        const double giantPenalty = 2.0 * double(plan.maxK) * double(plan.passes);
        const double segmentedPenalty = vtrace_paged_needed(14 + activeBaby) ? 0.06 * double(plan.terms) : 0.0;
        const double babyPenalty = 250.0 * double(plan.babyCount);
        plan.score = double(plan.terms) * (1.0 + passPenalty) + giantPenalty + segmentedPenalty + babyPenalty;
        plan.ok = true;
        return plan;
    };

    if (usePair95 && defaultAutoD && !denseStage2Primes.empty()) {
        const uint64_t maxRegs = options.pm1_vtrace_max_regs ? options.pm1_vtrace_max_regs :
                                 ((options.pm1_vtrace_auto_d_aggressive || options.pm1_vtrace_deep_d_auto) ? 8192ULL : 4096ULL);
        size_t maxAutoBatches = (size_t)std::max<uint64_t>(1, options.pm1_vtrace_max_batches);
        if (const char* envMaxB = std::getenv("PRMERS_PM1_VTRACE_MAX_BATCHES")) {
            const unsigned long long v = std::strtoull(envMaxB, nullptr, 10);
            if (v != 0) maxAutoBatches = (size_t)v;
        }
        const bool allowMorePasses = env_truthy("PRMERS_PM1_VTRACE_PAIR95_ALLOW_MORE_PASSES");
        const std::vector<uint64_t> candidates = {
            210, 420, 630, 840, 1050, 1260, 1470, 1680, 1890, 2100,
            2310, 2520, 2730, 3150, 3570, 3990, 4200, 4620, 5040, 5460,
            6300, 6930, 7560, 8190, 9240, 10080, 11550, 12600, 13860,
            18480, 23100, 30030, 60060
        };
        std::vector<size_t> lCandidates;
        if (pair95LForced) lCandidates.push_back(std::max<size_t>(1, pair95L));
        else { lCandidates.push_back(2); lCandidates.push_back(3); }

        Pair95AutoPlan best;
        Pair95AutoPlan bestRelaxed;
        for (uint64_t candD : candidates) {
            if (candD < 4 || (candD & 1ULL)) continue;
            if (B2u / candD == 0) continue;
            for (size_t candL : lCandidates) {
                if (candL < 1) candL = 1;
                if (candL > 8) candL = 8;
                const std::vector<uint64_t> offsets = pair95_make_offsets(candD, candL);
                if (offsets.empty()) continue;
                size_t maxActive = 0;
                for (size_t active = 1; active <= offsets.size(); ++active) {
                    const size_t regs = 14 + active;
                    if (regs <= maxRegs && vtrace_mem_fits(regs)) maxActive = active;
                }
                if (maxActive == 0) continue;
                Pair95AutoPlan cur = pair95_estimate_plan(candD, candL, maxActive, maxAutoBatches, true);
                if (cur.ok && cur.score < best.score) best = cur;
                Pair95AutoPlan relaxed = pair95_estimate_plan(candD, candL, maxActive, maxAutoBatches, false);
                if (relaxed.ok && relaxed.score < bestRelaxed.score) bestRelaxed = relaxed;
            }
        }
        if (!best.ok && allowMorePasses && bestRelaxed.ok) best = bestRelaxed;
        if (best.ok) {
            D = best.D;
            pair95L = best.L;
            autoDSelectedBabyBatching = best.activeBaby < best.babyCount;
            autoDSelectedActiveBabyCount = best.activeBaby;
            autoDSelectedFullPaged = !autoDSelectedBabyBatching;
            std::cout << "[PM1-VTRACE-PAIR95] v97 auto selected D=" << best.D
                      << ", L=" << best.L
                      << ", babies=" << best.babyCount
                      << ", active-babies=" << best.activeBaby
                      << ", passes=" << best.passes
                      << ", terms=" << best.terms
                      << ", greedy-pairs=" << best.pairs
                      << ", singletons=" << best.singles
                      << ", score=" << std::fixed << std::setprecision(2) << best.score
                      << ". Disable with -pm1-vtrace-pair95-off or PRMERS_PM1_VTRACE_PAIRING95_DISABLE=1.\n";
        } else {
            if (pair95L == 0) pair95L = 2;
            std::cout << "[PM1-VTRACE-PAIR95] v97 auto selector found no memory-safe Pair95 plan under max-batches="
                      << maxAutoBatches << "; keeping D=" << D << ", L=" << pair95L
                      << ". Set PRMERS_PM1_VTRACE_PAIR95_ALLOW_MORE_PASSES=1 to relax.\n";
        }
    }
    if (usePair95 && pair95L == 0) pair95L = 3;

    std::cout << "[PM1-VTRACE] Effective execution D=" << D << " after auto-D/paged-aware cost filtering.\n";

    // j in [1, D/2], odd, gcd(j,D)=1.  Prime q=kD±j with q not dividing D always lands here.
    // v97: default AtNashev/Woltman-inspired irregular prime pairing.  The classic
    // half-residue set is extended with baby offsets u + (2^i-1)D when Pair95 is
    // enabled.  A CPU greedy planner then covers the prime interval with fewer
    // trace terms.  Use -pm1-vtrace-pair95-off to get the old classic planner.
    std::vector<int32_t> j2i((size_t)(D / 2 + 1), -1);
    std::vector<uint64_t> babyOffset;
    babyOffset.reserve((size_t)std::max<uint64_t>(1, D / 4));
    for (uint64_t j = 1; j <= D / 2; j += 2) {
        if (gcd_u64(j, D) == 1) {
            j2i[(size_t)j] = (int32_t)babyOffset.size();
            babyOffset.push_back(j);
        }
    }

    if (pair95L < 1) pair95L = 1;
    if (pair95L > 8) {
        std::cout << "[PM1-VTRACE-PAIR95] L=" << pair95L
                  << " capped to 8 to avoid an excessive baby precompute window.\n";
        pair95L = 8;
    }

    if (usePair95 && pair95L > 1) {
        const size_t classicUnit = babyOffset.size();
        std::vector<uint64_t> extended;
        extended.reserve(classicUnit * pair95L);
        for (size_t level = 0; level < pair95L; ++level) {
            const uint64_t distMul = (level >= 63) ? std::numeric_limits<uint64_t>::max()
                                                   : ((uint64_t(1) << level) - 1u);
            if (distMul > 0 && D > std::numeric_limits<uint64_t>::max() / distMul) break;
            const uint64_t baseAdd = distMul * D;
            for (uint64_t u : babyOffset) {
                if (u <= std::numeric_limits<uint64_t>::max() - baseAdd) extended.push_back(u + baseAdd);
            }
        }
        std::sort(extended.begin(), extended.end());
        extended.erase(std::unique(extended.begin(), extended.end()), extended.end());
        babyOffset.swap(extended);
    }

    std::unordered_map<uint64_t,int32_t> babyOffsetToGlobal;
    babyOffsetToGlobal.reserve(babyOffset.size() * 2 + 16);
    uint64_t maxBabyN = 0;
    for (size_t i = 0; i < babyOffset.size(); ++i) {
        babyOffsetToGlobal[babyOffset[i]] = (int32_t)i;
        if (babyOffset[i] > maxBabyN) maxBabyN = babyOffset[i];
    }

    const size_t babyCount = babyOffset.size();
    if (!babyCount) {
        std::cerr << "[PM1-VTRACE] no usable baby residues for D=" << D << "\n";
        return -2;
    }

    struct Pair95Task {
        uint64_t k = 0;
        int32_t babyIndex = -1;
        uint64_t qlo = 0;
        uint64_t qhi = 0;
        bool paired = false;
    };
    std::vector<Pair95Task> pair95Tasks;
    uint64_t pair95Paired = 0;
    uint64_t pair95Singleton = 0;

    if (usePair95) {
        if (useProductTree) {
            std::cout << "[PM1-VTRACE-PAIR95] disabling product-tree for v97 irregular planner; "
                      << "pair95 already reduces term count and keeps execution linear/safe.\n";
            useProductTree = false;
        }

        std::vector<std::vector<int32_t>> residueToBaby((size_t)D);
        for (size_t bi = 0; bi < babyOffset.size(); ++bi) {
            const uint64_t r = babyOffset[bi];
            const uint64_t rr = r % D;
            const uint64_t residueLower = (rr == 0) ? 0 : (D - rr);
            if (residueLower < D) residueToBaby[(size_t)residueLower].push_back((int32_t)bi);
        }
        for (auto& v : residueToBaby) {
            std::sort(v.begin(), v.end(), [&](int32_t a, int32_t b){
                return babyOffset[(size_t)a] < babyOffset[(size_t)b];
            });
        }

        if (pair95PrimeIndex.empty()) {
            pair95PrimeIndex.reserve(denseStage2Primes.size() * 2 + 16);
            for (size_t i = 0; i < denseStage2Primes.size(); ++i) pair95PrimeIndex[denseStage2Primes[i]] = i;
        }
        const auto& primeIndex = pair95PrimeIndex;
        std::vector<uint8_t> covered(denseStage2Primes.size(), uint8_t(0));
        pair95Tasks.reserve(denseStage2Primes.size());

        for (size_t pi = 0; pi < denseStage2Primes.size(); ++pi) {
            if (covered[pi]) continue;
            const uint64_t q = denseStage2Primes[pi];
            int32_t chosenBi = -1;
            uint64_t chosenMate = 0;
            const auto& candidates = residueToBaby[(size_t)(q % D)];
            for (int32_t bi : candidates) {
                const uint64_t r = babyOffset[(size_t)bi];
                if (r > (std::numeric_limits<uint64_t>::max() - q) / 2u) continue;
                const uint64_t mate = q + 2u * r;
                if (mate > B2u) continue;
                auto itMate = primeIndex.find(mate);
                if (itMate == primeIndex.end()) continue;
                if (covered[itMate->second]) continue;
                chosenBi = bi;
                chosenMate = mate;
                covered[itMate->second] = uint8_t(1);
                break;
            }

            if (chosenBi >= 0) {
                const uint64_t r = babyOffset[(size_t)chosenBi];
                const uint64_t base = q + r;
                if ((base % D) != 0) {
                    std::cerr << "[PM1-VTRACE-PAIR95] internal residue error for q=" << q
                              << " r=" << r << " D=" << D << "\n";
                    return -4;
                }
                pair95Tasks.push_back(Pair95Task{base / D, chosenBi, q, chosenMate, true});
                covered[pi] = uint8_t(1);
                ++pair95Paired;
            } else {
                const TracePair tp = make_pair_for_prime(q);
                if (tp.j >= j2i.size() || j2i[(size_t)tp.j] < 0) {
                    std::cerr << "[PM1-VTRACE-PAIR95] internal fallback error: no classic baby for q="
                              << q << " j=" << tp.j << " D=" << D << "\n";
                    return -4;
                }
                pair95Tasks.push_back(Pair95Task{tp.k, j2i[(size_t)tp.j], q, q, false});
                covered[pi] = uint8_t(1);
                ++pair95Singleton;
            }
        }

        std::stable_sort(pair95Tasks.begin(), pair95Tasks.end(), [](const Pair95Task& a, const Pair95Task& b){
            if (a.k != b.k) return a.k < b.k;
            return a.babyIndex < b.babyIndex;
        });
        std::cout << "[PM1-VTRACE-PAIR95] enabled: irregular units L=" << pair95L
                  << " | baby-offsets=" << babyCount
                  << " | max-offset=" << maxBabyN
                  << " | primes=" << denseStage2Primes.size()
                  << " | terms=" << pair95Tasks.size()
                  << " | greedy-pairs=" << pair95Paired
                  << " | singletons=" << pair95Singleton
                  << " | term reduction=" << std::fixed << std::setprecision(2)
                  << (denseStage2Primes.empty() ? 0.0 : 100.0 * double(pair95Paired) / double(denseStage2Primes.size()))
                  << "%\n";
    }

    struct ProductTreeBucket {
        uint64_t k = 0;
        uint64_t primeCount = 0;
        uint64_t skippedUpper = 0;
        std::vector<int32_t> babyIndex; // -1 means V_0=2, otherwise index in baby table
    };
    std::vector<ProductTreeBucket> productTreeBuckets;
    if (useProductTree) {
        std::map<uint64_t, ProductTreeBucket> tmpBuckets;
        for (uint64_t q : denseStage2Primes) {
            const TracePair tp = make_pair_for_prime(q);
            auto& b = tmpBuckets[tp.k];
            b.k = tp.k;
            ++b.primeCount;
            bool processPair = true;
            if (tp.j != 0 && q == tp.qplus && tp.qminus > B1u && is_prime_fast(tp.qminus)) {
                processPair = false;
                ++b.skippedUpper;
            }
            if (!processPair) continue;
            int32_t bi = -1;
            if (tp.j != 0) {
                if (tp.j >= j2i.size() || j2i[(size_t)tp.j] < 0) {
                    std::cerr << "[PM1-VTRACE-PRODUCT-TREE] INTERNAL ERROR: no baby V_j for q="
                              << q << " j=" << tp.j << " D=" << D << "\n";
                    return -4;
                }
                bi = j2i[(size_t)tp.j];
            }
            b.babyIndex.push_back(bi);
        }
        productTreeBuckets.reserve(tmpBuckets.size());
        uint64_t ptTerms = 0, ptPrimes = 0, ptSkips = 0;
        size_t maxBucketTerms = 0;
        for (auto& kv : tmpBuckets) {
            ptTerms += (uint64_t)kv.second.babyIndex.size();
            ptPrimes += kv.second.primeCount;
            ptSkips += kv.second.skippedUpper;
            maxBucketTerms = std::max(maxBucketTerms, kv.second.babyIndex.size());
            productTreeBuckets.push_back(std::move(kv.second));
        }
        std::cout << "[PM1-VTRACE-PRODUCT-TREE] exact buckets=" << productTreeBuckets.size()
                  << " | terms=" << ptTerms
                  << " | paired upper skips=" << ptSkips
                  << " | max terms/bucket=" << maxBucketTerms
                  << " | avg terms/bucket="
                  << std::fixed << std::setprecision(2)
                  << (productTreeBuckets.empty() ? 0.0 : double(ptTerms) / double(productTreeBuckets.size()))
                  << "\n";
    }

    // Register layout.  Baby registers store scalar traces V_j, or -V_j in the
    // default v59 neg-baby/add mode.
    static constexpr size_t RSTATE    = 0;   // H loaded from stage-1 checkpoint, mostly for diagnostics
    static constexpr size_t RACC      = 1;   // accumulator Π(V_kD - V_j)
    static constexpr size_t RV1       = 2;   // V_1 = H + H^-1
    static constexpr size_t RMUL_V1   = 3;   // multiplicand(V_1)
    static constexpr size_t RVD       = 4;   // V_D
    static constexpr size_t RMUL_VD   = 5;   // multiplicand(V_D)
    static constexpr size_t RGPREV    = 6;   // V_(cur_k-1)D ; at k=0 this is V_-D = V_D
    static constexpr size_t RGCUR     = 7;   // V_cur_kD
    static constexpr size_t RGNEXT    = 8;   // scratch for next giant
    static constexpr size_t RBPREV    = 9;   // baby recurrence prev
    static constexpr size_t RBCUR     = 10;  // baby recurrence cur
    static constexpr size_t RBNEXT    = 11;  // baby recurrence next
    static constexpr size_t RTMP      = 12;  // scratch term / multiplication input
    static constexpr size_t RMUL_TMP  = 13;  // multiplicand(term)
    static constexpr size_t baseRegsVTrace = 14;
    static constexpr size_t baseRegsStage1 = 11;

    const size_t babyBase = baseRegsVTrace;

    // v71: V-trace baby batching.  Older builds placed all baby traces in one
    // contiguous OpenCL register slab.  On NVIDIA OpenCL this is limited by
    // CL_DEVICE_MAX_MEM_ALLOC_SIZE, not just total VRAM.  Instead of rejecting a
    // useful D (for example D=90 on RTX 3080), keep D and store only a window of
    // baby traces at once.  Stage 2 then scans the prime interval once per baby
    // window while keeping the same global accumulator.  This is slower than a
    // single slab when the slab fits, but it lets us benchmark/use larger D
    // values without a giant single allocation.
    const char* envBabyBatch = std::getenv("PRMERS_PM1_VTRACE_BABY_BATCH");
    size_t requestedBabyBatch = (size_t)options.pm1_vtrace_baby_batch;
    const bool cliOrEnvBabyBatchRequested = (requestedBabyBatch != 0) || (envBabyBatch && *envBabyBatch);
    if (envBabyBatch && *envBabyBatch) {
        const unsigned long long v = std::strtoull(envBabyBatch, nullptr, 10);
        requestedBabyBatch = (size_t)v;
    }

    size_t productTreeWidth = useProductTree ? std::min<size_t>(64, std::max<size_t>(2, options.pm1_vtrace_product_tree_width)) : 0;
    size_t activeBabyCount = babyCount;
    bool useBabyBatching = false;
    bool explicitFullBabySlab = false; // v86: user asked for batch>=babyCount or no-auto full slab; do not flat-fallback.
    const bool noAutoBatchRequested = (!options.pm1_vtrace_auto_batch) ||
                                      (std::getenv("PRMERS_PM1_VTRACE_NO_AUTO_BATCH") != nullptr);

    auto choose_max_baby_batch_that_fits = [&]()->size_t{
        size_t best = 0;
        for (size_t bc = 1; bc <= babyCount; ++bc) {
            const size_t regs = baseRegsVTrace + bc; // product-tree disabled while batching
            if (vtrace_flat_safe(regs)) best = bc;
        }
        return best;
    };

    // v96: pair95 creates virtual baby offsets (u, u+D, u+3D, ...).  On RTX-class
    // GPUs a flat single cl_mem slab is often much smaller than the segmented
    // regspace that can actually run.  The old emergency path therefore picked
    // active-babies=6 for p~205M even though 72 active babies fit with segmented
    // GPU-only storage.  For opt-in pair95, prefer the largest memory-safe
    // segmented window, not the tiny flat-safe window.
    auto choose_max_baby_batch_that_fits_segmented = [&]()->size_t{
        size_t best = 0;
        for (size_t bc = 1; bc <= babyCount; ++bc) {
            const size_t regs = baseRegsVTrace + bc; // product-tree disabled while batching
            if (vtrace_mem_fits(regs)) best = bc;
        }
        return best;
    };

    if (requestedBabyBatch != 0) {
        if (requestedBabyBatch < babyCount) {
            activeBabyCount = std::max<size_t>(1, requestedBabyBatch);
            useBabyBatching = true;
            std::cout << "[PM1-VTRACE-BATCH] forced baby batch size from CLI/env: " << activeBabyCount << "\n";
        } else {
            // v86: -pm1-vtrace-baby-batch equal/larger than the baby table means
            // "keep the whole baby table resident".  This is a valid benchmark for
            // the segmented GPU-only regspace (for example D=630 has exactly 72
            // babies).  Older builds fell through and the emergency flat-safe
            // fallback silently changed it to a tiny batch=6, so the requested
            // full-slab/segmented plan was never tested.
            activeBabyCount = babyCount;
            useBabyBatching = false;
            explicitFullBabySlab = true;
            std::cout << "[PM1-VTRACE-BATCH] CLI/env baby batch " << requestedBabyBatch
                      << " covers all " << babyCount
                      << " babies; using full baby slab (segmented if needed).\n";
        }
    }

    if (usePair95 && !useBabyBatching && !explicitFullBabySlab && !cliOrEnvBabyBatchRequested &&
        !noAutoBatchRequested && !vtrace_mem_fits(baseRegsVTrace + babyCount + productTreeWidth)) {
        const size_t bestSegmentedBatch = choose_max_baby_batch_that_fits_segmented();
        if (bestSegmentedBatch != 0 && bestSegmentedBatch < babyCount) {
            activeBabyCount = bestSegmentedBatch;
            useBabyBatching = true;
            std::cout << "[PM1-VTRACE-PAIR95] v96 memory-aware baby batching: "
                      << "active-babies-per-pass=" << activeBabyCount
                      << " using segmented/paged regspace instead of the tiny flat-safe fallback.\n";
        }
    }

    if (!useBabyBatching && !explicitFullBabySlab && autoDSelectedBabyBatching && autoDSelectedActiveBabyCount != 0 && autoDSelectedActiveBabyCount < babyCount) {
        activeBabyCount = std::max<size_t>(1, autoDSelectedActiveBabyCount);
        useBabyBatching = true;
        std::cout << "[PM1-VTRACE-BATCH] auto-D selected baby batching: active-babies-per-pass="
                  << activeBabyCount << "\n";
    }

    if (!useBabyBatching && !explicitFullBabySlab && !noAutoBatchRequested &&
        !autoDSelectedFullPaged && !vtrace_flat_safe(baseRegsVTrace + babyCount + productTreeWidth)) {
        // Emergency fallback only.  If the selected full plan is not flat-safe
        // and auto-D did not explicitly select batching, prefer a flat-safe baby
        // window instead of silently running a huge paged full-slab.  v86: do not
        // override explicit full-slab tests (-pm1-vtrace-no-auto-batch or
        // -pm1-vtrace-baby-batch >= babyCount); those are exactly how we validate
        // the segmented GPU-only regspace beyond CL_DEVICE_MAX_MEM_ALLOC_SIZE.
        const size_t bestBatch = choose_max_baby_batch_that_fits();
        if (bestBatch != 0 && bestBatch < babyCount) {
            activeBabyCount = bestBatch;
            useBabyBatching = true;
            std::cout << "[PM1-VTRACE-BATCH] emergency flat-safe batch fallback: active-babies-per-pass="
                      << activeBabyCount << "\n";
        }
    }

    if (useBabyBatching) {
        if (useProductTree) {
            std::cout << "[PM1-VTRACE-BATCH] product-tree disabled because baby batching is active.\n";
            useProductTree = false;
        }
        productTreeWidth = 0;
        if (activeBabyCount > babyCount) activeBabyCount = babyCount;
        if (activeBabyCount == 0) activeBabyCount = 1;
        std::cout << "[PM1-VTRACE-BATCH] enabled: D=" << D
                  << " total-babies=" << babyCount
                  << " active-babies-per-pass=" << activeBabyCount
                  << " passes=" << ((babyCount + activeBabyCount - 1) / activeBabyCount)
                  << ". Set PRMERS_PM1_VTRACE_BABY_BATCH=<n> to override.\n";
    }

    // v96: buckets built after activeBabyCount is final.  Each bucket contains only
    // the pair95 tasks whose baby trace is resident in that window.  This avoids
    // scanning the full task list once per baby window and gives a truthful ETA.
    std::vector<std::vector<size_t>> pair95TaskBuckets;
    if (usePair95) {
        const size_t pairPasses = useBabyBatching ? ((babyCount + activeBabyCount - 1) / activeBabyCount) : 1;
        pair95TaskBuckets.assign(std::max<size_t>(1, pairPasses), {});
        for (size_t ti = 0; ti < pair95Tasks.size(); ++ti) {
            const int32_t bi = pair95Tasks[ti].babyIndex;
            size_t bucket = 0;
            if (useBabyBatching && bi >= 0) bucket = std::min(pair95TaskBuckets.size() - 1, (size_t)bi / activeBabyCount);
            pair95TaskBuckets[bucket].push_back(ti);
        }
        if (useBabyBatching) {
            size_t nonEmpty = 0, maxBucket = 0;
            for (const auto& b : pair95TaskBuckets) { if (!b.empty()) ++nonEmpty; maxBucket = std::max(maxBucket, b.size()); }
            std::cout << "[PM1-VTRACE-PAIR95] v97 task buckets: passes=" << pair95TaskBuckets.size()
                      << ", non-empty=" << nonEmpty
                      << ", max-tasks/pass=" << maxBucket
                      << "; execution no longer scans all pair95 tasks in every baby pass.\n";
        }
    }

    const size_t productTreeScratchBase = babyBase + activeBabyCount;
    const size_t regCount = baseRegsVTrace + activeBabyCount + productTreeWidth;

    std::cout << "[PM1-VTRACE] Baby residues stored: " << babyCount
              << " (odd j<=D/2, gcd(j,D)=1), GPU registers=" << regCount << "\n";
    if (useNegBabyAdd) {
        std::cout << "[PM1-VTRACE-NEGADD] negative baby traces enabled: term = V_kD + (-V_j). "
                  << "Use -pm1-vtrace-negadd-off to benchmark the old copy+sub_reg path.\n";
    } else {
        std::cout << "[PM1-VTRACE-NEGADD] disabled: using old copy+sub_reg term builder.\n";
    }
    if (useProductTree) {
        std::cout << "[PM1-VTRACE-PRODUCT-TREE] scratch width=" << productTreeWidth
                  << " regs; accumulation is bucket-local tree chunks, then one ACC multiply per chunk.\n";
    }
    if (vtraceMem.ok) {
        const auto [regBytes, totalBytes] = vtrace_memory_plan(regCount);
        if (!vtrace_mem_fits(regCount)) {
            std::cerr << "[PM1-VTRACE-MEM] selected/manual D=" << D << " uses "
                      << std::fixed << std::setprecision(2) << gib_vtrace(regBytes)
                      << " GiB for the register slab and " << gib_vtrace(totalBytes)
                      << " GiB total before driver overhead on device "
                      << gib_vtrace((long double)vtraceMem.global) << " GiB";
            if (vtraceMem.maxAlloc != 0) std::cerr << ", max single allocation=" << gib_vtrace((long double)vtraceMem.maxAlloc) << " GiB";
            std::cerr << ".\n";
            if (options.pm1_vtrace_D != 0 && !allowTightVTraceMem) {
                std::cerr << "[PM1-VTRACE-MEM] explicit -pm1-vtrace-d exceeds the old flat-buffer safety model; "
                          << "v74 will let the Marin paged-regspace fallback decide instead of rejecting in the caller. "
                          << "Set PRMERS_MARIN_PAGED_DISABLE=1 to restore strict flat-only behavior.\n";
            }
        }
    }

    engine* eng = nullptr;
    try {
        eng = engine::create_gpu(pexp, regCount, (size_t)options.device_id, verbose);
    } catch (const std::exception& ex) {
        std::cerr << "[PM1-VTRACE] GPU engine allocation failed for D=" << D
                  << ", regs=" << regCount << ": " << ex.what() << "\n";
        if (options.pm1_vtrace_D == 0) {
            std::cerr << "[PM1-VTRACE] auto-D should have avoided this. Retry with -pm1-vtrace-d 42 or -pm1-vtrace-d 30; if intentional, set PRMERS_PM1_VTRACE_ALLOW_TIGHT_MEM=1.\n";
        }
        return -2;
    }
    if (!eng) {
        std::cerr << "[PM1-VTRACE] cannot allocate GPU engine.\n";
        return -2;
    }

    // v87: normal V-trace Stage-2 checkpoints are compact by default.
    // The old full-engine checkpoint copied every baby register to host RAM;
    // with segmented full slabs (e.g. D=630, 72 babies on RTX 3080) this can
    // mean a 6+ GiB host vector and may abort at the first periodic backup.
    // We only need the accumulator and current giant recurrence state; baby
    // traces are deterministic and are recomputed on resume.
    const int ckptVersionS2 = useProductTree ? 23 : 24;
    const bool compactVTraceCkpt = !useProductTree;
    std::ostringstream ck2;
    if (useProductTree) ck2 << "pm1_s2_vtrace_producttree_m_" << pexp << ".ckpt";
    else if (useNegBabyAdd) ck2 << "pm1_s2_vtrace_negadd_m_" << pexp << ".ckpt";
    else ck2 << "pm1_s2_vtrace_m_" << pexp << ".ckpt";
    const std::string ckpt_file_s2 = ck2.str();

    auto compact_vtrace_regs = [&](){
        std::vector<size_t> regs;
        // Core state sufficient for deterministic resume.  Baby registers and
        // multiplicand registers are recomputed on resume.
        regs.push_back(RACC);
        regs.push_back(RV1);
        // RVD and baby traces are recomputed from RV1 on compact resume.
        regs.push_back(RGPREV);
        regs.push_back(RGCUR);
        return regs;
    };

    auto read_ckpt_s2 = [&](engine* e, const std::string& file,
                            uint64_t& saved_p, uint64_t& saved_idx, uint64_t& saved_k,
                            double& et, uint64_t& sB1, uint64_t& sB2, uint64_t& sD,
                            bool& compactLoaded)->int{
        compactLoaded = false;
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != ckptVersionS2) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != pexp) return -2;
        if (!f.read(reinterpret_cast<char*>(&sB1), sizeof(sB1))) return -2;
        if (!f.read(reinterpret_cast<char*>(&sB2), sizeof(sB2))) return -2;
        if (!f.read(reinterpret_cast<char*>(&sD),  sizeof(sD)))  return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_p), sizeof(saved_p))) return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_idx), sizeof(saved_idx))) return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_k), sizeof(saved_k))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;

        if (version == 24) {
            uint32_t nregs = 0;
            if (!f.read(reinterpret_cast<char*>(&nregs), sizeof(nregs))) return -2;
            const size_t rsz = e->get_register_data_size();
            std::vector<char> one(rsz);
            for (uint32_t i = 0; i < nregs; ++i) {
                uint32_t reg = 0;
                if (!f.read(reinterpret_cast<char*>(&reg), sizeof(reg))) return -2;
                if (!f.read(one.data(), rsz)) return -2;
                if (!e->set_data((engine::Reg)reg, one)) return -2;
            }
            if (!f.check_crc32()) return -2;
            compactLoaded = true;
            return 0;
        }

        const size_t cksz = e->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!e->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };

    auto save_ckpt_s2 = [&](engine* e, uint64_t cur_p, uint64_t cur_idx, uint64_t cur_k, double et){
        const std::string oldf = ckpt_file_s2 + ".old", newf = ckpt_file_s2 + ".new";
        {
            File f(newf, "wb");
            int version = ckptVersionS2;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&pexp), sizeof(pexp))) return;
            if (!f.write(reinterpret_cast<const char*>(&B1u), sizeof(B1u))) return;
            if (!f.write(reinterpret_cast<const char*>(&B2u), sizeof(B2u))) return;
            if (!f.write(reinterpret_cast<const char*>(&D), sizeof(D))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_p), sizeof(cur_p))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_idx), sizeof(cur_idx))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_k), sizeof(cur_k))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;

            if (compactVTraceCkpt) {
                const std::vector<size_t> regs = compact_vtrace_regs();
                const uint32_t nregs = (uint32_t)regs.size();
                if (!f.write(reinterpret_cast<const char*>(&nregs), sizeof(nregs))) return;
                const size_t rsz = e->get_register_data_size();
                std::vector<char> one(rsz);
                for (size_t r : regs) {
                    const uint32_t reg = (uint32_t)r;
                    if (!f.write(reinterpret_cast<const char*>(&reg), sizeof(reg))) return;
                    if (!e->get_data(one, (engine::Reg)r)) return;
                    if (!f.write(one.data(), rsz)) return;
                }
                f.write_crc32();
            } else {
                const size_t cksz = e->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!e->get_checkpoint(data)) return;
                if (!f.write(data.data(), cksz)) return;
                f.write_crc32();
            }
        }
        std::remove(oldf.c_str());
        struct stat st;
        if ((stat(ckpt_file_s2.c_str(), &st) == 0) && (std::rename(ckpt_file_s2.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file_s2.c_str());
    };

    uint64_t resume_idx = 0, resume_p_u64 = 0, cur_k = 0, saved_k = 0;
    double restored_time = 0.0;
    uint64_t s2B1=0, s2B2=0, s2D=0;
    bool compactS2Loaded = false;
    int rs2 = read_ckpt_s2(eng, ckpt_file_s2, resume_p_u64, resume_idx, saved_k, restored_time, s2B1, s2B2, s2D, compactS2Loaded);
    bool resumed_s2 = (rs2 == 0) && (s2B1 == B1u) && (s2B2 == B2u) && (s2D == D);
    if (usePair95 && resumed_s2) {
        std::cout << "[PM1-VTRACE-PAIR95] existing Stage 2 checkpoint ignored; "
                  << "v95 irregular-pair checkpoints are not implemented yet.\n";
        resumed_s2 = false;
    }

    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;

    if (useBabyBatching && resumed_s2) {
        std::cout << "[PM1-VTRACE-BATCH] existing non-batched Stage 2 checkpoint ignored; "
                  << "baby-batched checkpoints are not implemented in v71.\n";
        resumed_s2 = false;
    }

    std::vector<int32_t> babyGlobalToLocal(babyCount, int32_t(-1));
    auto precompute_baby_window = [&](size_t batchStart, size_t batchEnd,
                                      std::vector<int32_t>& babyGlobalToLocal){
        if (batchEnd > babyCount) batchEnd = babyCount;
        std::fill(babyGlobalToLocal.begin(), babyGlobalToLocal.end(), int32_t(-1));
        for (size_t gi = batchStart; gi < batchEnd; ++gi) {
            babyGlobalToLocal[gi] = int32_t(gi - batchStart);
        }

        eng->set((engine::Reg)RBPREV, 2);          // V_0
        eng->copy((engine::Reg)RBCUR, (engine::Reg)RV1); // V_1

        size_t stored = 0;
        const size_t want = batchEnd - batchStart;
        int pct = -1;
        const uint64_t loopEnd = std::max<uint64_t>(D, maxBabyN);
        for (uint64_t n = 1; n <= loopEnd; ++n) {
            auto itBaby = babyOffsetToGlobal.find(n);
            if (itBaby != babyOffsetToGlobal.end()) {
                const int32_t gbi = itBaby->second;
                if (gbi >= 0 && (size_t)gbi >= batchStart && (size_t)gbi < batchEnd) {
                    const size_t local = (size_t)gbi - batchStart;
                    const size_t slot = babyBase + local;
                    if (useNegBabyAdd) {
                        eng->set((engine::Reg)slot, 0u);
                        eng->sub_reg((engine::Reg)slot, (engine::Reg)RBCUR);
                    } else {
                        eng->copy((engine::Reg)slot, (engine::Reg)RBCUR);
                    }
                    ++stored;
                    int newPct = int((stored * 100ull) / std::max<size_t>(1, want));
                    if (newPct > pct) { pct = newPct; std::cout << "\rV-trace baby window " << (batchStart + 1) << "-" << batchEnd << ": " << pct << "%" << std::flush; }
                }
            }
            if (n == D) {
                eng->copy((engine::Reg)RVD, (engine::Reg)RBCUR);
                if (n == loopEnd) break;
            }
            if (n == loopEnd) break;
            // V_{n+1} = V_1 * V_n - V_{n-1}
            eng->copy((engine::Reg)RBNEXT, (engine::Reg)RBCUR);
            eng->mul((engine::Reg)RBNEXT, (engine::Reg)RMUL_V1);
            eng->sub_reg((engine::Reg)RBNEXT, (engine::Reg)RBPREV);
            eng->copy((engine::Reg)RBPREV, (engine::Reg)RBCUR);
            eng->copy((engine::Reg)RBCUR, (engine::Reg)RBNEXT);
        }
        std::cout << "\rV-trace baby window " << (batchStart + 1) << "-" << batchEnd << ": 100%\n";
    };

    if (!resumed_s2) {
        // ---- load H from the Stage-1 checkpoint written by runPM1Marin() ----
        engine* eng_load = engine::create_gpu(pexp, baseRegsStage1, (size_t)options.device_id, verbose);
        if (!eng_load) { delete eng; return -2; }

        std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
        const std::string ckpt_file = ck.str();

        auto read_ckpt_stage1 = [&](engine* e, const std::string& file)->int{
            File f(file);
            if (!f.exists()) return -1;
            std::string backend_reason;
            if (!pm1_checkpoint_backend_matches(file, e, &backend_reason)) {
                std::cerr << "[PM1] Stage 1 checkpoint ignored: " << backend_reason << "\n";
                return -3;
            }
            int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 3) return -2;
            uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != pexp) return -2;
            uint32_t ri = 0; double et = 0.0;
            if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!e->set_checkpoint(data)) return -2;
            uint64_t tmp64;
            for (int i = 0; i < 4; ++i) if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
            uint8_t inlot=0; if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
            uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
            if (eacc_len) { std::string skip(eacc_len, '\0'); if (!f.read(skip.data(), eacc_len)) return -2; }
            uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
            if (wbits_len) { std::string skip(wbits_len, '\0'); if (!f.read(skip.data(), wbits_len)) return -2; }
            uint64_t chunkIdx=0, startP=0; uint8_t first=0; uint64_t processedBits=0, bitsInChunk=0;
            if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
            if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
            if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
            if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
            if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
            if (!f.check_crc32()) return -2;
            return 0;
        };

        int rr = read_ckpt_stage1(eng_load, ckpt_file);
        if (rr < 0) rr = read_ckpt_stage1(eng_load, ckpt_file + ".old");
        if (rr != 0) {
            delete eng_load; delete eng;
            std::cerr << "[PM1-VTRACE] cannot load pm1 stage1 checkpoint " << ckpt_file << "\n";
            if (guiServer_) guiServer_->appendLog("PM1-VTRACE: cannot load stage1 checkpoint.");
            return -2;
        }

        mpz_t H; mpz_init(H);
        eng_load->get_mpz(H, (engine::Reg)RSTATE);
        delete eng_load;

        eng->set_mpz((engine::Reg)RSTATE, H);

        mpz_class Hc(H);
        Hc %= Mp;
        mpz_class Hinvc;
        if (mpz_invert(Hinvc.get_mpz_t(), Hc.get_mpz_t(), Mp.get_mpz_t()) == 0) {
            mpz_class g;
            mpz_gcd(g.get_mpz_t(), Hc.get_mpz_t(), Mp.get_mpz_t());
            mpz_clear(H);
            delete eng;
            if (g != 1 && g != Mp) {
                std::string filename = "stage2_vtrace_result_B2_" + B2.get_str() + "_p_" + std::to_string(options.exponent) + ".txt";
                writeStageResult(filename, "inverse(H) failed because factor already divides H: factor=" + g.get_str());
                std::cout << "\n>>> P-1 factor found while building V_1: " << g.get_str() << "\n";
                return 0;
            }
            std::cerr << "[PM1-VTRACE] H is not invertible modulo M_p, but no proper factor was isolated.\n";
            return -2;
        }
        mpz_clear(H);

        mpz_class V1 = (Hc + Hinvc) % Mp;
        mpz_t V1t; mpz_init_set(V1t, V1.get_mpz_t());
        eng->set_mpz((engine::Reg)RV1, V1t);
        mpz_clear(V1t);
        eng->set_multiplicand((engine::Reg)RMUL_V1, (engine::Reg)RV1);

        std::cout << "[PM1-VTRACE] V1 = H + H^-1 built with one GMP inverse; precomputing V_j and V_D...\n";

        if (!useBabyBatching) {
            precompute_baby_window(0, babyCount, babyGlobalToLocal);
        } else {
            precompute_baby_window(0, std::min(activeBabyCount, babyCount), babyGlobalToLocal);
        }

        eng->set_multiplicand((engine::Reg)RMUL_VD, (engine::Reg)RVD);

        // Giant recurrence W_k = V_{kD}.  At k=0, W_0=2 and W_-1=V_D because V_-D=V_D.
        eng->copy((engine::Reg)RGPREV, (engine::Reg)RVD);
        eng->set((engine::Reg)RGCUR, 2);
        eng->set((engine::Reg)RACC, 1);
        cur_k = 0;
    } else {
        cur_k = saved_k;
        eng->set_multiplicand((engine::Reg)RMUL_V1, (engine::Reg)RV1);
        if (!useBabyBatching) {
            if (compactS2Loaded) {
                std::cout << "[PM1-VTRACE-CKPT] compact checkpoint loaded; recomputing deterministic baby table before resume...\n";
                precompute_baby_window(0, babyCount, babyGlobalToLocal);
            } else {
                for (size_t gi = 0; gi < babyCount; ++gi) babyGlobalToLocal[gi] = int32_t(gi);
            }
        }
        eng->set_multiplicand((engine::Reg)RMUL_VD, (engine::Reg)RVD);
        std::cout << "[PM1-VTRACE] Resuming Stage 2 V-trace from checkpoint at prime "
                  << resume_p_u64 << " (idx=" << resume_idx << ", k=" << cur_k << ") D=" << D << "\n";
    }

    uint64_t idx = 0;
    uint64_t p_ui = p0u;
    auto t0 = high_resolution_clock::now();
    if (resumed_s2) {
        idx = resume_idx;
        p_ui = resume_p_u64;
        t0 = high_resolution_clock::now() - duration_cast<high_resolution_clock::duration>(duration<double>(restored_time));
    }

    std::vector<uint64_t> primesRun;
    bool denseRun = !denseStage2Primes.empty();
    size_t densePosRun = 0;
    size_t posRun = 0;
    uint64_t segLowRun = 0;
    uint64_t segHighRun = 0;

    if (!useProductTree) {
        if (denseRun) {
            auto it = std::lower_bound(denseStage2Primes.begin(), denseStage2Primes.end(), p_ui);
            if (it == denseStage2Primes.end() || *it != p_ui) {
                delete eng;
                std::cerr << "[PM1-VTRACE] start prime not found in dense prime table.\n";
                return -3;
            }
            densePosRun = (size_t)std::distance(denseStage2Primes.begin(), it);
        } else {
            segLowRun = p_ui;
            segHighRun = std::min(B2u, segLowRun + SEG_SPAN - 1);
            segmented_primes_odd(segLowRun, segHighRun, basePrimes, primesRun);
            while (posRun < primesRun.size() && primesRun[posRun] < p_ui) ++posRun;
            if (posRun >= primesRun.size() || primesRun[posRun] != p_ui) {
                delete eng;
                std::cerr << "[PM1-VTRACE] start prime not found in segmented sieve.\n";
                return -3;
            }
        }
    }

    auto advancePrime = [&](uint64_t& out)->bool{
        if (denseRun) {
            if (densePosRun + 1 >= denseStage2Primes.size()) return false;
            out = denseStage2Primes[++densePosRun];
            return true;
        }
        for (;;) {
            if (posRun + 1 < primesRun.size()) {
                ++posRun;
                out = primesRun[posRun];
                return true;
            }
            if (segHighRun >= B2u) return false;
            segLowRun = segHighRun + 1;
            segHighRun = std::min(B2u, segLowRun + SEG_SPAN - 1);
            segmented_primes_odd(segLowRun, segHighRun, basePrimes, primesRun);
            posRun = 0;
            if (primesRun.empty()) continue;
            out = primesRun[0];
            return true;
        }
    };

    auto advance_giant_to = [&](uint64_t target_k){
        while (cur_k < target_k) {
            // next = V_D * cur - prev
            eng->copy((engine::Reg)RGNEXT, (engine::Reg)RGCUR);
            eng->mul((engine::Reg)RGNEXT, (engine::Reg)RMUL_VD);
            eng->sub_reg((engine::Reg)RGNEXT, (engine::Reg)RGPREV);
            eng->copy((engine::Reg)RGPREV, (engine::Reg)RGCUR);
            eng->copy((engine::Reg)RGCUR, (engine::Reg)RGNEXT);
            ++cur_k;
        }
    };

    uint64_t primesSeen = 0;
    uint64_t termsAccumulated = 0;
    uint64_t skippedPairedUpper = 0;

    auto start = t0;
    auto lastBackup  = high_resolution_clock::now();
    auto lastDisplay = high_resolution_clock::now();

    auto build_trace_term_into = [&](engine::Reg dst, int32_t bi){
        eng->copy(dst, (engine::Reg)RGCUR);
        if (bi < 0) {
            eng->sub(dst, 2);
        } else {
            const size_t babyReg = babyBase + (size_t)bi;
            if (useNegBabyAdd) eng->add(dst, (engine::Reg)babyReg);
            else eng->sub_reg(dst, (engine::Reg)babyReg);
        }
    };

    if (usePair95) {
        std::cout << "[PM1-VTRACE-PAIR95] executing greedy irregular pair plan"
                  << (useBabyBatching ? " with baby-window batching" : " with resident baby table")
                  << ".\n";
        const size_t numPasses = useBabyBatching ? ((babyCount + activeBabyCount - 1) / activeBabyCount) : 1;
        const size_t passStep = useBabyBatching ? activeBabyCount : babyCount;
        for (size_t batchStart = 0; batchStart < babyCount; batchStart += passStep) {
            const size_t passNo = batchStart / passStep;
            const size_t batchEnd = std::min(babyCount, batchStart + passStep);
            if (useBabyBatching || batchStart == 0) {
                if (numPasses > 1) {
                    const size_t bucketTerms = (passNo < pair95TaskBuckets.size()) ? pair95TaskBuckets[passNo].size() : 0;
                    std::cout << "[PM1-VTRACE-PAIR95] pass " << (passNo + 1)
                              << "/" << numPasses << " babies " << (batchStart + 1)
                              << ".." << batchEnd << " of " << babyCount
                              << " | queued-tasks=" << bucketTerms << "\n";
                }
                precompute_baby_window(batchStart, batchEnd, babyGlobalToLocal);
                eng->copy((engine::Reg)RGPREV, (engine::Reg)RVD);
                eng->set((engine::Reg)RGCUR, 2);
                cur_k = 0;
            }

            uint64_t passTerms = 0;
            const std::vector<size_t>* bucketPtr = nullptr;
            if (!pair95TaskBuckets.empty()) bucketPtr = &pair95TaskBuckets[std::min(passNo, pair95TaskBuckets.size() - 1)];
            std::vector<size_t> fallbackAll;
            if (!bucketPtr) {
                fallbackAll.reserve(pair95Tasks.size());
                for (size_t ti = 0; ti < pair95Tasks.size(); ++ti) fallbackAll.push_back(ti);
                bucketPtr = &fallbackAll;
            }
            const auto& bucket = *bucketPtr;
            for (size_t bucketPos = 0; bucketPos < bucket.size(); ++bucketPos) {
                const size_t ti = bucket[bucketPos];
                const Pair95Task& task = pair95Tasks[ti];
                if (task.babyIndex < 0) continue;
                int32_t localBaby = -1;
                if ((size_t)task.babyIndex < babyGlobalToLocal.size()) {
                    localBaby = babyGlobalToLocal[(size_t)task.babyIndex];
                }
                if (localBaby < 0) {
                    std::cerr << "\n[PM1-VTRACE-PAIR95] INTERNAL ERROR: bucketed task has non-resident baby index "
                              << task.babyIndex << " in pass " << (passNo + 1) << ".\n";
                    delete eng;
                    return -4;
                }

                if (interrupted) {
                    delete eng;
                    std::cout << "\nInterrupted by user during V-trace pair95 Stage 2. "
                              << "v97 does not checkpoint inside the irregular pair planner yet.\n";
                    interrupted = false;
                    return 0;
                }

                advance_giant_to(task.k);
                build_trace_term_into((engine::Reg)RTMP, localBaby);
                eng->set_multiplicand((engine::Reg)RMUL_TMP, (engine::Reg)RTMP);
                eng->mul((engine::Reg)RACC, (engine::Reg)RMUL_TMP);
                ++termsAccumulated;
                ++passTerms;
                primesSeen += task.paired ? 2u : 1u;

                auto now = high_resolution_clock::now();
                if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
                    const double percent = pair95Tasks.empty() ? 100.0 : 100.0 * double(termsAccumulated) / double(pair95Tasks.size());
                    const double elapsedSec = duration<double>(now - start).count();
                    const double ipsTerm  = (elapsedSec > 0.0) ? double(termsAccumulated) / elapsedSec : 0.0;
                    const double etaSec = (percent > 0.0) ? elapsedSec * (100.0 - percent) / percent : 0.0;
                    int days = int(etaSec) / 86400;
                    int hours = (int(etaSec) % 86400) / 3600;
                    int minutes = (int(etaSec) % 3600) / 60;
                    int seconds = int(etaSec) % 60;
                    std::cout << "Progress V-trace pair95: " << std::fixed << std::setprecision(2) << percent
                              << "% | pass=" << (passNo + 1) << "/" << numPasses
                              << " | bucket-task=" << (bucketPos + 1) << "/" << bucket.size()
                              << " | k=" << task.k
                              << " | terms=" << termsAccumulated << "/" << pair95Tasks.size()
                              << " | pair-skips=" << pair95Paired
                              << " | term/s=" << std::fixed << std::setprecision(2) << ipsTerm
                              << " | ETA=" << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                              << std::flush;
                    lastDisplay = now;
                }
            }
            if (numPasses > 1) {
                std::cout << "\n[PM1-VTRACE-PAIR95] pass " << (passNo + 1)
                          << " accumulated " << passTerms << " term(s).\n";
            }
        }
        skippedPairedUpper = pair95Paired;
        std::cout << "\n[PM1-VTRACE-PAIR95] processed " << pair95Tasks.size()
                  << " terms covering " << denseStage2Primes.size()
                  << " primes; greedy-pairs=" << pair95Paired
                  << ", singletons=" << pair95Singleton << ".\n";
    } else if (useBabyBatching) {
        std::cout << "[PM1-VTRACE-BATCH] scanning Stage 2 in "
                  << ((babyCount + activeBabyCount - 1) / activeBabyCount)
                  << " baby-window pass(es).  This avoids one huge OpenCL register slab.\n";

        const size_t numPasses = (babyCount + activeBabyCount - 1) / activeBabyCount;
        for (size_t batchStart = 0; batchStart < babyCount; batchStart += activeBabyCount) {
            const size_t batchEnd = std::min(babyCount, batchStart + activeBabyCount);
            if (interrupted) {
                delete eng;
                std::cout << "\nInterrupted by user during V-trace baby-batched Stage 2. "
                          << "v71 does not checkpoint inside a baby-batched pass.\n";
                interrupted = false;
                return 0;
            }

            std::cout << "[PM1-VTRACE-BATCH] pass " << (batchStart / activeBabyCount + 1)
                      << "/" << numPasses << " babies " << (batchStart + 1)
                      << ".." << batchEnd << " of " << babyCount << "\n";

            precompute_baby_window(batchStart, batchEnd, babyGlobalToLocal);

            // Restart giant recurrence for this baby window.  RACC is deliberately
            // NOT reset: all pass products accumulate into the same final GCD value.
            eng->copy((engine::Reg)RGPREV, (engine::Reg)RVD);
            eng->set((engine::Reg)RGCUR, 2);
            cur_k = 0;

            std::vector<uint64_t> bPrimesRun;
            size_t bDensePos = 0;
            size_t bPos = 0;
            uint64_t bSegLow = 0, bSegHigh = 0;
            uint64_t bPrime = p0u;
            if (denseRun) {
                auto it = std::lower_bound(denseStage2Primes.begin(), denseStage2Primes.end(), bPrime);
                if (it == denseStage2Primes.end() || *it != bPrime) {
                    delete eng;
                    std::cerr << "[PM1-VTRACE-BATCH] start prime not found in dense prime table.\n";
                    return -3;
                }
                bDensePos = (size_t)std::distance(denseStage2Primes.begin(), it);
            } else {
                bSegLow = bPrime;
                bSegHigh = std::min(B2u, bSegLow + SEG_SPAN - 1);
                segmented_primes_odd(bSegLow, bSegHigh, basePrimes, bPrimesRun);
                while (bPos < bPrimesRun.size() && bPrimesRun[bPos] < bPrime) ++bPos;
                if (bPos >= bPrimesRun.size() || bPrimesRun[bPos] != bPrime) {
                    delete eng;
                    std::cerr << "[PM1-VTRACE-BATCH] start prime not found in segmented sieve.\n";
                    return -3;
                }
            }
            auto bAdvancePrime = [&](uint64_t& out)->bool{
                if (denseRun) {
                    if (bDensePos + 1 >= denseStage2Primes.size()) return false;
                    out = denseStage2Primes[++bDensePos];
                    return true;
                }
                for (;;) {
                    if (bPos + 1 < bPrimesRun.size()) {
                        ++bPos; out = bPrimesRun[bPos]; return true;
                    }
                    if (bSegHigh >= B2u) return false;
                    bSegLow = bSegHigh + 1;
                    bSegHigh = std::min(B2u, bSegLow + SEG_SPAN - 1);
                    segmented_primes_odd(bSegLow, bSegHigh, basePrimes, bPrimesRun);
                    bPos = 0;
                    if (bPrimesRun.empty()) continue;
                    out = bPrimesRun[0]; return true;
                }
            };

            for (;;) {
                if (interrupted) {
                    delete eng;
                    std::cout << "\nInterrupted by user during V-trace baby-batched Stage 2. "
                              << "v71 does not checkpoint inside a baby-batched pass.\n";
                    interrupted = false;
                    return 0;
                }

                const uint64_t q = bPrime;
                ++primesSeen;
                const TracePair tp = make_pair_for_prime(q);

                bool processPair = true;
                if (tp.j == 0) {
                    processPair = (batchStart == 0);
                } else if (q == tp.qplus) {
                    if (tp.qminus > B1u && is_prime_fast(tp.qminus)) {
                        processPair = false;
                        ++skippedPairedUpper;
                    }
                }

                int32_t localBaby = -1;
                if (processPair && tp.j != 0) {
                    if (tp.j >= j2i.size() || j2i[(size_t)tp.j] < 0) {
                        delete eng;
                        std::cerr << "\n[PM1-VTRACE-BATCH] INTERNAL ERROR: no baby V_j for q=" << q
                                  << " j=" << tp.j << " D=" << D << "\n";
                        return -4;
                    }
                    const int32_t globalBaby = j2i[(size_t)tp.j];
                    localBaby = (globalBaby >= 0 && (size_t)globalBaby < babyGlobalToLocal.size())
                              ? babyGlobalToLocal[(size_t)globalBaby] : int32_t(-1);
                    if (localBaby < 0) processPair = false;
                }

                if (processPair) {
                    advance_giant_to(tp.k);
                    eng->copy((engine::Reg)RTMP, (engine::Reg)RGCUR);
                    if (tp.j == 0) {
                        eng->sub((engine::Reg)RTMP, 2);
                    } else {
                        const size_t babyReg = babyBase + (size_t)localBaby;
                        if (useNegBabyAdd) eng->add((engine::Reg)RTMP, (engine::Reg)babyReg);
                        else eng->sub_reg((engine::Reg)RTMP, (engine::Reg)babyReg);
                    }
                    eng->set_multiplicand((engine::Reg)RMUL_TMP, (engine::Reg)RTMP);
                    eng->mul((engine::Reg)RACC, (engine::Reg)RMUL_TMP);
                    ++termsAccumulated;
                }

                uint64_t next_p = 0;
                if (!bAdvancePrime(next_p)) break;
                bPrime = next_p;

                auto now = high_resolution_clock::now();
                if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
                    const double rangeFrac = double((bPrime > p0u) ? (bPrime - p0u) : 0ull) /
                                             double((B2u > p0u) ? (B2u - p0u) : 1ull);
                    const double passFrac = (double(batchStart) + rangeFrac * double(batchEnd - batchStart)) / double(babyCount);
                    const double percent = 100.0 * passFrac;
                    const double elapsedSec = duration<double>(now - start).count();
                    const double ipsPrime = (elapsedSec > 0.0) ? double(primesSeen) / elapsedSec : 0.0;
                    const double ipsTerm  = (elapsedSec > 0.0) ? double(termsAccumulated) / elapsedSec : 0.0;
                    const double etaSec = (percent > 0.0) ? elapsedSec * (100.0 - percent) / percent : 0.0;
                    int days = int(etaSec) / 86400;
                    int hours = (int(etaSec) % 86400) / 3600;
                    int minutes = (int(etaSec) % 3600) / 60;
                    int seconds = int(etaSec) % 60;
                    std::cout << "Progress V-trace batch: " << std::fixed << std::setprecision(2) << percent
                              << "% | pass=" << (batchStart / activeBabyCount + 1) << "/" << numPasses
                              << " | prime=" << bPrime
                              << " | primes(scanned)=" << primesSeen
                              << " | terms=" << termsAccumulated
                              << " | paired-skip=" << skippedPairedUpper
                              << " | p/s=" << std::fixed << std::setprecision(2) << ipsPrime
                              << " | term/s=" << std::fixed << std::setprecision(2) << ipsTerm
                              << " | ETA=" << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                              << std::flush;
                    lastDisplay = now;
                }
            }
        }
        std::cout << "\n[PM1-VTRACE-BATCH] all baby-window passes completed.\n";
    } else if (useProductTree) {
        size_t bucketPos = resumed_s2 ? (size_t)resume_idx : 0;
        if (bucketPos > productTreeBuckets.size()) {
            delete eng;
            std::cerr << "[PM1-VTRACE-PRODUCT-TREE] checkpoint bucket index is outside bucket table.\n";
            return -3;
        }
        for (; bucketPos < productTreeBuckets.size(); ++bucketPos) {
            if (interrupted) {
                double et = duration<double>(high_resolution_clock::now() - t0).count();
                save_ckpt_s2(eng, productTreeBuckets[bucketPos].k, (uint64_t)bucketPos, cur_k, et);
                delete eng;
                std::cout << "\nInterrupted by user, V-trace product-tree Stage 2 state saved at bucket "
                          << bucketPos << " k=" << cur_k << "\n";
                interrupted = false;
                return 0;
            }

            const ProductTreeBucket& bucket = productTreeBuckets[bucketPos];
            advance_giant_to(bucket.k);

            size_t termPos = 0;
            while (termPos < bucket.babyIndex.size()) {
                const size_t chunk = std::min(productTreeWidth, bucket.babyIndex.size() - termPos);
                for (size_t i = 0; i < chunk; ++i) {
                    build_trace_term_into((engine::Reg)(productTreeScratchBase + i), bucket.babyIndex[termPos + i]);
                }

                size_t active = chunk;
                while (active > 1) {
                    size_t dst = 0;
                    for (size_t i = 0; i < active; i += 2) {
                        if (i + 1 < active) {
                            eng->set_multiplicand((engine::Reg)RMUL_TMP, (engine::Reg)(productTreeScratchBase + i + 1));
                            eng->mul((engine::Reg)(productTreeScratchBase + i), (engine::Reg)RMUL_TMP);
                            if (dst != i) eng->copy((engine::Reg)(productTreeScratchBase + dst), (engine::Reg)(productTreeScratchBase + i));
                        } else if (dst != i) {
                            eng->copy((engine::Reg)(productTreeScratchBase + dst), (engine::Reg)(productTreeScratchBase + i));
                        }
                        ++dst;
                    }
                    active = dst;
                }

                eng->set_multiplicand((engine::Reg)RMUL_TMP, (engine::Reg)productTreeScratchBase);
                eng->mul((engine::Reg)RACC, (engine::Reg)RMUL_TMP);
                termsAccumulated += (uint64_t)chunk;
                termPos += chunk;
            }

            primesSeen += bucket.primeCount;
            skippedPairedUpper += bucket.skippedUpper;

            auto now = high_resolution_clock::now();
            if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
                const double percent = productTreeBuckets.empty() ? 100.0 :
                    100.0 * double(bucketPos + 1) / double(productTreeBuckets.size());
                const double elapsedSec = duration<double>(now - start).count();
                const double ipsPrime = (elapsedSec > 0.0) ? double(primesSeen) / elapsedSec : 0.0;
                const double ipsTerm  = (elapsedSec > 0.0) ? double(termsAccumulated) / elapsedSec : 0.0;
                const double etaSec = (percent > 0.0) ? elapsedSec * (100.0 - percent) / percent : 0.0;
                int days = int(etaSec) / 86400;
                int hours = (int(etaSec) % 86400) / 3600;
                int minutes = (int(etaSec) % 3600) / 60;
                int seconds = int(etaSec) % 60;
                std::cout << "Progress V-trace product-tree: " << std::fixed << std::setprecision(2) << percent
                          << "% | bucket=" << (bucketPos + 1) << "/" << productTreeBuckets.size()
                          << " | k=" << bucket.k
                          << " | primes=" << primesSeen
                          << " | terms=" << termsAccumulated
                          << " | paired-skip=" << skippedPairedUpper
                          << " | p/s=" << std::fixed << std::setprecision(2) << ipsPrime
                          << " | term/s=" << std::fixed << std::setprecision(2) << ipsTerm
                          << " | ETA=" << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                          << std::flush;
                lastDisplay = now;
            }

            auto now0 = high_resolution_clock::now();
            if (now0 - lastBackup >= std::chrono::seconds(options.backup_interval)) {
                double et = duration<double>(now0 - t0).count();
                const uint64_t saveIdx = (uint64_t)(bucketPos + 1);
                const uint64_t saveK = (bucketPos + 1 < productTreeBuckets.size()) ? productTreeBuckets[bucketPos + 1].k : bucket.k;
                std::cout << "\nBackup V-trace product-tree Stage 2 at bucket " << saveIdx
                          << " k=" << saveK << " start...\n";
                save_ckpt_s2(eng, saveK, saveIdx, cur_k, et);
                lastBackup = now0;
                std::cout << "Backup V-trace product-tree Stage 2 done.\n";
            }
        }
        std::cout << "\n[PM1-VTRACE-PRODUCT-TREE] processed " << productTreeBuckets.size()
                  << " exact buckets with width=" << productTreeWidth << ".\n";
    } else {
    for (;;) {
        if (interrupted) {
            double et = duration<double>(high_resolution_clock::now() - t0).count();
            save_ckpt_s2(eng, p_ui, idx, cur_k, et);
            delete eng;
            std::cout << "\nInterrupted by user, V-trace Stage 2 state saved at prime " << p_ui
                      << " idx=" << idx << " k=" << cur_k << "\n";
            interrupted = false;
            return 0;
        }

        const uint64_t q = p_ui;
        ++primesSeen;
        const TracePair tp = make_pair_for_prime(q);

        bool processPair = true;
        if (tp.j == 0) {
            processPair = true;
        } else if (q == tp.qplus) {
            // q is the upper member kD+j.  If the lower member kD-j was also a Stage-2 prime,
            // it was processed earlier, so skip this duplicate pair.
            if (tp.qminus > B1u && is_prime_fast(tp.qminus)) {
                processPair = false;
                ++skippedPairedUpper;
            }
        }

        if (processPair) {
            advance_giant_to(tp.k);

            eng->copy((engine::Reg)RTMP, (engine::Reg)RGCUR); // V_kD
            if (tp.j == 0) {
                eng->sub((engine::Reg)RTMP, 2);               // V_kD - V_0
            } else {
                if (tp.j >= j2i.size() || j2i[(size_t)tp.j] < 0) {
                    delete eng;
                    std::cerr << "\n[PM1-VTRACE] INTERNAL ERROR: no baby V_j for q=" << q
                              << " j=" << tp.j << " D=" << D << "\n";
                    return -4;
                }
                const size_t babyReg = babyBase + (size_t)j2i[(size_t)tp.j];
                if (useNegBabyAdd) {
                    eng->add((engine::Reg)RTMP, (engine::Reg)babyReg); // V_kD + (-V_j)
                } else {
                    eng->sub_reg((engine::Reg)RTMP, (engine::Reg)babyReg); // V_kD - V_j
                }
            }
            eng->set_multiplicand((engine::Reg)RMUL_TMP, (engine::Reg)RTMP);
            eng->mul((engine::Reg)RACC, (engine::Reg)RMUL_TMP);
            ++termsAccumulated;
        }

        uint64_t next_p = 0;
        if (!advancePrime(next_p)) break;
        p_ui = next_p;
        ++idx;

        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
            const double denom = double((B2u > p0u) ? (B2u - p0u) : 1ull);
            const double numer = double((p_ui > p0u) ? (p_ui - p0u) : 0ull);
            const double percent = 100.0 * (numer / denom);
            const double elapsedSec = duration<double>(now - start).count();
            const double ipsPrime = (elapsedSec > 0.0) ? double(primesSeen) / elapsedSec : 0.0;
            const double ipsTerm  = (elapsedSec > 0.0) ? double(termsAccumulated) / elapsedSec : 0.0;
            const double etaSec = (percent > 0.0) ? elapsedSec * (100.0 - percent) / percent : 0.0;
            int days = int(etaSec) / 86400;
            int hours = (int(etaSec) % 86400) / 3600;
            int minutes = (int(etaSec) % 3600) / 60;
            int seconds = int(etaSec) % 60;
            std::cout << "Progress V-trace: " << std::fixed << std::setprecision(2) << percent
                      << "% | prime=" << p_ui
                      << " | primes=" << primesSeen
                      << " | terms=" << termsAccumulated
                      << " | paired-skip=" << skippedPairedUpper
                      << " | p/s=" << std::fixed << std::setprecision(2) << ipsPrime
                      << " | term/s=" << std::fixed << std::setprecision(2) << ipsTerm
                      << " | ETA=" << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                      << std::flush;
            lastDisplay = now;
        }

        auto now0 = high_resolution_clock::now();
        if (now0 - lastBackup >= std::chrono::seconds(options.backup_interval)) {
            double et = duration<double>(now0 - t0).count();
            std::cout << "\nBackup V-trace Stage 2 at prime " << p_ui << " idx=" << idx << " k=" << cur_k << " start...\n";
            save_ckpt_s2(eng, p_ui, idx, cur_k, et);
            lastBackup = now0;
            std::cout << "Backup V-trace Stage 2 done.\n";
        }
    }

    }

    std::cout << "\n";
    auto t1 = high_resolution_clock::now();
    double elapsed = duration<double>(t1 - t0).count();

    std::cout << "[PM1-VTRACE] primes seen=" << primesSeen
              << " | accumulated trace terms=" << termsAccumulated
              << " | paired upper skips=" << skippedPairedUpper
              << " | term reduction=";
    if (primesSeen) {
        double red = 100.0 * double(primesSeen - termsAccumulated) / double(primesSeen);
        std::cout << std::fixed << std::setprecision(2) << red << "%\n";
    } else {
        std::cout << "0.00%\n";
    }

    mpz_class X  = compute_X_with_dots(eng, (engine::Reg)RACC, Mp);
    mpz_class g  = gcd_with_dots(X, Mp);

    auto gcd_mpz = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
        mpz_class r;
        mpz_gcd(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
        return r;
    };

    mpz_class gNew = g;
    for (const std::string& fs : options.knownFactors) {
        if (gNew == 1) break;
        mpz_class f;
        try { f = mpz_class(fs); } catch (...) { continue; }
        if (f <= 1) continue;
        mpz_class d = gcd_mpz(gNew, f);
        while (d != 1) { gNew /= d; d = gcd_mpz(gNew, f); }
    }

    bool found = (gNew != 1 && gNew != Mp);
    std::vector<std::string> newlyFound;
    auto pushFactor = [&](const std::string& fs){
        if (fs.empty() || fs == "1") return;
        if (std::find(options.knownFactors.begin(), options.knownFactors.end(), fs) != options.knownFactors.end()) return;
        options.knownFactors.push_back(fs);
        newlyFound.push_back(fs);
    };
    if (found) pushFactor(gNew.get_str());

    std::cout << "\nElapsed time (stage 2 V-trace) = " << std::fixed << std::setprecision(2) << elapsed << " s.\n";

    std::string filename = "stage2_vtrace_result_B2_" + B2.get_str() + "_p_" + std::to_string(options.exponent) + ".txt";
    if (found) {
        std::ostringstream fs;
        for (size_t i = 0; i < newlyFound.size(); ++i) { if (i) fs << ","; fs << newlyFound[i]; }
        writeStageResult(filename, "B2=" + B2.get_str() + " D=" + std::to_string(D) +
                         " terms=" + std::to_string(termsAccumulated) +
                         " primes=" + std::to_string(primesSeen) + " factor=" + fs.str());
        std::cout << "\n>>>  Factor P-1 (stage 2 V-trace) found : " << fs.str() << "\n";
    } else {
        writeStageResult(filename, "No factor P-1 V-trace up to B2=" + B2.get_str() +
                         " D=" + std::to_string(D) +
                         " terms=" + std::to_string(termsAccumulated) +
                         " primes=" + std::to_string(primesSeen));
        std::cout << "\nNo factor P-1 (stage 2 V-trace) until B2 = " << B2 << "\n";
    }

    std::remove(ckpt_file_s2.c_str());
    std::remove((ckpt_file_s2 + ".old").c_str());
    std::remove((ckpt_file_s2 + ".new").c_str());

    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
    std::cout << "Manual submission JSON:\n" << json << "\n";
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage2_vtrace", json);
    wm.appendToResultsTxt(json);

    delete eng;
    return found ? 0 : 1;
}

int App::runPM1Stage2Marin() {
    const bool useVTraceDefault = (!options.pm1_vtrace_off && !options.pm1_lowmem);
    const bool useVTrace = (!options.pm1_vtrace_off && (options.pm1_vtrace || useVTraceDefault));
    if (useVTrace) {
        if (options.pm1_lowmem) {
            std::cerr << "[PM1-VTRACE] V-trace is a normal-memory Stage 2 path; do not combine it with -pm1-lowmem/-pm1-ultralowmem.\n";
            return -1;
        }
        return runPM1Stage2MarinVTrace();
    }
    if (options.pm1_lowmem) {
        return runPM1Stage2MarinLowMem();
    }
    using namespace std::chrono;

    if (guiServer_) guiServer_->setStatus("P-1 factoring stage 2 (BSGS)");

    const uint64_t B1u = options.B1, B2u = options.B2;
    mpz_class B1 = mpz_from_u64(B1u);
    mpz_class B2 = mpz_from_u64(B2u);

    if (B2 <= B1) {
        std::cerr << "Stage 2 error B2 < B1.\n";
        if (guiServer_) guiServer_->appendLog("Stage 2 error B2 < B1.");
        return -1;
    }

    std::cout << "\nStart a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
    if (guiServer_) {
        std::ostringstream oss;
        oss << "Start a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2;
        guiServer_->appendLog(oss.str());
    }

    const uint32_t pexp = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;

    // --------- BSGS parameters ----------
    // v66: classic Stage-2 BSGS also needs a memory-aware D.
    // The previous classic path always used D=630, which stores phi(630)=144
    // baby steps plus base registers.  At p≈205M (FFT 10,485,760) this is
    // about 12.3 GiB for the register slab alone, so it cannot fit on 10 GB
    // GPUs and can throw CL_MEM_OBJECT_ALLOCATION_FAILURE.  Keep the old
    // algorithm, but select a smaller D when the register slab would exceed
    // conservative OpenCL allocation/global-memory limits.
    auto gcd_u64_classic_autod = [](uint64_t a, uint64_t b)->uint64_t{
        while (b) { uint64_t t = a % b; a = b; b = t; }
        return a;
    };
    auto classic_baby_count_for_D = [&](uint64_t d)->size_t{
        size_t c = 0;
        for (uint64_t e = 1; e < d; e += 2) {
            if (gcd_u64_classic_autod(e, d) == 1) ++c;
        }
        return c;
    };

    struct ClassicDeviceMemInfo {
        cl_ulong global = 0;
        cl_ulong maxAlloc = 0;
        std::string name;
        std::string vendor;
        bool ok = false;
    };
    auto query_classic_device_mem = [](size_t wantedDevice)->ClassicDeviceMemInfo {
        ClassicDeviceMemInfo out;
        cl_uint numPlatforms = 0;
        cl_platform_id platforms[64];
        if (clGetPlatformIDs(64, platforms, &numPlatforms) != CL_SUCCESS) return out;

        auto scan = [&](cl_device_type dtype, bool& anyFound)->bool {
            size_t idx = 0;
            for (cl_uint pi = 0; pi < numPlatforms; ++pi) {
                cl_uint numDevices = 0;
                cl_device_id devices[64];
                cl_int r = clGetDeviceIDs(platforms[pi], dtype, 64, devices, &numDevices);
                if (r != CL_SUCCESS) continue;
                anyFound = true;
                for (cl_uint di = 0; di < numDevices; ++di, ++idx) {
                    if (idx != wantedDevice) continue;
                    char dname[1024] = {0};
                    char dvendor[1024] = {0};
                    (void)clGetDeviceInfo(devices[di], CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
                    (void)clGetDeviceInfo(devices[di], CL_DEVICE_VENDOR, sizeof(dvendor), dvendor, nullptr);
                    cl_ulong global = 0, maxAlloc = 0;
                    if (clGetDeviceInfo(devices[di], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global), &global, nullptr) != CL_SUCCESS) return false;
                    if (clGetDeviceInfo(devices[di], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc), &maxAlloc, nullptr) != CL_SUCCESS) maxAlloc = 0;
                    out.global = global;
                    out.maxAlloc = maxAlloc;
                    out.name = dname;
                    out.vendor = dvendor;
                    out.ok = true;
                    return true;
                }
            }
            return false;
        };

        bool anyGpu = false;
        if (scan(CL_DEVICE_TYPE_GPU, anyGpu)) return out;
        if (!anyGpu) { bool anyAll = false; (void)scan(CL_DEVICE_TYPE_ALL, anyAll); }
        return out;
    };

    static constexpr size_t baseRegsStage2 = 13;
    static constexpr size_t baseRegsStage1 = 11;

    const size_t classicTransformN = ibdwt::transform_size(pexp);
    auto gib_classic = [](long double b)->long double { return b / 1073741824.0L; };
    auto classic_memory_plan = [&](size_t regs)->std::pair<long double,long double> {
        const long double n = (long double)classicTransformN;
        const long double regBytes = (long double)regs * n * (long double)sizeof(uint64_t);
        const long double totalBytes = regBytes
                                   + (n / 4.0L) * (long double)sizeof(uint64_t)
                                   + 3.0L * n * (long double)sizeof(uint64_t)
                                   + 2.0L * n * (long double)sizeof(uint64_t)
                                   + n * (long double)sizeof(uint8_t);
        return {regBytes, totalBytes};
    };

    const ClassicDeviceMemInfo classicMem = query_classic_device_mem((size_t)options.device_id);
    const bool allowTightClassicMem = (std::getenv("PRMERS_PM1_CLASSIC_ALLOW_TIGHT_MEM") != nullptr);
    const long double classicMaxAllocFrac = allowTightClassicMem ? 0.985L : 0.680L;
    const long double classicGlobalFrac   = allowTightClassicMem ? 0.940L : 0.700L;
    auto classic_mem_fits = [&](size_t regs)->bool {
        if (!classicMem.ok) return true;
        const auto [regBytes, totalBytes] = classic_memory_plan(regs);
        const bool pagedDisabled = (std::getenv("PRMERS_MARIN_PAGED_DISABLE") != nullptr);
        if (pagedDisabled && classicMem.maxAlloc != 0 && regBytes > (long double)classicMem.maxAlloc * classicMaxAllocFrac) return false;
        if (pagedDisabled && classicMem.global != 0 && totalBytes > (long double)classicMem.global * classicGlobalFrac) return false;
        return true;
    };

    uint64_t D = 630;
    if (const char* envD = std::getenv("PRMERS_PM1_CLASSIC_D")) {
        const uint64_t forced = std::strtoull(envD, nullptr, 10);
        if (forced >= 4 && (forced % 2ULL) == 0) {
            D = forced;
            const size_t forcedRegs = baseRegsStage2 + classic_baby_count_for_D(D);
            if (classicMem.ok && !classic_mem_fits(forcedRegs) && !allowTightClassicMem) {
                const auto [regBytes, totalBytes] = classic_memory_plan(forcedRegs);
                std::cerr << "[PM1-CLASSIC-MEM] WARNING: PRMERS_PM1_CLASSIC_D=" << D
                          << " uses " << std::fixed << std::setprecision(2) << gib_classic(regBytes)
                          << " GiB register slab and " << gib_classic(totalBytes)
                          << " GiB total before driver overhead on device "
                          << gib_classic((long double)classicMem.global) << " GiB";
                if (classicMem.maxAlloc != 0) std::cerr << ", max single allocation=" << gib_classic((long double)classicMem.maxAlloc) << " GiB";
                std::cerr << ". Set PRMERS_PM1_CLASSIC_ALLOW_TIGHT_MEM=1 only if intentional.\n";
            }
        } else {
            std::cerr << "[PM1-CLASSIC-MEM] Ignoring invalid PRMERS_PM1_CLASSIC_D=" << envD << " (need even D>=4).\n";
        }
    } else if (classicMem.ok) {
        std::cout << "[PM1-CLASSIC-MEM] auto-D memory guard: transform=" << classicTransformN
                  << " | device='" << classicMem.name << "'"
                  << " | global=" << std::fixed << std::setprecision(2) << gib_classic((long double)classicMem.global) << " GiB";
        if (classicMem.maxAlloc != 0) std::cout << " | max-alloc=" << gib_classic((long double)classicMem.maxAlloc) << " GiB";
        std::cout << "\n";

        // Ordered from the old, faster/larger D down to conservative small-D fallbacks.
        const uint64_t candidates[] = {630, 420, 330, 300, 210, 180, 150, 126, 120, 90, 84, 70, 60, 42, 30, 24, 18, 12, 10, 8, 6};
        uint64_t bestD = 0;
        size_t rejectedByMem = 0;
        for (uint64_t cand : candidates) {
            const size_t bc = classic_baby_count_for_D(cand);
            const size_t regs = baseRegsStage2 + bc;
            if (!classic_mem_fits(regs)) { ++rejectedByMem; continue; }
            bestD = cand;
            break;
        }
        if (bestD == 0) {
            std::cerr << "[PM1-CLASSIC-MEM] No classic BSGS D candidate fits this transform/device. "
                      << "Retry V-trace default, -pm1-lowmem, or set PRMERS_PM1_CLASSIC_D/PRMERS_PM1_CLASSIC_ALLOW_TIGHT_MEM if intentional.\n";
            return -2;
        }
        if (bestD != 630 || rejectedByMem) {
            const size_t bc = classic_baby_count_for_D(bestD);
            const size_t regs = baseRegsStage2 + bc;
            std::cout << "[PM1-CLASSIC-MEM] rejected " << rejectedByMem
                      << " classic D candidate(s) that were too tight; selected D=" << bestD
                      << " (baby=" << bc << ", regs=" << regs << ").\n";
        }
        D = bestD;
    }

    const uint64_t SEG_SPAN = 100000000ULL;

    #ifndef PM1_BSGS_SELFTEST
    #define PM1_BSGS_SELFTEST 0
    #endif

    // ---- Registers ----
    static constexpr size_t RSTATE    = 0;   // H (digits)
    static constexpr size_t RACC_L    = 1;   // accumulator Π(H^r - 1) (digits)
    static constexpr size_t RGIANT    = 2;   // (H^D)^k (digits)
    static constexpr size_t RGIANT_F  = 3;   // forward((H^D)^k)
    static constexpr size_t RSTEP     = 4;   // H^2 (digits)
    static constexpr size_t RX        = 5;   // H^D (digits)
    static constexpr size_t RMUL_STEP = 6;   // multiplicand(H^2)
    static constexpr size_t RMUL_RX   = 7;   // multiplicand(H^D)
    static constexpr size_t RTMP      = 8;   // scratch (forward or multiplicand depending)
    [[maybe_unused]] static constexpr size_t RREF      = 9;   // selftest pow result (digits)
    static constexpr size_t RSAVE_Q   = 10;  // pow-src scratch (digits)
    static constexpr size_t RSAVE_HQ  = 11;  // compat
    static constexpr size_t RSAVE_Q2  = 12;  // compat

    // ---- Stage2 checkpoint ----
    std::ostringstream ck2; ck2 << "pm1_s2_m_" << pexp << ".ckpt";
    const std::string ckpt_file_s2 = ck2.str();

    auto read_ckpt_s2 = [&](engine* e, const std::string& file,
                            uint64_t& saved_p, uint64_t& saved_idx, double& et,
                            uint64_t& sB1, uint64_t& sB2, uint64_t& sD)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 10) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != pexp) return -2;
        if (!f.read(reinterpret_cast<char*>(&sB1), sizeof(sB1))) return -2;
        if (!f.read(reinterpret_cast<char*>(&sB2), sizeof(sB2))) return -2;
        if (!f.read(reinterpret_cast<char*>(&sD),  sizeof(sD)))  return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_p), sizeof(saved_p))) return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_idx), sizeof(saved_idx))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;

        const size_t cksz = e->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!e->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };

    auto save_ckpt_s2 = [&](engine* e, uint64_t cur_p, uint64_t cur_idx, double et, uint64_t Dsave){
        const std::string oldf = ckpt_file_s2 + ".old", newf = ckpt_file_s2 + ".new";
        {
            File f(newf, "wb");
            int version = 10;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&pexp), sizeof(pexp))) return;
            if (!f.write(reinterpret_cast<const char*>(&B1u), sizeof(B1u))) return;
            if (!f.write(reinterpret_cast<const char*>(&B2u), sizeof(B2u))) return;
            if (!f.write(reinterpret_cast<const char*>(&Dsave), sizeof(Dsave))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_p), sizeof(cur_p))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_idx), sizeof(cur_idx))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;

            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!e->get_checkpoint(data)) return;
            if (!f.write(data.data(), cksz)) return;
            f.write_crc32();
        }
        std::remove(oldf.c_str());
        struct stat s;
        if ((stat(ckpt_file_s2.c_str(), &s) == 0) && (std::rename(ckpt_file_s2.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file_s2.c_str());
    };

    // ---- early prime check ----
    mpz_class p0;
    mpz_nextprime(p0.get_mpz_t(), B1.get_mpz_t());
    if (p0 > B2) {
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
        if (guiServer_) guiServer_->appendLog("No factor P-1 (stage 2) (no primes in range).");
        return 1;
    }
    const uint64_t p0u = mpz_get_u64(p0.get_mpz_t());

    // ---- sieve base ----
    const uint64_t root = isqrt_u64(B2u);
    const std::vector<uint32_t> basePrimes = sieve_base_primes((uint32_t)root);

    auto gcd_u64 = [](uint64_t a, uint64_t b)->uint64_t{
        while (b) { uint64_t t = a % b; a = b; b = t; }
        return a;
    };

    // residues e in [1..D-1], gcd(e,D)=1 (odd only)
    std::vector<int32_t> e2i(D, -1);
    std::vector<uint32_t> residues;
    for (uint64_t e = 1; e < D; e += 2) {
        if (gcd_u64(e, D) == 1) {
            e2i[(size_t)e] = (int32_t)residues.size();
            residues.push_back((uint32_t)e);
        }
    }
    const size_t babyCount = residues.size();
    if (!babyCount) {
        std::cerr << "Stage2 BSGS: no residues for D=" << D << "\n";
        return -2;
    }

    std::cout << "Stage 2 BSGS: D=" << D << " | baby=" << babyCount << "\n";

    // ---- engine ----
    const size_t babyBase = baseRegsStage2;
    const size_t regCount = baseRegsStage2 + babyCount;
    engine* eng = engine::create_gpu(pexp, regCount, (size_t)options.device_id, verbose);

    // ---- resume stage2? (DECLARE ONCE) ----
    uint64_t resume_idx = 0, resume_p_u64 = 0;
    double restored_time = 0.0;
    uint64_t s2B1=0, s2B2=0, s2D=0;

    int rs2 = read_ckpt_s2(eng, ckpt_file_s2, resume_p_u64, resume_idx, restored_time, s2B1, s2B2, s2D);
    bool resumed_s2 = (rs2 == 0) && (s2B1 == B1u) && (s2B2 == B2u) && (s2D == D);

    if (!resumed_s2) {
        // ---- load stage1 checkpoint and inject H ----
        engine* eng_load = engine::create_gpu(pexp, baseRegsStage1, (size_t)options.device_id, verbose);

        std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
        const std::string ckpt_file = ck.str();

        auto read_ckpt_stage1 = [&](engine* e, const std::string& file)->int{
            File f(file);
            if (!f.exists()) return -1;
            std::string backend_reason;
            if (!pm1_checkpoint_backend_matches(file, e, &backend_reason)) {
                std::cerr << "[PM1] Stage 1 checkpoint ignored: " << backend_reason << "\n";
                return -3;
            }
            int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 3) return -2;
            uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != pexp) return -2;
            uint32_t ri = 0; double et = 0.0;
            if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;

            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!e->set_checkpoint(data)) return -2;

            // trailing metadata (compat)
            uint64_t tmp64;
            for (int i = 0; i < 4; ++i) if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
            uint8_t inlot=0; if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;

            uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
            if (eacc_len) { std::string skip(eacc_len, '\0'); if (!f.read(skip.data(), eacc_len)) return -2; }

            uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
            if (wbits_len) { std::string skip(wbits_len, '\0'); if (!f.read(skip.data(), wbits_len)) return -2; }

            uint64_t chunkIdx=0, startP=0; uint8_t first=0; uint64_t processedBits=0, bitsInChunk=0;
            if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
            if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
            if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
            if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
            if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;

            if (!f.check_crc32()) return -2;
            return 0;
        };

        int rr = read_ckpt_stage1(eng_load, ckpt_file);
        if (rr < 0) rr = read_ckpt_stage1(eng_load, ckpt_file + ".old");
        if (rr != 0) {
            delete eng_load; delete eng;
            std::cerr << "Stage 2: cannot load pm1 stage1 checkpoint.\n";
            if (guiServer_) guiServer_->appendLog("Stage 2: cannot load pm1 stage1 checkpoint.");
            return -2;
        }

        mpz_t H; mpz_init(H);
        eng_load->get_mpz(H, (engine::Reg)RSTATE);
        delete eng_load;

        eng->set_mpz((engine::Reg)RSTATE, H);
        mpz_clear(H);

        // ---- H^2 multiplicand ----
        eng->copy((engine::Reg)RSTEP, (engine::Reg)RSTATE);
        eng->square_mul((engine::Reg)RSTEP);
        eng->set_multiplicand((engine::Reg)RMUL_STEP, (engine::Reg)RSTEP);

        // ---- baby steps H^e ----
        std::cout << "Precomputing baby steps (H^e, gcd(e,D)=1)...\n";
        eng->copy((engine::Reg)RTMP, (engine::Reg)RSTATE); // RTMP = H^1

        size_t stored = 0;
        int pct = -1;

        for (uint64_t e = 1; e < D; e += 2) {
            int32_t bi = e2i[(size_t)e];
            if (bi >= 0) {
                const size_t slot = babyBase + (size_t)bi;
                eng->copy((engine::Reg)slot, (engine::Reg)RTMP);               // digits
                eng->set_multiplicand((engine::Reg)slot, (engine::Reg)slot);  // multiplicand(H^e)
                ++stored;
                int newPct = int((stored * 100ull) / std::max<size_t>(1, babyCount));
                if (newPct > pct) { pct = newPct; std::cout << "\rBaby steps: " << pct << "%" << std::flush; }
            }
            eng->mul((engine::Reg)RTMP, (engine::Reg)RMUL_STEP); // RTMP *= H^2
        }
        std::cout << "\rBaby steps: 100%\n";
        std::cout << "Stored baby steps: " << babyCount << " (D=" << D << ")\n";

        // ---- RX = H^D (NO aliasing pow) ----
        eng->copy((engine::Reg)RSAVE_Q, (engine::Reg)RSTATE);
        eng->pow((engine::Reg)RX, (engine::Reg)RSAVE_Q, D);
        eng->set_multiplicand((engine::Reg)RMUL_RX, (engine::Reg)RX);

        // ---- init giant = (H^D)^(floor(p0/D)) ----
        const uint64_t k0 = p0u / D;
        eng->copy((engine::Reg)RSAVE_Q, (engine::Reg)RX);
        eng->pow((engine::Reg)RGIANT, (engine::Reg)RSAVE_Q, k0);
        eng->set_multiplicand2((engine::Reg)RGIANT_F, (engine::Reg)RGIANT);

        eng->set((engine::Reg)RACC_L, 1);
    } else {
        // resume: just rebuild forward cache (safe)
        eng->set_multiplicand2((engine::Reg)RGIANT_F, (engine::Reg)RGIANT);
    }

    // ---- timers + start prime ----
    uint64_t idx = 0;
    uint64_t p_ui = p0u;

    auto t0 = high_resolution_clock::now();
    if (resumed_s2) {
        idx = resume_idx;
        p_ui = resume_p_u64;
        std::cout << "Resuming Stage 2 from checkpoint at prime " << p_ui << " (idx=" << idx << ") D=" << D << "\n";
        t0 = high_resolution_clock::now() - duration_cast<high_resolution_clock::duration>(duration<double>(restored_time));
    }

    auto lastBackup  = high_resolution_clock::now();
    auto lastDisplay = high_resolution_clock::now();
    auto start       = t0;
    auto start_sys   = std::chrono::system_clock::now();

    // ---- prime stream ----
    std::vector<uint64_t> primesRun;
    uint64_t segLowRun = p_ui;
    uint64_t segHighRun = std::min(B2u, segLowRun + SEG_SPAN - 1);
    segmented_primes_odd(segLowRun, segHighRun, basePrimes, primesRun);

    size_t posRun = 0;
    while (posRun < primesRun.size() && primesRun[posRun] < p_ui) ++posRun;
    if (posRun >= primesRun.size() || primesRun[posRun] != p_ui) {
        delete eng;
        std::cerr << "Stage 2: start prime not found in segmented sieve.\n";
        return -3;
    }

    auto advancePrime = [&](uint64_t& out)->bool{
        for (;;) {
            if (posRun + 1 < primesRun.size()) {
                ++posRun;
                out = primesRun[posRun];
                return true;
            }
            if (segHighRun >= B2u) return false;
            segLowRun = segHighRun + 1;
            segHighRun = std::min(B2u, segLowRun + SEG_SPAN - 1);
            segmented_primes_odd(segLowRun, segHighRun, basePrimes, primesRun);
            posRun = 0;
            if (primesRun.empty()) continue;
            out = primesRun[0];
            return true;
        }
    };

    // current k for giant
    uint64_t cur_k = p_ui / D;

    // ---- main loop ----
    for (;;) {
        if (interrupted) {
            double et = duration<double>(high_resolution_clock::now() - t0).count();
            save_ckpt_s2(eng, p_ui, idx, et, D);
            delete eng;
            std::cout << "\nInterrupted by user, Stage 2 state saved at prime " << p_ui << " idx=" << idx << "\n";
            interrupted = false;
            return 0;
        }

        const uint64_t r = p_ui;
        const uint64_t k = r / D;
        const uint64_t e = r - k * D;

        while (cur_k < k) {
            eng->mul((engine::Reg)RGIANT, (engine::Reg)RMUL_RX);
            ++cur_k;
            eng->set_multiplicand2((engine::Reg)RGIANT_F, (engine::Reg)RGIANT);
        }

        const int32_t bi = (e < D) ? e2i[(size_t)e] : -1;
        if (bi < 0) {
            std::cerr << "\n[BSGS] INTERNAL ERROR: residue not found for prime r=" << r
                      << " (e=" << e << ", D=" << D << ")\n";
            std::abort();
        }

        const size_t babyReg = babyBase + (size_t)bi;
        eng->copy((engine::Reg)RTMP, (engine::Reg)RGIANT_F);   // forward
        eng->mul_new((engine::Reg)RTMP, (engine::Reg)babyReg); // -> digits H^r

        #if PM1_BSGS_SELFTEST
        if (idx < 20) {
            eng->copy((engine::Reg)RSAVE_Q, (engine::Reg)RSTATE);
            eng->pow((engine::Reg)RREF, (engine::Reg)RSAVE_Q, r);
            mpz_t a,b; mpz_init(a); mpz_init(b);
            eng->get_mpz(a, (engine::Reg)RTMP);
            eng->get_mpz(b, (engine::Reg)RREF);
            if (mpz_cmp(a,b) != 0) {
                std::cerr << "\n[BSGS] mismatch at prime r=" << r << " D=" << D
                          << " k=" << k << " e=" << e << " bi=" << bi << "\n";
                mpz_clear(a); mpz_clear(b);
                std::abort();
            }
            mpz_clear(a); mpz_clear(b);
        }
        #endif

        // Acc *= (H^r - 1) (reuse RTMP as multiplicand)
        eng->sub((engine::Reg)RTMP, 1);
        eng->set_multiplicand((engine::Reg)RTMP, (engine::Reg)RTMP);
        eng->mul((engine::Reg)RACC_L, (engine::Reg)RTMP);

        uint64_t next_p = 0;
        if (!advancePrime(next_p)) break;
        p_ui = next_p;
        ++idx;

        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
            const size_t done_abs = idx + 1;
            const double denom = double((B2u > p0u) ? (B2u - p0u) : 1ull);
            const double numer = double((p_ui > p0u) ? (p_ui - p0u) : 0ull);
            const double percent = 100.0 * (numer / denom);
            const double elapsedSec = duration<double>(now - start).count();
            const double ips = (elapsedSec > 0.0) ? double(done_abs) / elapsedSec : 0.0;
            const double etaSec = (percent > 0.0) ? elapsedSec * (100.0 - percent) / percent : 0.0;

            int days = int(etaSec) / 86400;
            int hours = (int(etaSec) % 86400) / 3600;
            int minutes = (int(etaSec) % 3600) / 60;
            int seconds = int(etaSec) % 60;

            std::cout << "Progress: " << std::fixed << std::setprecision(2) << percent
                      << "% | Iter: " << done_abs
                      << " | prime: " << p_ui
                      << " | Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s"
                      << " | IPS: " << std::fixed << std::setprecision(2) << ips
                      << " | ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                      << std::flush;

            lastDisplay = now;
        }

        auto now0 = high_resolution_clock::now();
        if (now0 - lastBackup >= std::chrono::seconds(options.backup_interval)) {
            double et = duration<double>(now0 - t0).count();
            std::cout << "\nBackup Stage 2 at prime " << p_ui << " idx=" << idx << " start...\n";
            save_ckpt_s2(eng, p_ui, idx, et, D);
            lastBackup = now0;
            std::cout << "Backup Stage 2 done.\n";
        }
    }

    std::cout << "\n";

    // ---- end timing ----
    auto end_sys = std::chrono::system_clock::now();
    (void)end_sys; // keep if you use it in resume file timestamps elsewhere

    auto t1 = high_resolution_clock::now();
    double elapsed = duration<double>(t1 - t0).count();

    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    mpz_class X  = compute_X_with_dots(eng, (engine::Reg)RACC_L, Mp);
    mpz_class g  = gcd_with_dots(X, Mp);

    auto gcd_mpz = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
        mpz_class r;
        mpz_gcd(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
        return r;
    };

    mpz_class gNew = g;
    for (const std::string& fs : options.knownFactors) {
        if (gNew == 1) break;
        mpz_class f;
        try { f = mpz_class(fs); } catch (...) { continue; }
        if (f <= 1) continue;
        mpz_class d = gcd_mpz(gNew, f);
        while (d != 1) { gNew /= d; d = gcd_mpz(gNew, f); }
    }

    bool found = (gNew != 1 && gNew != Mp);

    std::vector<std::string> newlyFound;
    auto pushFactor = [&](const std::string& s){
        if (s.empty() || s == "1") return;
        if (std::find(options.knownFactors.begin(), options.knownFactors.end(), s) != options.knownFactors.end()) return;
        options.knownFactors.push_back(s);
        newlyFound.push_back(s);
    };

    if (found) pushFactor(gNew.get_str());

    std::cout << "\nElapsed time (stage 2) = " << std::fixed << std::setprecision(2) << elapsed << " s.\n";

    std::string filename = "stage2_result_B2_" + B2.get_str() + "_p_" + std::to_string(options.exponent) + ".txt";
    if (found) {
        std::ostringstream fs;
        for (size_t i = 0; i < newlyFound.size(); ++i) { if (i) fs << ","; fs << newlyFound[i]; }
        writeStageResult(filename, "B2=" + B2.get_str() + "  factor=" + fs.str());
        std::cout << "\n>>>  Factor P-1 (stage 2) found : " << fs.str() << "\n";
    } else {
        writeStageResult(filename, "No factor P-1 up to B2=" + B2.get_str());
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << "\n";
    }

    // cleanup stage2 ckpts
    std::remove(ckpt_file_s2.c_str());
    std::remove((ckpt_file_s2 + ".old").c_str());
    std::remove((ckpt_file_s2 + ".new").c_str());

    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
    std::cout << "Manual submission JSON:\n" << json << "\n";
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage2", json);
    wm.appendToResultsTxt(json);

    delete eng;
    return found ? 0 : 1;
}
#include <array>
#include <cstdlib>

// ------------------------------------------------------------
// FAST SLn-torus Stage-1 (Marin engine)
// Key speedups:
//  - No torus_copy(): swap register banks (swap indices only)
//  - Multiply-by-x is O(n) (shift+reduce), no NTT multiplications
// ------------------------------------------------------------

static inline void add_times_small(engine* eng, const engine::Reg dst, const engine::Reg src, const uint32_t a)
{
    // a is small (from {1,3,5,7,11,13,17,19} typically)
    switch (a) {
        case 1:  eng->add(dst, src); break;
        case 3:  eng->add(dst, src); eng->add(dst, src); eng->add(dst, src); break;
        case 5:  eng->add(dst, src); eng->add(dst, src); eng->add(dst, src); eng->add(dst, src); eng->add(dst, src); break;
        case 7:  for (int i=0;i<7;i++)  eng->add(dst, src); break;
        case 11: for (int i=0;i<11;i++) eng->add(dst, src); break;
        case 13: for (int i=0;i<13;i++) eng->add(dst, src); break;
        case 17: for (int i=0;i<17;i++) eng->add(dst, src); break;
        case 19: for (int i=0;i<19;i++) eng->add(dst, src); break;
        default: for (uint32_t i = 0; i < a; ++i) eng->add(dst, src); break;
    }
}

template<int N>
static inline void torus_set_identity(engine* eng, const std::array<engine::Reg, N>& v)
{
    eng->set(v[0], 1u);
    for (int i = 1; i < N; ++i) eng->set(v[i], 0u);
}

template<int N>
static inline void torus_mul_by_x_ax1(engine* eng,
                                      const std::array<engine::Reg, N>& out,
                                      const std::array<engine::Reg, N>& in,
                                      const uint32_t a_coeff)
{
    // modulus: x^N = a*x + 1
    // (c0 + c1 x + ... + c_{N-1} x^{N-1})*x
    // = c_{N-1} + (c0 + a*c_{N-1}) x + c1 x^2 + ... + c_{N-2} x^{N-1}

    eng->copy(out[0], in[N - 1]);         // out0 = c_{N-1}

    eng->copy(out[1], in[0]);             // out1 = c0
    add_times_small(eng, out[1], in[N - 1], a_coeff); // out1 += a*c_{N-1}

    for (int i = 2; i < N; ++i) {
        eng->copy(out[i], in[i - 1]);     // out[i] = c_{i-1}
    }
}

template<int N>
static inline void torus_sqr_ax1(engine* eng,
                                 const std::array<engine::Reg, N>& out,
                                 const std::array<engine::Reg, N>& x,
                                 const std::array<engine::Reg, N>& xMul,
                                 const engine::Reg tmp,
                                 const uint32_t a_coeff)
{
    // out = x^2 in R=(Z/NZ)[X]/(X^N - aX - 1), N odd
    for (int i = 0; i < N; ++i) eng->set(out[i], 0u);

    // prepare multiplicands of x[j]
    for (int j = 0; j < N; ++j) {
        eng->copy(xMul[j], x[j]);
        eng->set_multiplicand(xMul[j], xMul[j]);
    }

    for (int i = 0; i < N; ++i) {
        // diagonal term: x[i]^2
        eng->copy(tmp, x[i]);
        eng->square_mul(tmp);

        {
            const int k = i + i;
            if (k < N) {
                eng->add(out[k], tmp);
            } else {
                const int m = k - N; // 0..N-2
                eng->add(out[m], tmp);
                add_times_small(eng, out[m + 1], tmp, a_coeff);
            }
        }

        // cross terms: 2*x[i]*x[j]
        for (int j = i + 1; j < N; ++j) {
            eng->copy(tmp, x[i]);
            eng->mul(tmp, xMul[j], 2u);

            const int k = i + j;
            if (k < N) {
                eng->add(out[k], tmp);
            } else {
                const int m = k - N; // 0..N-2
                eng->add(out[m], tmp);
                add_times_small(eng, out[m + 1], tmp, a_coeff);
            }
        }
    }
}

// -------- n=2 (quadratic): x^2 = t x - 1 --------

static inline void torus2_mul_by_x(engine* eng,
                                  const std::array<engine::Reg, 2>& out,
                                  const std::array<engine::Reg, 2>& in,
                                  const uint32_t t)
{
    // (c0 + c1 x)*x = -c1 + (c0 + t*c1) x
    eng->set(out[0], 0u);
    eng->sub_reg(out[0], in[1]);      // out0 = -c1

    eng->copy(out[1], in[0]);         // out1 = c0
    for (uint32_t i = 0; i < t; ++i) eng->add(out[1], in[1]); // out1 += t*c1
}

static inline void torus2_sqr(engine* eng,
                              const std::array<engine::Reg, 2>& out,
                              const std::array<engine::Reg, 2>& a,
                              const std::array<engine::Reg, 2>& aMul,
                              const engine::Reg tmpA1sq,
                              const uint32_t t)
{
    // (a0+a1x)^2, x^2 = t x - 1
    // c0 = a0^2 - a1^2
    // c1 = 2*a0*a1 + t*a1^2

    eng->copy(aMul[1], a[1]);
    eng->set_multiplicand(aMul[1], aMul[1]);

    // tmpA1sq = a1^2
    eng->copy(tmpA1sq, a[1]);
    eng->square_mul(tmpA1sq);

    // out0 = a0^2 - a1^2
    eng->copy(out[0], a[0]);
    eng->square_mul(out[0]);
    eng->sub_reg(out[0], tmpA1sq);

    // out1 = 2*a0*a1 + t*a1^2
    eng->copy(out[1], a[0]);
    eng->mul(out[1], aMul[1], 2u);
    for (uint32_t i = 0; i < t; ++i) eng->add(out[1], tmpA1sq);
}

static uint32_t pick_quadratic_t(const mpz_class& Mp)
{
    const uint32_t cand[] = {5u, 6u, 7u, 10u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u};
    for (size_t i = 0; i < sizeof(cand)/sizeof(cand[0]); ++i) {
        const uint32_t t = cand[i];
        mpz_class delta = mpz_class(t) * mpz_class(t) - 4; // t^2 - 4
        if (mpz_jacobi(delta.get_mpz_t(), Mp.get_mpz_t()) == -1) return t;
    }
    return 5u;
}

template<int N>
static inline bool torus_gcd_check(engine* eng,
                                  const mpz_class& Mp,
                                  const std::array<engine::Reg, N>& S,
                                  mpz_t& z,
                                  mpz_class& outFactor)
{
    for (int i = 0; i < N; ++i) {
        eng->get_mpz(z, S[i]);
        mpz_class diff(z);
        if (i == 0) {
            diff -= 1;
            if (diff < 0) diff += Mp; // avoid mpz_mod
        }
        if (diff == 0) continue;

        mpz_class g;
        mpz_gcd(g.get_mpz_t(), diff.get_mpz_t(), Mp.get_mpz_t());
        if (g > 1 && g < Mp) { outFactor = g; return true; }
    }
    return false;
}

template<int N>
static int run_pm1_stage1_torus_oddN(uint32_t pexp,
                                    uint64_t B1,
                                    int device_id,
                                    bool verbose,
                                    ui::WebGuiServer* gui,
                                    const mpz_class& E,
                                    const mp_bitcnt_t ebits,
                                    const uint32_t a_coeff)
{
    const mpz_class Mp = (mpz_class(1) << pexp) - 1;

    // regs: S[N], T[N], MUL[N], TMP
    size_t base = 0;
    std::array<engine::Reg, N> S{}, T{}, MUL{};
    for (int i = 0; i < N; ++i) S[i]   = base++;
    for (int i = 0; i < N; ++i) T[i]   = base++;
    for (int i = 0; i < N; ++i) MUL[i] = base++;
    const engine::Reg TMP = base++;
    const size_t regCount = base;

    engine* eng = engine::create_gpu(pexp, regCount, (size_t)device_id, verbose);

    torus_set_identity<N>(eng, S);

    const mp_bitcnt_t mask = (mp_bitcnt_t)1023u;

    for (mp_bitcnt_t i = ebits; i-- > 0; ) {
        // S = S^2
        torus_sqr_ax1<N>(eng, T, S, MUL, TMP, a_coeff);
        std::swap(S, T); // NO COPY of big registers

        // if bit set: S = S * x  (cheap!)
        if (mpz_tstbit(E.get_mpz_t(), i) != 0) {
            torus_mul_by_x_ax1<N>(eng, T, S, a_coeff);
            std::swap(S, T);
        }

        if ((i & mask) == 0) {
            std::cout << "[TORUS] n=" << N << " a=" << a_coeff
                      << " bit " << (unsigned long long)i
                      << "/" << (unsigned long long)ebits << "\r" << std::flush;
            if (interrupted) { std::cout << "\nInterrupted\n"; delete eng; return 0; }
        }
    }
    std::cout << "\n[TORUS] n=" << N << " a=" << a_coeff << " done, gcd...\n";

    mpz_t z; mpz_init(z);
    mpz_class factor;
    const bool ok = torus_gcd_check<N>(eng, Mp, S, z, factor);
    mpz_clear(z);
    delete eng;

    if (ok) {
        std::cout << "[TORUS] FACTOR FOUND (n=" << N << ", a=" << a_coeff << "): " << factor.get_str() << "\n";
        if (gui) { std::ostringstream oss; oss << "[TORUS] FACTOR FOUND (n=" << N << ", a=" << a_coeff << "): " << factor.get_str(); gui->appendLog(oss.str()); }
        return 1;
    }
    std::cout << "[TORUS] n=" << N << " a=" << a_coeff << " : no factor\n";
    return 0;
}

static int run_pm1_stage1_torus_n2(uint32_t pexp,
                                  uint64_t B1,
                                  int device_id,
                                  bool verbose,
                                  ui::WebGuiServer* gui,
                                  const mpz_class& E,
                                  const mp_bitcnt_t ebits)
{
    const mpz_class Mp = (mpz_class(1) << pexp) - 1;
    const uint32_t t = pick_quadratic_t(Mp);

    std::cout << "[TORUS] n=2 using t=" << t << " (Jacobi(t^2-4,N)=-1)\n";
    if (gui) { std::ostringstream oss; oss << "[TORUS] n=2 using t=" << t; gui->appendLog(oss.str()); }

    // regs: S[2], T[2], MUL[2], TMP (a1^2)
    size_t base = 0;
    std::array<engine::Reg, 2> S{}, T{}, MUL{};
    for (int i = 0; i < 2; ++i) S[i]   = base++;
    for (int i = 0; i < 2; ++i) T[i]   = base++;
    for (int i = 0; i < 2; ++i) MUL[i] = base++;
    const engine::Reg TMPA1SQ = base++;
    const size_t regCount = base;

    engine* eng = engine::create_gpu(pexp, regCount, (size_t)device_id, verbose);

    torus_set_identity<2>(eng, S);

    const mp_bitcnt_t mask = (mp_bitcnt_t)1023u;

    for (mp_bitcnt_t i = ebits; i-- > 0; ) {
        torus2_sqr(eng, T, S, MUL, TMPA1SQ, t);
        std::swap(S, T);

        if (mpz_tstbit(E.get_mpz_t(), i) != 0) {
            torus2_mul_by_x(eng, T, S, t);
            std::swap(S, T);
        }

        if ((i & mask) == 0) {
            std::cout << "[TORUS] n=2 bit " << (unsigned long long)i
                      << "/" << (unsigned long long)ebits << "\r" << std::flush;
            if (interrupted) { std::cout << "\nInterrupted\n"; delete eng; return 0; }
        }
    }
    std::cout << "\n[TORUS] n=2 done, gcd...\n";

    mpz_t z; mpz_init(z);
    mpz_class factor;
    const bool ok = torus_gcd_check<2>(eng, Mp, S, z, factor);
    mpz_clear(z);
    delete eng;

    if (ok) {
        std::cout << "[TORUS] FACTOR FOUND (n=2): " << factor.get_str() << "\n";
        if (gui) { std::ostringstream oss; oss << "[TORUS] FACTOR FOUND (n=2): " << factor.get_str(); gui->appendLog(oss.str()); }
        return 1;
    }
    std::cout << "[TORUS] n=2 : no factor\n";
    return 0;
}

static int run_pm1_stage1_torus_oneN(uint32_t pexp,
                                     uint64_t B1,
                                     int device_id,
                                     bool verbose,
                                     ui::WebGuiServer* gui,
                                     const int n,
                                     const bool forceNonP1)
{
    const mpz_class Mp = (mpz_class(1) << pexp) - 1;

    // Build E once
    mpz_class E = buildE(B1);

    // NON-P-1: remove P^k from E (do NOT clamp B1)
    if (forceNonP1 && (uint64_t)pexp <= B1) {
        uint64_t pPow = (uint64_t)pexp;
        while (pPow <= B1 / (uint64_t)pexp) pPow *= (uint64_t)pexp;
        mpz_t pp;
        mpz_init(pp);
        mpz_import(pp, 1, 1, sizeof(pPow), 0, 0, &pPow); // charge uint64_t -> mpz_t (portable)
        mpz_divexact(E.get_mpz_t(), E.get_mpz_t(), pp);
        mpz_clear(pp);

        std::cout << "[TORUS] NON-P-1: removed P^k, kPow=" << pPow << "\n";
        if (gui) { std::ostringstream oss; oss << "[TORUS] NON-P-1: removed P^k, kPow=" << pPow; gui->appendLog(oss.str()); }
    }

    const size_t ebits_sz = mpz_sizeinbase(E.get_mpz_t(), 2);
    const mp_bitcnt_t ebits = (mp_bitcnt_t)ebits_sz;

    std::cout << "[TORUS] n=" << n << " | B1=" << B1 << " | E bits=" << (unsigned long long)ebits << "\n";
    if (gui) {
        std::ostringstream oss;
        oss << "[TORUS] n=" << n << " | B1=" << B1 << " | E bits=" << (unsigned long long)ebits;
        gui->appendLog(oss.str());
    }

    if (n == 2) {
        return run_pm1_stage1_torus_n2(pexp, B1, device_id, verbose, gui, E, ebits);
    }

    const uint32_t a_list[] = {1u,3u,5u,7u,11u,13u,17u,19u};

    for (size_t ai = 0; ai < sizeof(a_list)/sizeof(a_list[0]); ++ai) {
        const uint32_t a_coeff = a_list[ai];

        std::cout << "[TORUS] n=" << n << " trying a=" << a_coeff << "\n";
        if (gui) { std::ostringstream oss; oss << "[TORUS] n=" << n << " trying a=" << a_coeff; gui->appendLog(oss.str()); }

        int ok = 0;
        switch (n) {
            case 3:  ok = run_pm1_stage1_torus_oddN<3 >(pexp, B1, device_id, verbose, gui, E, ebits, a_coeff); break;
            case 5:  ok = run_pm1_stage1_torus_oddN<5 >(pexp, B1, device_id, verbose, gui, E, ebits, a_coeff); break;
            case 7:  ok = run_pm1_stage1_torus_oddN<7 >(pexp, B1, device_id, verbose, gui, E, ebits, a_coeff); break;
            case 13: ok = run_pm1_stage1_torus_oddN<13>(pexp, B1, device_id, verbose, gui, E, ebits, a_coeff); break;
            default:
                std::cerr << "[TORUS] unsupported n=" << n << " (allowed 2,3,5,7,13)\n";
                return 0;
        }
        if (ok) return 1;
        if (interrupted) return 0;
    }

    std::cout << "[TORUS] n=" << n << " : no factor\n";
    return 0;
}

// Run torus suite
int App::runPM1Stage1SLnTorusMarin()
{
    const uint32_t pexp = (uint32_t)options.exponent;
    const bool verbose = options.debug;
    ui::WebGuiServer* gui = guiServer_ ? guiServer_.get() : nullptr;

    // env option: PRMERS_TORUS_NONP1=1
    bool forceNonP1 = false;
    if (const char* e = std::getenv("PRMERS_TORUS_NONP1")) {
        if (e[0] == '1') forceNonP1 = true;
    }

    uint64_t B1 = (uint64_t)options.B1;

    std::cout << "[TORUS] Stage-1 on M(" << pexp << "), B1=" << B1
              << (forceNonP1 ? " (NON-P-1 ON)" : "") << "\n";
    if (gui) {
        std::ostringstream oss;
        oss << "[TORUS] Stage-1 on M(" << pexp << "), B1=" << B1
            << (forceNonP1 ? " (NON-P-1 ON)" : "");
        gui->appendLog(oss.str());
    }

    // Default degrees
    static const int degreesDefault[] = { 3, 5, 7 };

    const int* degrees = degreesDefault;
    size_t degreesCount = sizeof(degreesDefault) / sizeof(degreesDefault[0]);

    // If options.B4 != 0, force a single n = B4
    const int forcedN = (int)options.B4;
    if (forcedN != 1) {
        if (!(forcedN == 2 || forcedN == 3 || forcedN == 5 || forcedN == 7 || forcedN == 13)) {
            std::cerr << "[TORUS] invalid B4=" << forcedN << " (allowed: 2,3,5,7,13)\n";
            return 0;
        }
        degrees = &forcedN;
        degreesCount = 1;

        std::cout << "[TORUS] B4 forced: testing only n=" << forcedN << "\n";
        if (gui) { std::ostringstream oss; oss << "[TORUS] B4 forced: testing only n=" << forcedN; gui->appendLog(oss.str()); }
    }

    for (size_t idx = 0; idx < degreesCount; ++idx) {
        const int n = degrees[idx];
        const int ok = run_pm1_stage1_torus_oneN(pexp, B1, (int)options.device_id, verbose, gui, n, forceNonP1);
        if (ok != 0) return 1;
        if (interrupted) return 0;
    }
    return 0;
}



/* ===== n^K Stage-2 (Topics in advanced scientific computation. by: Crandall, Richard E) ===== */
/* Fast b^{n^K} (Stirling init + z-chain); product of differences; GCD. */
int App::runPM1Stage2MarinNKVersion() {
    using namespace std::chrono;
    const uint32_t pexp  = (uint32_t)options.exponent;
    const uint32_t K     = (uint32_t)options.K;
    const uint64_t nmax  = (uint64_t)options.nmax;
    if (K == 0 || nmax == 0) { std::cout << "Nothing to do (K=0 or nmax=0)\n"; return 0; }
    if (guiServer_) { std::ostringstream oss; oss << "P-1 Stage 2 (n^K) — K=" << K << ", nmax=" << nmax; guiServer_->setStatus(oss.str()); }

    const size_t RSTATE=0, RACC=1, RTMP=2, RPOW=3, RDIFF=4, RONE=5;
    size_t regCount = 6 + (size_t)K + 1 + (size_t)nmax;

    engine* eng_s1 = engine::create_gpu(pexp, 11, (size_t)options.device_id, options.debug);
    std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
    auto read_ckpt_s1 = [&](engine* e, const std::string& file)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 3) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != pexp) return -2;
        uint32_t ri = 0; double et = 0.0;
        if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = e->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!e->set_checkpoint(data)) return -2;
        uint64_t tmp64;
        if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
        if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
        if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
        if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
        uint8_t inlot=0; if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
        uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
        if (eacc_len) { std::string skip; skip.resize(eacc_len); if (!f.read(skip.data(), eacc_len)) return -2; }
        uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
        if (wbits_len) { std::string skip; skip.resize(wbits_len); if (!f.read(skip.data(), wbits_len)) return -2; }
        uint64_t chunkIdx=0, startP=0; uint8_t first=0; uint64_t processedBits=0, bitsInChunk=0;
        if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
        if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
        if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
        if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
        if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };
    int rr = read_ckpt_s1(eng_s1, ck.str());
    if (rr < 0) rr = read_ckpt_s1(eng_s1, ck.str() + ".old");
    if (rr != 0) { delete eng_s1; std::cout << "Stage 2 (n^K): cannot load stage-1 checkpoint\n"; if (guiServer_) { std::ostringstream oss; oss << "Stage 2 (n^K): cannot load stage-1 checkpoint"; guiServer_->appendLog(oss.str()); } return -2; }
    mpz_t H; mpz_init(H); eng_s1->get_mpz(H, (engine::Reg)0); delete eng_s1;

    engine* eng = engine::create_gpu(pexp, regCount, (size_t)options.device_id, options.debug);
    eng->set_mpz((engine::Reg)RSTATE, H);
    mpz_clear(H);

    auto pow_big_mpz = [&](size_t dst, size_t baseReg, const mpz_class& e)->bool{
        eng->set((engine::Reg)dst, 1);
        if (mpz_sgn(e.get_mpz_t()) == 0) return true;
        eng->set_multiplicand((engine::Reg)RTMP, (engine::Reg)baseReg);
        size_t nb = mpz_sizeinbase(e.get_mpz_t(), 2);
        const size_t chunk = 4096;
        for (size_t k = 0; k < nb; ++k) {
            eng->square_mul((engine::Reg)dst);
            size_t bit = nb - 1 - k;
            if (mpz_tstbit(e.get_mpz_t(), bit)) eng->mul((engine::Reg)dst, (engine::Reg)RTMP);
            if ((k % chunk) == 0 && interrupted) return false;
        }
        return true;
    };

    std::vector<std::vector<mpz_class>> S(K+1, std::vector<mpz_class>(K+1));
    S[0][0] = 1;
    for (uint32_t n = 1; n <= K; ++n) { S[n][0] = mpz_class(0); for (uint32_t j = 1; j <= n; ++j) S[n][j] = mpz_class(j) * S[n-1][j] + S[n-1][j-1]; }
    std::vector<mpz_class> fact(K+1); fact[0] = 1; for (uint32_t j = 1; j <= K; ++j) fact[j] = fact[j-1] * j;

    size_t Z0 = 6, VAL0 = Z0 + (size_t)K + 1;
    eng->set((engine::Reg)(Z0 + 0), 1);
    for (uint32_t j = 1; j <= K; ++j) { mpz_class e = fact[j] * S[K][j]; if (!pow_big_mpz(Z0 + j, RSTATE, e)) { delete eng; std::cout << "Interrupted during initialization\n"; return 0; } }

    eng->set((engine::Reg)RACC, 1);
    eng->set((engine::Reg)RONE, 1);

    auto t0 = high_resolution_clock::now();
    auto last = high_resolution_clock::now();

    for (uint64_t m = 1; m <= nmax; ++m) {
        for (uint32_t q = 0; q < K; ++q) { 
            eng->set_multiplicand((engine::Reg)RPOW, (engine::Reg)(Z0 + q + 1)); 
            eng->mul((engine::Reg)(Z0 + q), (engine::Reg)RPOW); 
            }
        eng->copy((engine::Reg)(VAL0 + (m-1)), (engine::Reg)(Z0 + 0));
        auto now = high_resolution_clock::now();
        if (duration_cast<milliseconds>(now - last).count() >= 300) {
            double done = double(m), total = double(nmax);
            double elapsed = duration<double>(now - t0).count();
            double ips = done / std::max(1e-9, elapsed);
            double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
            std::cout << "build " << m << "/" << nmax << " | " << std::fixed << std::setprecision(2) << (done*100.0/total) << "% | ETA " << (int(eta)/3600) << "h " << (int(eta)%3600)/60 << "m\r" << std::flush;
            last = now;
        }
        if (interrupted) { delete eng; std::cout << "\nInterrupted by user\n"; return 0; }
    }
    std::cout << "\nAccumulating pairwise differences on GPU\n";

    uint64_t totalPairs = (nmax > 1) ? (nmax * (nmax - 1)) / 2 : 0, pairsDone = 0;
    for (uint64_t i = 0; i < nmax; ++i) {
        for (uint64_t j = i + 1; j < nmax; ++j) {
            eng->copy((engine::Reg)RDIFF, (engine::Reg)(VAL0 + j));
            eng->sub_reg((engine::Reg)RDIFF, (engine::Reg)(VAL0 + i));
            eng->set_multiplicand((engine::Reg)RPOW, (engine::Reg)RDIFF);
            eng->mul((engine::Reg)RACC, (engine::Reg)RPOW);
            ++pairsDone;
            /* Verification correct pour sub reg
            mpz_class Mpx = (mpz_class(1) << pexp) - 1; 
            mpz_t za, zb, zd_gpu, zd_exp;
            mpz_inits(za, zb, zd_gpu, zd_exp, nullptr);
            eng->get_mpz(za, (engine::Reg)(VAL0 + j));
            eng->get_mpz(zb, (engine::Reg)(VAL0 + i));
            eng->get_mpz(zd_gpu, (engine::Reg)RDIFF);
            mpz_sub(zd_exp, za, zb);
            mpz_mod(zd_exp, zd_exp, Mpx.get_mpz_t());
            if (mpz_cmp(zd_gpu, zd_exp) != 0) {
                std::cerr << "sub_reg mismatch at i=" << i << " j=" << j << std::endl;
                std::cerr << "gpu=" << mpz_class(zd_gpu).get_str() << std::endl;
                std::cerr << "exp=" << mpz_class(zd_exp).get_str() << std::endl;
                std::abort();
            }
            
            mpz_clears(za, zb, zd_gpu, zd_exp, nullptr);*/
            auto now = high_resolution_clock::now();
            if (duration_cast<milliseconds>(now - last).count() >= 400) {
                double done = double(pairsDone), total = double(totalPairs);
                double elapsed = duration<double>(now - t0).count();
                double ips = done / std::max(1e-9, elapsed);
                double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                std::cout << "pairs " << pairsDone << "/" << totalPairs << " | " << std::fixed << std::setprecision(2) << (total ? (done*100.0/total) : 100.0) << "% | ETA " << (int(eta)/3600) << "h " << (int(eta)%3600)/60 << "m\r" << std::flush;
                last = now;
            }
            if (interrupted) { delete eng; std::cout << "\nInterrupted by user\n"; return 0; }
        }
    }
    std::cout << "\nComputing GCD...\n";

    mpz_t Xz; mpz_init(Xz); eng->get_mpz(Xz, (engine::Reg)RACC); mpz_class Mp = (mpz_class(1) << pexp) - 1; mpz_class X; mpz_set(X.get_mpz_t(), Xz); mpz_clear(Xz);
    mpz_class g; mpz_gcd(g.get_mpz_t(), X.get_mpz_t(), Mp.get_mpz_t());

    bool found = (g > 1 && g < Mp);
    if (found) { std::cout << "Stage 2 n^K Factor found : " << g.get_str() << std::endl; if (guiServer_) { std::ostringstream oss; oss << "Stage 2 n^K Factor found : " << g.get_str(); guiServer_->appendLog(oss.str()); } }
    else { std::cout << "No factor" << std::endl; if (guiServer_) { std::ostringstream oss; oss << "No factor"; guiServer_->appendLog(oss.str()); } }

    double elapsed = duration<double>(high_resolution_clock::now() - t0).count();
    std::cout << "Elapsed (n^K) = " << std::fixed << std::setprecision(2) << elapsed << " s\n";
    delete eng;
    return found ? 0 : 1;
}

static inline unsigned u64_bits(uint64_t x){
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return x ? 64u - static_cast<unsigned>(std::countl_zero(x)) : 0u;
#elif defined(_MSC_VER)
    if (!x) return 0u;
    unsigned long idx;
#if defined(_M_X64) || defined(_M_ARM64)
    _BitScanReverse64(&idx, x);
    return idx + 1u;
#else
    if (x >> 32) { _BitScanReverse(&idx, static_cast<unsigned long>(x >> 32)); return idx + 33u; }
    _BitScanReverse(&idx, static_cast<unsigned long>(x));
    return idx + 1u;
#endif
#else
    return x ? 64u - static_cast<unsigned>(__builtin_clzll(x)) : 0u;
#endif
}

static bool load_pm1_s1_from_save(const std::string& path,
                                  uint64_t& B1_out,
                                  uint32_t& p_out,
                                  mpz_class& X_out)
{
    std::string txt;
    if (!read_text_file(path, txt)) return false;

    std::string hexX;
    if (!parse_ecm_resume_line(txt, B1_out, p_out, hexX)) return false;

    if (hexX.empty()) return false;

    mpz_set_str(X_out.get_mpz_t(), hexX.c_str(), 16);
    return true;
}

static bool load_pm1_s1_from_p95(const std::string& path,
                                 uint64_t& B1_out,
                                 uint32_t& p_out,
                                 mpz_class& X_out)
{
    std::vector<uint8_t> data;

    if (!core::algo::read_prime95_s1_to_bytes(path, p_out, B1_out, data)) {
        return false;
    }
    if (data.empty()) return false;

    mpz_import(X_out.get_mpz_t(),
               data.size(),   
               -1,            
               1,             
               0,             
               0,             
               data.data());

    return true;
}



static mpz_class buildE_full(uint64_t B1)
{
    if (B1 < 2)
        return mpz_class(1);

    mpz_class E(1);

    if (B1 >= 2) {
        uint64_t pw2 = 2;
        while (pw2 <= B1 / 2)
            pw2 <<= 1;
        mpz_mul_ui(E.get_mpz_t(), E.get_mpz_t(), pw2);
    }

    if (B1 < 3)
        return E;
    uint64_t R = (uint64_t)std::sqrt((long double)B1);
    if (R < 3) R = 3;

    std::vector<uint8_t> base((R >> 1) + 1, 1);
    for (uint64_t i = 3; i * i <= R; i += 2) {
        if (base[i >> 1]) {
            for (uint64_t j = i * i; j <= R; j += (i << 1))
                base[j >> 1] = 0;
        }
    }

    std::vector<uint64_t> small_primes;
    for (uint64_t i = 3; i <= R; i += 2)
        if (base[i >> 1])
            small_primes.push_back(i);

    const uint64_t span = 1ULL << 24;
    uint64_t low = 3;
    if ((low & 1ULL) == 0) low += 1;

    while (low <= B1) {
        uint64_t high;
        if (B1 - low + 1 <= span)
            high = B1;
        else
            high = low + span - 1;

        if ((high & 1ULL) == 0) high -= 1;
        if (high < low) break;

        size_t len = (size_t)(((high - low) >> 1) + 1);
        std::vector<uint8_t> seg(len, 1);

        for (uint64_t q : small_primes) {
            uint64_t q2 = q * q;
            if (q2 > high) break;

            uint64_t start = (q2 > low)
                           ? q2
                           : ((low + q - 1) / q) * q;
            if ((start & 1ULL) == 0) start += q;
            if (start < low) start += q;

            for (uint64_t j = start; j <= high; j += (q << 1)) {
                size_t idx = (size_t)((j - low) >> 1);
                seg[idx] = 0;
            }
        }

        for (uint64_t n = low; n <= high; n += 2) {
            size_t idx = (size_t)((n - low) >> 1);
            if (!seg[idx]) continue;

            uint64_t p  = n;
            uint64_t pw = p;
            while (pw <= B1 / p)
                pw *= p;

            mpz_mul_ui(E.get_mpz_t(), E.get_mpz_t(), pw);
        }

        if (high >= B1)
            break;
        low = high + 2;
    }

    return E;
}

/*
static mpz_class buildE_incremental(uint64_t B1_old, uint64_t B1_new)
{
    if (B1_new <= B1_old)
        return mpz_class(1);

    mpz_class E_old = buildE_full(B1_old);
    mpz_class E_new = buildE_full(B1_new);

    mpz_class E_diff;
    mpz_divexact(E_diff.get_mpz_t(), E_new.get_mpz_t(), E_old.get_mpz_t());
    return E_diff;
}*/


static inline void mpz_mul_u64(mpz_class& a, uint64_t x) {
    if (x <= (uint64_t)std::numeric_limits<unsigned long>::max()) {
        mpz_mul_ui(a.get_mpz_t(), a.get_mpz_t(), (unsigned long)x);
    } else {
        mpz_class t;
        t = mpz_from_u64(x);
        mpz_mul(a.get_mpz_t(), a.get_mpz_t(), t.get_mpz_t());
    }
}

static std::vector<uint32_t> primes_up_to(uint32_t n) {
    std::vector<uint8_t> is(n + 1, 1);
    is[0] = 0;
    if (n >= 1u) is[1] = 0;
    for (uint32_t i = 2; (uint64_t)i * i <= n; ++i) {
        if (!is[i]) continue;
        for (uint64_t j = (uint64_t)i * i; j <= n; j += i) is[(size_t)j] = 0;
    }
    std::vector<uint32_t> pr;
    pr.reserve(n / 10);
    for (uint32_t i = 2; i <= n; ++i) if (is[i]) pr.push_back(i);
    return pr;
}


static inline double now_seconds_since(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now() - t0).count();
}

static constexpr uint64_t SIEVE_BLOCK = 1ULL << 20;

static mpz_class buildE_incremental_fast(uint64_t B1_old, uint64_t B1_new)
{
    if (B1_new <= B1_old) return mpz_class(1);
    if (B1_old < 2) B1_old = 1;

    mpz_class E_diff(1);

    const uint64_t high = B1_new;
    const uint32_t root = (uint32_t)std::sqrt((long double)high);
    auto primes = primes_up_to(root);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto t_last = t0;

    auto should_print = [&]() {
        auto t = clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(t - t_last).count() >= 400) {
            t_last = t;
            return true;
        }
        return false;
    };

    auto print_line = [&](const char* phase, double pct, uint64_t cur, uint64_t total) {
        double elapsed = now_seconds_since(t0);
        double eta = (pct > 0.0001) ? elapsed * (1.0 / (pct / 100.0) - 1.0) : 0.0;
        std::fprintf(stderr,
            "\r[%s] %6.2f%%  (%" PRIu64 "/%" PRIu64 ")  elapsed=%.1fs  eta=%.1fs      ",
            phase, pct, cur, total, elapsed, eta);
        std::fflush(stderr);
    };

    const uint64_t total_primes = (uint64_t)primes.size();
    for (uint64_t idx = 0; idx < total_primes; ++idx) {
        uint32_t p = primes[(size_t)idx];
        uint64_t pp = (uint64_t)p;

        uint64_t pw = pp;
        while (pw <= B1_old / pp) pw *= pp;

        while (pw <= B1_new / pp) {
            pw *= pp;
            mpz_mul_u64(E_diff, pp);
        }

        if (should_print()) {
            double pct = total_primes ? (100.0 * (double)(idx + 1) / (double)total_primes) : 100.0;
            print_line("prime-powers", pct, idx + 1, total_primes);
        }
    }
    const uint64_t total_range = (B1_new - B1_old);
    uint64_t done_range = 0;

    uint64_t start = B1_old + 1;
    while (start <= B1_new) {
        uint64_t end = std::min(B1_new, start + SIEVE_BLOCK - 1);
        size_t len = (size_t)(end - start + 1);

        std::vector<uint8_t> seg(len, 1);

        for (uint32_t q : primes) {
            uint64_t qq = (uint64_t)q * (uint64_t)q;
            if (qq > end) break;

            uint64_t first = (start + q - 1) / q;
            uint64_t j = first * (uint64_t)q;
            if (j < qq) j = qq;

            for (; j <= end; j += q) {
                seg[(size_t)(j - start)] = 0;
            }
        }

        for (size_t i = 0; i < len; ++i) {
            uint64_t n = start + (uint64_t)i;
            if (n < 2) continue;
            if (!seg[i]) continue;
            mpz_mul_u64(E_diff, n);
        }

        done_range += (end - start + 1);

        if (should_print()) {
            double pct = total_range ? (100.0 * (double)done_range / (double)total_range) : 100.0;
            print_line("sieve-range", pct, done_range, total_range);
        }

        start = end + 1;
    }

    std::fprintf(stderr, "\r[done]                                   elapsed=%.1fs                    \n",
                 now_seconds_since(t0));

    return E_diff;
}



int App::runPM1Marin() {
   // v56: fail fast on invalid default V-trace options before spending time in Stage 1.
   // Normal-memory P-1 Stage 2 uses V-trace by default. Stage 2 V-trace
   // requires even D so odd primes q can be represented as q = kD ± j with
   // odd baby j and gcd(j,D)=1. Use -pm1-vtrace-off to force the previous
   // classic BSGS Stage 2 path.
   const bool pm1Stage2WillUseVTrace = (options.B2 > 0 && !options.pm1_vtrace_off && !options.pm1_lowmem);
   if (!options.pm1_vtrace_off && (pm1Stage2WillUseVTrace || (options.pm1_vtrace && options.B2 > 0))) {
        uint64_t vd = options.pm1_vtrace_D ? options.pm1_vtrace_D : 30030ULL;
        if (!options.pm1_vtrace_auto_d && (vd < 4 || (vd & 1ULL))) {
            std::cerr << "[PM1-VTRACE] D must be even and >= 4. "
                      << "Odd D=" << vd << " is rejected before Stage 1. "
                      << "Use e.g. -pm1-vtrace-d 4620, 30030 or -pm1-vtrace-deep-d auto, "
                      << "or -pm1-vtrace-off for classic Stage 2.\n";
            return -1;
        }
        if (options.pm1_lowmem) {
            std::cerr << "[PM1-VTRACE] V-trace is a normal-memory Stage 2 path; "
                      << "do not combine it with -pm1-lowmem/-pm1-ultralowmem.\n";
            return -1;
        }
   }

   if (guiServer_) {
        std::ostringstream oss;
        oss << "P-1 factoring stage 1";
        guiServer_->setStatus(oss.str());
    }

    uint64_t B1_new = options.B1;
    uint64_t B1_old = options.B1old;
    bool doExtend = (B1_old > 0 && B1_new > B1_old);

    // In ultra-low-memory mode a normal resume extension uses the generic path
    // and historically allocated many logical registers.  For large transforms
    // such as MM31 we instead use a dedicated low-memory strategy:
    //   1) Run the real delta extension H_new = H_old^Delta using the compact canonical 3-register path.
    //      This is the preferred path on 16 GB GPUs such as P100 because it pays
    //      only for the B1 delta.
    //   2) No automatic recompute fallback in this diagnostic path: P100 must run
    //      the true delta extension, otherwise -b1old has no value.
    bool ultralowmem_delta_extend = doExtend && options.pm1_ultralowmem && options.pm1_lowmem;
    bool ultralowmem_delta2_extend = ultralowmem_delta_extend && (std::getenv("PRMERS_PM1_MM31_DELTA2") != nullptr);
    bool ultralowmem_fast3_recompute_extend = false;
    if (ultralowmem_delta_extend) {
        options.gerbiczli = false;
        std::cout << "[PM1] Ultra-low-memory B1 extension requested: B1old="
                  << B1_old << " -> B1=" << B1_new << "\n";
        std::cout << "[PM1] Ultra-low-memory extension will run the true "
                  << "delta path H_old^Delta with " << (ultralowmem_delta2_extend ? "2 GPU registers (persistent multiplicand, experimental)" : "3 GPU registers")
                  << "; v93 uses compact GPU weights, no fast3 recompute fallback by default.\n";
        if (guiServer_) {
            std::ostringstream oss;
            oss << "[PM1] Ultra-low-memory B1 extension requested: B1old="
                << B1_old << " -> B1=" << B1_new
                << "\n[PM1] Running true delta extension H_old^Delta with "
                << (ultralowmem_delta2_extend ? "2 GPU registers (persistent multiplicand, experimental)." : "3 GPU registers.")
                << " v93 compact GPU weights are enabled for MM31/RTX3080-class memory pressure.\n";
            guiServer_->appendLog(oss.str());
        }
    }

    const char* const arithmetic_backend = engine::configured_gpu_backend_name();
    std::cout << "[Backend " << arithmetic_backend << "] Start a P-1 factoring stage 1 up to B1="
              << B1_new << (ultralowmem_delta_extend ? (ultralowmem_delta2_extend ? " (ULTRALOWMEM DELTA 2-REG SPLIT-AUX)" : " (ULTRALOWMEM DELTA 3-REG)") : (doExtend ? " (EXTEND mode)" : (ultralowmem_fast3_recompute_extend ? " (ULTRALOWMEM FAST3 RECOMPUTE)" : ""))) << std::endl;

    if (guiServer_) {
        std::ostringstream oss;
        oss << "[Backend " << arithmetic_backend << "] Start a P-1 factoring stage 1 up to B1="
            << B1_new << (ultralowmem_delta_extend ? (ultralowmem_delta2_extend ? " (ULTRALOWMEM DELTA 2-REG SPLIT-AUX)" : " (ULTRALOWMEM DELTA 3-REG)") : (doExtend ? " (EXTEND mode)" : (ultralowmem_fast3_recompute_extend ? " (ULTRALOWMEM FAST3 RECOMPUTE)" : "")));
        guiServer_->appendLog(oss.str());
    }

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;

    const bool want_p95_stage2 = options.p95stage2 && !options.p95path.empty();
    bool p95_stage2_enabled = false;
    fs::path p95_dir;
    fs::path p95_exe;
    auto p95_log = [&](const std::string& s) {
        std::cout << s << std::endl;
        if (guiServer_) guiServer_->appendLog(s);
    };
    if (want_p95_stage2) {
        p95_dir = p95_expand_user_path(options.p95path);
        std::error_code ec;
        if (!fs::exists(p95_dir, ec) || !fs::is_directory(p95_dir, ec)) {
            p95_log(std::string("[PM1] Prime95 Stage2 disabled: invalid directory '") + options.p95path + "' -> resolved to '" + p95_dir.string() + "'");
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
                p95_log(std::string("[PM1] Prime95 Stage2 disabled: no Prime95/mprime executable found in '") + options.p95path + "'");
            } else {
                std::string moved_worktodo;
                std::string moved_results;
                bool ok_worktodo = p95_backup_existing_file(p95_dir / "worktodo.txt", &moved_worktodo);
                bool ok_results = p95_backup_existing_file(p95_dir / "results.json.txt", &moved_results);
                if (!ok_worktodo || !ok_results) {
                    p95_log(std::string("[PM1] Prime95 Stage2 disabled: failed to backup worktodo/results in '") + options.p95path + "'");
                } else {
                    if (!moved_worktodo.empty()) p95_log(std::string("[PM1] Prime95 worktodo backed up to ") + moved_worktodo);
                    if (!moved_results.empty()) p95_log(std::string("[PM1] Prime95 results.json.txt backed up to ") + moved_results);
                    p95_stage2_enabled = true;
                    p95_log(std::string("[PM1] Prime95 Stage2 enabled using ") + p95_exe.string());
                }
            }
        }
    }

    auto run_pm1_stage2_external = [&](const std::string& resume_save_path,
                                       const std::string& ds,
                                       const std::string& de,
                                       bool& factor_found_out)->bool {
        factor_found_out = false;
        if (!p95_stage2_enabled || resume_save_path.empty()) return false;

        std::ostringstream p95_name;
        p95_name << 'm' << std::setw(7) << std::setfill('0') << p;
        const fs::path p95_state = p95_dir / p95_name.str();
        int conv_rc = convertEcmResumeToPrime95(resume_save_path, p95_state.string(), ds, de);
        if (conv_rc != 0) {
            p95_log(std::string("[PM1] Prime95 Stage2 failed: could not write state file ") + p95_state.string());
            return false;
        }

        const std::string known_csv = p95_join_known_factors_csv(options.knownFactors);
        std::ostringstream wt;
        wt << "Pminus1=1,2," << p << ",-1," << options.B1 << "," << options.B2;
        if (options.sieveDepth > 0.0) {
            wt << "," << std::setprecision(12) << options.sieveDepth;
        }
        if (options.B2Start > 0) {
            if (options.sieveDepth <= 0.0) wt << ",0";
            wt << "," << options.B2Start;
        }
        if (!known_csv.empty()) {
            wt << ",\"" << known_csv << "\"";
        }

        const std::string log_filename = std::string("prmers_p95stage2_pm1_p") + std::to_string(p) + ".log";
        p95_log(std::string("[PM1] Prime95 Stage2 start | state=") + p95_state.string() + " | log=" + (p95_dir / log_filename).string());
        PM1Prime95Stage2Result rr = p95_run_pm1_stage2_task(p95_dir, p95_exe, p95_state, wt.str(), log_filename, options.knownFactors);
        if (!rr.error.empty()) {
            p95_log(std::string("[PM1] Prime95 Stage2 error: ") + rr.error);
            return false;
        }

        std::ostringstream done;
        done << "[PM1] Prime95 Stage2 done | exit_code=" << rr.exit_code;
        if (!rr.json_line.empty()) done << " | result=" << rr.json_line;
        p95_log(done.str());

        const std::string result_file = std::string("stage2_result_B2_") + std::to_string(options.B2) + "_p_" + std::to_string(options.exponent) + ".txt";
        if (rr.factor_found && !rr.factor.empty() && !rr.known_factor) {
            writeStageResult(result_file, std::string("B2=") + std::to_string(options.B2) + "  factor=" + rr.factor);
            std::cout << "\n>>>  Factor P-1 (stage 2) found : " << rr.factor << '\n';
            if (guiServer_) {
                std::ostringstream oss;
                oss << "\n>>>  Factor P-1 (stage 2) found : " << rr.factor << '\n';
                guiServer_->appendLog(oss.str());
            }
            options.knownFactors.push_back(rr.factor);
            factor_found_out = true;
        } else {
            writeStageResult(result_file, std::string("No factor P-1 up to B2=") + std::to_string(options.B2));
            std::cout << "\nNo factor P-1 (stage 2) until B2 = " << options.B2 << '\n';
            if (guiServer_) {
                std::ostringstream oss;
                oss << "\nNo factor P-1 (stage 2) until B2 = " << options.B2 << '\n';
                guiServer_->appendLog(oss.str());
            }
            if (rr.factor_found && rr.known_factor) {
                p95_log(std::string("[PM1] Prime95 Stage2 found known factor ") + rr.factor + ", continuing");
            }
        }

        std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
        std::cout << "Manual submission JSON:\n" << json << "\n";
        io::WorktodoManager wm(options);
        wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage2_ext", json);
        wm.appendToResultsTxt(json);

        return true;
    };

    mpz_class X_old;
    if (doExtend) {
        std::string basePath = options.pm1_extend_save_path;
        if (basePath.empty()) {
            basePath = "resume_p" + std::to_string(options.exponent) +
                    "_B1_" + std::to_string(B1_old);
        }

        std::string resumeSave = basePath;
        std::string resumeP95  = basePath;

        if (resumeSave.size() >= 5 &&
            resumeSave.substr(resumeSave.size() - 5) == ".save")
        {
            resumeP95 = resumeSave.substr(0, resumeSave.size() - 5) + ".p95";
        }
        else if (resumeSave.size() >= 4 &&
                resumeSave.substr(resumeSave.size() - 4) == ".p95")
        {
            resumeP95  = resumeSave;
            resumeSave = resumeSave.substr(0, resumeSave.size() - 4) + ".save";
        }
        else {
            resumeSave += ".save";
            resumeP95  += ".p95";
        }

        uint64_t B1_file = 0;
        uint32_t p_file  = 0;
        std::string usedPath;

        if (load_pm1_s1_from_save(resumeSave, B1_file, p_file, X_old)) {
            usedPath = resumeSave;
        }
        else if (load_pm1_s1_from_p95(resumeP95, B1_file, p_file, X_old)) {
            usedPath = resumeP95;
        }
        else {
            std::cerr << "Cannot load PM1 S1 state from \"" << resumeSave
                    << "\" nor from \"" << resumeP95 << "\"\n";
            return -1;
        }

        if (B1_file != B1_old || p_file != p) {
            std::cerr << "Mismatch between resume file (B1=" << B1_file
                    << ", p=" << p_file << ") and options (B1old=" << B1_old
                    << ", p=" << p << ")\n";
            return -1;
        }

        std::cout << "Extending PM1 from B1=" << B1_old << " to B1="
                << B1_new << " using state from " << usedPath << "\n";
        if (guiServer_) {
            std::ostringstream oss;
            oss << "Extending PM1 from B1=" << B1_old << " to B1="
                << B1_new << " using state from " << usedPath << "\n";
            guiServer_->appendLog(oss.str());
        }
    }
    //bool debug = false;
    uint64_t B1 = options.B1;
    const double L_est_bits = 1.4426950408889634 * static_cast<double>(B1);
    const uint64_t MAX_E_BITS = options.max_e_bits;
    std::cout << "MAX_E_BITS = " << MAX_E_BITS << " bits (~ " << (MAX_E_BITS >> 23) << " MiB)" << std::endl;
    uint64_t estChunks = std::max<uint64_t>(1, (uint64_t)std::ceil(L_est_bits / (double)MAX_E_BITS));
    //const uint32_t p = static_cast<uint32_t>(options.exponent);
    //const bool verbose = true;//options.debug;
    bool pm1_ultralowmem_stage1 = options.pm1_ultralowmem && options.pm1_lowmem && !doExtend;
    bool pm1_lowmem_stage1 = options.pm1_lowmem && !doExtend;
    if (ultralowmem_delta_extend) {
        options.gerbiczli = false;
        std::cout << "[PM1] Ultra-low-memory delta extension enabled: using " << (ultralowmem_delta2_extend ? "2 GPU registers (persistent multiplicand)" : "3 GPU registers") << "; Gerbicz-Li disabled.\n";
        if (guiServer_) guiServer_->appendLog(std::string("[PM1] Ultra-low-memory delta extension enabled: using ") + (ultralowmem_delta2_extend ? "2 GPU registers (persistent multiplicand)" : "3 GPU registers") + "; Gerbicz-Li disabled.\n");
    } else if (pm1_ultralowmem_stage1) {
        options.gerbiczli = false;
        std::cout << "[PM1] Ultra-low-memory Stage 1 enabled: using 1 GPU register; fast3-only path; Gerbicz-Li disabled.\n";
        if (guiServer_) guiServer_->appendLog("[PM1] Ultra-low-memory Stage 1 enabled: using 1 GPU register; fast3-only path; Gerbicz-Li disabled.\n");
    } else if (pm1_lowmem_stage1) {
        options.gerbiczli = false;
        std::cout << "[PM1] Low-memory Stage 1 enabled: using 3 GPU registers; Gerbicz-Li disabled.\n";
        if (guiServer_) guiServer_->appendLog("[PM1] Low-memory Stage 1 enabled: using 3 GPU registers; Gerbicz-Li disabled.\n");
    }

    auto report_true_delta_compact_plan = [&]() {
        if (!ultralowmem_delta_extend) return;
        const size_t nPlan = ibdwt::transform_size(p);
        const size_t regsPlan = ultralowmem_delta2_extend ? 2u : 3u;
        const long double regBytes = static_cast<long double>(regsPlan) * static_cast<long double>(nPlan) * static_cast<long double>(sizeof(uint64_t));
        const long double carryBytes = (static_cast<long double>(nPlan) / 4.0L) * static_cast<long double>(sizeof(uint64_t));
        const long double rootBytes = 3.0L * static_cast<long double>(nPlan) * static_cast<long double>(sizeof(uint64_t));
        const long double widthBytes = static_cast<long double>(nPlan) * static_cast<long double>(sizeof(uint8_t));
        // Runtime CWM is usually 256 here, so one compact base pair per 1024 digits + 1024 relative pairs.
        // The engine computes the exact allocation from CWM; this log is a conservative MM31 estimate.
        const long double compactWeightBytes = (static_cast<long double>((nPlan + 1023) / 1024) + 1024.0L) * 2.0L * static_cast<long double>(sizeof(uint64_t));
        const long double fullWeightBytes = 2.0L * static_cast<long double>(nPlan) * static_cast<long double>(sizeof(uint64_t));
        const long double totalCompact = regBytes + carryBytes + rootBytes + widthBytes + compactWeightBytes;
        auto gib = [](long double b)->long double { return b / 1073741824.0L; };
        std::cout << "[PM1] v93 true-delta compact-weight estimate: regs=" << regsPlan
                  << " compact-total≈" << std::fixed << std::setprecision(3) << gib(totalCompact)
                  << " GiB before driver overhead; saved≈" << gib(fullWeightBytes - compactWeightBytes)
                  << " GiB versus full GPU weight table.\n";
        std::cout.unsetf(std::ios::floatfield);
    };
    report_true_delta_compact_plan();

    size_t stage1RegCount = ultralowmem_delta_extend ? (ultralowmem_delta2_extend ? 2u : 3u) : (pm1_ultralowmem_stage1 ? 1u : (pm1_lowmem_stage1 ? 3u : 11u));
    engine* eng = nullptr;
    try {
        eng = engine::create_gpu(p, stage1RegCount, static_cast<size_t>(options.device_id), verbose);
    } catch (const std::exception& ex) {
        if (!ultralowmem_delta_extend || std::getenv("PRMERS_PM1_ENABLE_FAST3_FALLBACK") == nullptr) throw;
        std::cerr << "[PM1] True-delta engine failed before start: " << ex.what() << "\n"
                  << "[PM1] PRMERS_PM1_ENABLE_FAST3_FALLBACK=1 is set, so falling back to 1-reg fast3 recompute.\n";
        ultralowmem_delta_extend = false;
        ultralowmem_delta2_extend = false;
        ultralowmem_fast3_recompute_extend = true;
        doExtend = false;
        B1_old = 0;
        pm1_ultralowmem_stage1 = true;
        pm1_lowmem_stage1 = true;
        options.gerbiczli = false;
        stage1RegCount = 1u;
        eng = engine::create_gpu(p, stage1RegCount, static_cast<size_t>(options.device_id), verbose);
    }

    const bool aevum_backend = eng->is_aevum_backend();
    if (aevum_backend && !pm1_ultralowmem_stage1) {
        std::cout << "[PM1] Aevum uses the generic square plus base-3 multiply path; fast3 is disabled.\n";
        if (guiServer_) guiServer_->appendLog("[PM1] Aevum uses the generic square plus base-3 multiply path; fast3 is disabled.\n");
    }
    if (aevum_backend && pm1_ultralowmem_stage1) {
        delete eng;
        throw std::runtime_error("-pm1-ultralowmem requires Marin because the one-register path depends on fast3; use -engine-marin or omit -aevum");
    }

    const size_t RSTATE=0;
    const size_t RACC_L = (pm1_lowmem_stage1 || ultralowmem_delta_extend) ? 0u : 1u;
    const size_t RACC_R = (pm1_lowmem_stage1 || ultralowmem_delta_extend) ? 0u : 2u;
    const size_t RCHK   = (pm1_lowmem_stage1 || ultralowmem_delta_extend) ? 0u : 3u;
    const size_t RPOW   = ultralowmem_delta_extend ? (ultralowmem_delta2_extend ? 1u : 2u) : (pm1_lowmem_stage1 ? 1u : 4u);
    const size_t RTMP   = ultralowmem_delta_extend ? (ultralowmem_delta2_extend ? 1u : 2u) : (pm1_lowmem_stage1 ? 1u : 5u);
    const size_t RSTART = (pm1_lowmem_stage1 || ultralowmem_delta_extend) ? 0u : 6u;
    const size_t RSAVE_S= (pm1_lowmem_stage1 || ultralowmem_delta_extend) ? 0u : 7u;
    const size_t RSAVE_L= (pm1_lowmem_stage1 || ultralowmem_delta_extend) ? 0u : 8u;
    const size_t RSAVE_R= (pm1_lowmem_stage1 || ultralowmem_delta_extend) ? 0u : 9u;
    const size_t RBASE  = ultralowmem_delta_extend ? 1u : (pm1_ultralowmem_stage1 ? 0u : (pm1_lowmem_stage1 ? 2u : 10u));
    std::ostringstream ck; ck << "pm1_m_" << p << ".ckpt";
    const std::string ckpt_file = ck.str();
    auto save_ckpt = [&](uint32_t i, double et, uint64_t chk, uint64_t blks, uint64_t bib, uint64_t cbl, uint8_t inlot, const mpz_class& ceacc, const mpz_class& cwbits, uint64_t chunkIdx, uint64_t startP, uint8_t first, uint64_t processedBits, uint64_t bitsInChunk){
        const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
        { File f(newf, "wb"); int version = 3; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; if (!f.write(reinterpret_cast<const char*>(&chk), sizeof(chk))) return; if (!f.write(reinterpret_cast<const char*>(&blks), sizeof(blks))) return; if (!f.write(reinterpret_cast<const char*>(&bib), sizeof(bib))) return; if (!f.write(reinterpret_cast<const char*>(&cbl), sizeof(cbl))) return; if (!f.write(reinterpret_cast<const char*>(&inlot), sizeof(inlot))) return; char* eacc_hex_c = mpz_get_str(nullptr, 16, ceacc.get_mpz_t()); uint32_t eacc_len = eacc_hex_c ? (uint32_t)std::strlen(eacc_hex_c) : 0; if (!f.write(reinterpret_cast<const char*>(&eacc_len), sizeof(eacc_len))) { if (eacc_hex_c) std::free(eacc_hex_c); return; } if (eacc_len && !f.write(eacc_hex_c, eacc_len)) { std::free(eacc_hex_c); return; } if (eacc_hex_c) std::free(eacc_hex_c); char* wbits_hex_c = mpz_get_str(nullptr, 16, cwbits.get_mpz_t()); uint32_t wbits_len = wbits_hex_c ? (uint32_t)std::strlen(wbits_hex_c) : 0; if (!f.write(reinterpret_cast<const char*>(&wbits_len), sizeof(wbits_len))) { if (wbits_hex_c) std::free(wbits_hex_c); return; } if (wbits_len && !f.write(wbits_hex_c, wbits_len)) { std::free(wbits_hex_c); return; } if (wbits_hex_c) std::free(wbits_hex_c); if (!f.write(reinterpret_cast<const char*>(&chunkIdx), sizeof(chunkIdx))) return; if (!f.write(reinterpret_cast<const char*>(&startP), sizeof(startP))) return; if (!f.write(reinterpret_cast<const char*>(&first), sizeof(first))) return; if (!f.write(reinterpret_cast<const char*>(&processedBits), sizeof(processedBits))) return; if (!f.write(reinterpret_cast<const char*>(&bitsInChunk), sizeof(bitsInChunk))) return; f.write_crc32(); }
        std::error_code ec; fs::remove(oldf, ec); fs::rename(ckpt_file, oldf, ec); fs::rename(ckpt_file + ".new", ckpt_file, ec); fs::remove(oldf, ec);
        write_pm1_checkpoint_backend(ckpt_file, eng);
    };
    auto read_ckpt = [&](const std::string& file, uint32_t& ri, double& et, uint64_t& chk, uint64_t& blks, uint64_t& bib, uint64_t& cbl, uint8_t& inlot, mpz_class& ceacc, mpz_class& cwbits, uint64_t& chunkIdx, uint64_t& startP, uint8_t& first, uint64_t& processedBits, uint64_t& bitsInChunk)->int{
        File f(file);
        if (!f.exists()) return -1;
        std::string backend_reason;
        if (!pm1_checkpoint_backend_matches(file, eng, &backend_reason)) {
            std::cerr << "[PM1] Ignoring incompatible checkpoint " << file << ": " << backend_reason << "\n";
            return -3;
        }
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 3) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != p) return -2;
        if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = eng->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!eng->set_checkpoint(data)) return -2;
        if (!f.read(reinterpret_cast<char*>(&chk), sizeof(chk))) return -2;
        if (!f.read(reinterpret_cast<char*>(&blks), sizeof(blks))) return -2;
        if (!f.read(reinterpret_cast<char*>(&bib), sizeof(bib))) return -2;
        if (!f.read(reinterpret_cast<char*>(&cbl), sizeof(cbl))) return -2;
        if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
        uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
        std::string eacc_hex; eacc_hex.resize(eacc_len);
        if (eacc_len && !f.read(eacc_hex.data(), eacc_len)) return -2;
        if (eacc_len) mpz_set_str(ceacc.get_mpz_t(), eacc_hex.c_str(), 16); else ceacc = 0;
        uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
        std::string wbits_hex; wbits_hex.resize(wbits_len);
        if (wbits_len && !f.read(wbits_hex.data(), wbits_len)) return -2;
        if (wbits_len) mpz_set_str(cwbits.get_mpz_t(), wbits_hex.c_str(), 16); else cwbits = 0;
        if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
        if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
        if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
        if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
        if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };
    timer.start();
    timer2.start();
    auto start_clock = std::chrono::high_resolution_clock::now();
    auto lastDisplay = start_clock;
    auto lastBackup = start_clock;
    interrupted.store(false, std::memory_order_relaxed);

    if (doExtend) {    
        if (ultralowmem_delta_extend) std::cout << "[PM1] Loading H_old into RBASE using lowmem streamed set_mpz..." << std::endl;
        mpz_t Xtmp;
        mpz_init(Xtmp);
        mpz_set(Xtmp, X_old.get_mpz_t()); 

        eng->set_mpz(static_cast<engine::Reg>(RBASE), Xtmp);

        if (ultralowmem_delta2_extend) {
            // v91d: initialize the other live register while the queue only has
            // host uploads behind it.  On RTX 3080/NVIDIA OpenCL, doing a
            // blocking host write immediately after the huge RBASE forward
            // transform can report a delayed CL_MEM_OBJECT_ALLOCATION_FAILURE.
            std::cout << "[PM1] DELTA2: H_old loaded; pre-initializing RSTATE=1 before RBASE transform..." << std::endl;
            eng->set(static_cast<engine::Reg>(RSTATE), 1);
            std::cout << "[PM1] DELTA2: RSTATE initialized; transforming RBASE once as persistent multiplicand..." << std::endl;
            eng->set_multiplicand(static_cast<engine::Reg>(RBASE), static_cast<engine::Reg>(RBASE));
            std::cout << "[PM1] DELTA2: waiting for persistent multiplicand transform to complete..." << std::endl;
            eng->sync();
            std::cout << "[PM1] DELTA2: RBASE persistent multiplicand ready; building E_diff next." << std::endl;
        }

        mpz_clear(Xtmp);
        if (!ultralowmem_delta2_extend) {
            if (ultralowmem_delta_extend) std::cout << "[PM1] H_old loaded; initializing RSTATE=1..." << std::endl;

            eng->set(static_cast<engine::Reg>(RSTATE), 1);
            if (ultralowmem_delta_extend) std::cout << "[PM1] RSTATE initialized; building E_diff next." << std::endl;
        }
        if (options.gerbiczli) {
            eng->set(static_cast<engine::Reg>(RACC_L), 1);
            eng->set(static_cast<engine::Reg>(RACC_R), 1);
            eng->set(static_cast<engine::Reg>(RSTART), 1);
            eng->copy(static_cast<engine::Reg>(RSAVE_S), static_cast<engine::Reg>(RSTATE));
            eng->copy(static_cast<engine::Reg>(RSAVE_L), static_cast<engine::Reg>(RACC_L));
            eng->copy(static_cast<engine::Reg>(RSAVE_R), static_cast<engine::Reg>(RACC_R));
        }
    }
    else{
        eng->set(RSTATE, 1);
        if (options.gerbiczli) {
            eng->set(RACC_L, 1);
            eng->set(RACC_R, 1);
            eng->copy(RSTART, RSTATE);
            eng->copy(RSAVE_S, RSTATE);
            eng->copy(RSAVE_L, RACC_L);
            eng->copy(RSAVE_R, RACC_R);
        }
    }
    uint64_t chunkIndex = 0;
    uint64_t startPrime = 3;
    bool firstChunk = true;
    uint64_t processed_total_bits = 0;
    uint32_t resumeI_ck = 0;
    double restored_time = 0.0;
    uint64_t gl_checkpass_ck = 0, gl_blocks_since_check_ck = 0, gl_bits_in_block_ck = 0, gl_current_block_len_ck = 0, bits_in_chunk_ck = 0;
    uint8_t in_lot_ck = 0, firstChunk_ck = 1;
    mpz_class eacc_ck = 0, wbits_ck = 0;
    bool restored = false;
    int rr = read_ckpt(ckpt_file, resumeI_ck, restored_time, gl_checkpass_ck, gl_blocks_since_check_ck, gl_bits_in_block_ck, gl_current_block_len_ck, in_lot_ck, eacc_ck, wbits_ck, chunkIndex, startPrime, firstChunk_ck, processed_total_bits, bits_in_chunk_ck);
    if (rr < 0) rr = read_ckpt(ckpt_file + ".old", resumeI_ck, restored_time, gl_checkpass_ck, gl_blocks_since_check_ck, gl_bits_in_block_ck, gl_current_block_len_ck, in_lot_ck, eacc_ck, wbits_ck, chunkIndex, startPrime, firstChunk_ck, processed_total_bits, bits_in_chunk_ck);
    if (rr == 0) { restored = true; firstChunk = (firstChunk_ck != 0); }
    auto start_sys = std::chrono::system_clock::now();
    if (doExtend) {
        std::cout << "Building E_diff for (B1old=" << B1_old << ", B1new=" << B1_new << ")...\n" << std::flush;
        mpz_class E_diff = buildE_incremental_fast(B1_old, B1_new);
        std::cout << "E_diff built (" << mpz_sizeinbase(E_diff.get_mpz_t(), 2) << " bits)\n" << std::flush;

        mp_bitcnt_t bits = mpz_sizeinbase(E_diff.get_mpz_t(), 2);

        std::cout << "Extending PM1 exponent: E_diff has " << bits << " bits\n";
        if (guiServer_) {
            std::ostringstream oss;
            oss << "Extending PM1 exponent: E_diff has " << bits << " bits\n";
            guiServer_->appendLog(oss.str());
        }

        if (bits == 0) {
            std::cout << "Nothing to extend (E_diff = 1)\n";
        } else {
            // Réinitialise les timers locaux pour l'extension
            start_clock = std::chrono::high_resolution_clock::now();
            lastDisplay  = start_clock;
            lastBackup   = start_clock;

            // Checkpoint spécifique à la phase d'extension
            std::ostringstream ckext;
            ckext << "pm1_m_" << p << "_ext.ckpt";
            const std::string ckpt_file_ext = ckext.str();

            auto save_ckpt_ext = [&](uint32_t i, double et, uint64_t chk, uint64_t blks, uint64_t bib, uint64_t cbl, uint8_t inlot, const mpz_class& ceacc, const mpz_class& cwbits, uint64_t chunkIdx, uint64_t startP, uint8_t first, uint64_t processedBits, uint64_t bitsInChunk){
                const std::string oldf = ckpt_file_ext + ".old", newf = ckpt_file_ext + ".new";
                { File f(newf, "wb"); int version = 3; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; if (!f.write(reinterpret_cast<const char*>(&chk), sizeof(chk))) return; if (!f.write(reinterpret_cast<const char*>(&blks), sizeof(blks))) return; if (!f.write(reinterpret_cast<const char*>(&bib), sizeof(bib))) return; if (!f.write(reinterpret_cast<const char*>(&cbl), sizeof(cbl))) return; if (!f.write(reinterpret_cast<const char*>(&inlot), sizeof(inlot))) return; char* eacc_hex_c = mpz_get_str(nullptr, 16, ceacc.get_mpz_t()); uint32_t eacc_len = eacc_hex_c ? (uint32_t)std::strlen(eacc_hex_c) : 0; if (!f.write(reinterpret_cast<const char*>(&eacc_len), sizeof(eacc_len))) { if (eacc_hex_c) std::free(eacc_hex_c); return; } if (eacc_len && !f.write(eacc_hex_c, eacc_len)) { std::free(eacc_hex_c); return; } if (eacc_hex_c) std::free(eacc_hex_c); char* wbits_hex_c = mpz_get_str(nullptr, 16, cwbits.get_mpz_t()); uint32_t wbits_len = wbits_hex_c ? (uint32_t)std::strlen(wbits_hex_c) : 0; if (!f.write(reinterpret_cast<const char*>(&wbits_len), sizeof(wbits_len))) { if (wbits_hex_c) std::free(wbits_hex_c); return; } if (wbits_len && !f.write(wbits_hex_c, wbits_len)) { std::free(wbits_hex_c); return; } if (wbits_hex_c) std::free(wbits_hex_c); if (!f.write(reinterpret_cast<const char*>(&chunkIdx), sizeof(chunkIdx))) return; if (!f.write(reinterpret_cast<const char*>(&startP), sizeof(startP))) return; if (!f.write(reinterpret_cast<const char*>(&first), sizeof(first))) return; if (!f.write(reinterpret_cast<const char*>(&processedBits), sizeof(processedBits))) return; if (!f.write(reinterpret_cast<const char*>(&bitsInChunk), sizeof(bitsInChunk))) return; f.write_crc32(); }
                std::error_code ec; fs::remove(oldf, ec); fs::rename(ckpt_file_ext, oldf, ec); fs::rename(ckpt_file_ext + ".new", ckpt_file_ext, ec); fs::remove(oldf, ec);
                write_pm1_checkpoint_backend(ckpt_file_ext, eng);
            };

            auto read_ckpt_ext = [&](const std::string& file, uint32_t& ri, double& et, uint64_t& chk, uint64_t& blks, uint64_t& bib, uint64_t& cbl, uint8_t& inlot, mpz_class& ceacc, mpz_class& cwbits, uint64_t& chunkIdx, uint64_t& startP, uint8_t& first, uint64_t& processedBits, uint64_t& bitsInChunk)->int{
                File f(file);
                if (!f.exists()) return -1;
                std::string backend_reason;
                if (!pm1_checkpoint_backend_matches(file, eng, &backend_reason)) {
                    std::cerr << "[PM1] Ignoring incompatible extension checkpoint " << file << ": " << backend_reason << "\n";
                    return -3;
                }
                int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
                if (version != 3) return -2;
                uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
                if (rp != p) return -2;
                if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
                const size_t cksz = eng->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!f.read(data.data(), cksz)) return -2;
                if (!eng->set_checkpoint(data)) return -2;
                if (!f.read(reinterpret_cast<char*>(&chk), sizeof(chk))) return -2;
                if (!f.read(reinterpret_cast<char*>(&blks), sizeof(blks))) return -2;
                if (!f.read(reinterpret_cast<char*>(&bib), sizeof(bib))) return -2;
                if (!f.read(reinterpret_cast<char*>(&cbl), sizeof(cbl))) return -2;
                if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
                uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
                std::string eacc_hex; eacc_hex.resize(eacc_len);
                if (eacc_len && !f.read(eacc_hex.data(), eacc_len)) return -2;
                if (eacc_len) mpz_set_str(ceacc.get_mpz_t(), eacc_hex.c_str(), 16); else ceacc = 0;
                uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
                std::string wbits_hex; wbits_hex.resize(wbits_len);
                if (wbits_len && !f.read(wbits_hex.data(), wbits_len)) return -2;
                if (wbits_len) mpz_set_str(cwbits.get_mpz_t(), wbits_hex.c_str(), 16); else cwbits = 0;
                if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
                if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
                if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
                if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
                if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
                if (!f.check_crc32()) return -2;
                return 0;
            };

            uint32_t resumeI_ext_ck = 0;
            double restored_time_ext = 0.0;
            uint64_t gl_checkpass_ext_ck = 0, gl_blocks_since_check_ext_ck = 0, gl_bits_in_block_ext_ck = 0, gl_current_block_len_ext_ck = 0, bits_in_chunk_ext_ck = 0;
            uint8_t in_lot_ext_ck = 0, firstChunk_ext_ck = 0;
            mpz_class eacc_ext_ck = 0, wbits_ext_ck = 0;
            uint64_t chunkIdx_ext = 0;
            uint64_t startP_ext = 0;
            uint64_t processedBits_ext_ck = 0;
            bool restored_ext = false;

            int rr_ext = read_ckpt_ext(ckpt_file_ext, resumeI_ext_ck, restored_time_ext, gl_checkpass_ext_ck, gl_blocks_since_check_ext_ck, gl_bits_in_block_ext_ck, gl_current_block_len_ext_ck, in_lot_ext_ck, eacc_ext_ck, wbits_ext_ck, chunkIdx_ext, startP_ext, firstChunk_ext_ck, processedBits_ext_ck, bits_in_chunk_ext_ck);
            if (rr_ext < 0) rr_ext = read_ckpt_ext(ckpt_file_ext + ".old", resumeI_ext_ck, restored_time_ext, gl_checkpass_ext_ck, gl_blocks_since_check_ext_ck, gl_bits_in_block_ext_ck, gl_current_block_len_ext_ck, in_lot_ext_ck, eacc_ext_ck, wbits_ext_ck, chunkIdx_ext, startP_ext, firstChunk_ext_ck, processedBits_ext_ck, bits_in_chunk_ext_ck);
            if (rr_ext == 0 && bits_in_chunk_ext_ck == (uint64_t)bits) {
                restored_ext = true;
            }
            restored_time = restored_ext ? restored_time_ext : 0.0;

            uint64_t B = std::max<uint64_t>(1, (uint64_t)std::sqrt((double)bits));
            double desiredIntervalSeconds = 600.0;
            uint64_t checkpasslevel_auto =
                (uint64_t)((1000.0 * desiredIntervalSeconds) / (double)B);
            if (checkpasslevel_auto == 0) {
                uint64_t tmpB = (uint64_t)std::sqrt((double)B);
                if (tmpB == 0) tmpB = 1;
                checkpasslevel_auto = ((uint64_t)bits / B) / tmpB;
            }
            uint64_t checkpasslevel = (options.checklevel > 0)
                ? options.checklevel
                : checkpasslevel_auto;
            if (checkpasslevel == 0) checkpasslevel = 1;

            uint64_t blocks_since_check = restored_ext ? gl_blocks_since_check_ext_ck : 0;
            uint64_t bits_in_block      = restored_ext ? gl_bits_in_block_ext_ck     : 0;
            uint64_t current_block_len  = restored_ext && gl_current_block_len_ext_ck
                                          ? gl_current_block_len_ext_ck
                                          : (((uint64_t)(( (restored_ext ? resumeI_ext_ck : (uint32_t)bits ) - 1) % B)) + 1);
            mpz_class eacc              = restored_ext ? eacc_ext_ck : 0;
            mpz_class wbits             = restored_ext ? wbits_ext_ck : 0;
            uint64_t gl_checkpass       = restored_ext ? gl_checkpass_ext_ck : 0;
            bool in_lot                 = restored_ext ? (in_lot_ext_ck != 0) : false;
            bool errordone              = false;

            mp_bitcnt_t resumeI = restored_ext ? (mp_bitcnt_t)resumeI_ext_ck : bits;
            uint64_t processed_bits_ext_base = restored_ext ? processedBits_ext_ck : 0;

            // Affichage de départ (même style que la branche normale)
            {
                std::string res64_x_ext;
                uint64_t perChunkDone0 = restored_ext ? (bits - resumeI) : 0;
                uint64_t globalDone0   = processed_bits_ext_base + perChunkDone0;
                spinner.displayProgress2(
                    globalDone0,
                    bits,
                    timer.elapsed() + restored_time,
                    timer2.elapsed(),
                    options.exponent,
                    globalDone0,
                    processed_bits_ext_base,
                    res64_x_ext,
                    guiServer_ ? guiServer_.get() : nullptr,
                    1,
                    1,
                    perChunkDone0,
                    bits,
                    true
                );
                timer2.start();
            }

            uint64_t lastIter_ext = (uint64_t)resumeI;

            // v36: keep RBASE in normal representation.  The previous 2-register
            // attempt stored RBASE itself in multiplicand representation and reused it.
            // That was compact but proved unreliable for the MM31 extension test.
            // With 3 registers on 16 GB devices, use the canonical Marin pattern:
            // prepare RTMP = multiplicand(RBASE) immediately before each multiply.
            for (mp_bitcnt_t i = resumeI; i > 0; --i) {
                lastIter_ext = (uint64_t)i;

                if (interrupted) {
                    std::cout << "\nInterrupted by user, state saved at iteration " << i << std::endl;
                    if (guiServer_) { std::ostringstream oss; oss << "\nInterrupted signal received\n "; guiServer_->appendLog(oss.str()); }

                    auto now_int = std::chrono::high_resolution_clock::now();
                    double elapsed_ext = std::chrono::duration<double>(now_int - start_clock).count() + restored_time;
                    save_ckpt_ext(
                        (uint32_t)lastIter_ext,
                        elapsed_ext,
                        gl_checkpass,
                        blocks_since_check,
                        bits_in_block,
                        current_block_len,
                        in_lot ? 1 : 0,
                        eacc,
                        wbits,
                        1,          // chunkIdx
                        0,          // startP
                        1,          // first
                        processed_bits_ext_base + (bits - i),
                        (uint64_t)bits
                    );
                    delete eng;
                    return 0;
                }

                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - lastBackup).count() >= options.backup_interval) {
                    std::cout << "\nBackup point done at i=" << i << " start...." << std::endl;
                    double elapsed_ext = std::chrono::duration<double>(now - start_clock).count() + restored_time;
                    save_ckpt_ext(
                        (uint32_t)lastIter_ext,
                        elapsed_ext,
                        gl_checkpass,
                        blocks_since_check,
                        bits_in_block,
                        current_block_len,
                        in_lot ? 1 : 0,
                        eacc,
                        wbits,
                        1,          // chunkIdx
                        0,          // startP
                        1,          // first
                        processed_bits_ext_base + (bits - i),
                        (uint64_t)bits
                    );
                    std::cout << "\nBackup point done at i=" << i << " done...." << std::endl;
                    lastBackup = now;
                }

                if (bits_in_block == 0) {
                    current_block_len = ((uint64_t)((i - 1) % B)) + 1;

                    // In the ultra-low-memory delta path we only use a compact register set
                    // RSTATE=0, RBASE=1, RTMP=2.  The Gerbicz/checkpoint
                    // helper registers are intentionally aliased away above
                    // so they MUST NOT be touched unless Gerbicz-Li is actually
                    // enabled.  v33/v34 reset RACC_L/RSAVE_* here even though
                    // Gerbicz-Li was disabled, which aliased to RSTATE and
                    // periodically did set(RSTATE, 1), corrupting H_old^Delta.
                    if (options.gerbiczli) {
                        if (current_block_len == B) {
                            if (gl_checkpass == 0 &&
                                blocks_since_check == 0 &&
                                wbits == 0 &&
                                eacc == 0)
                            {
                                eng->set(RACC_L, 1);
                                eng->set(RACC_R, 1);
                                eng->copy(RSAVE_S, RSTATE);
                                eng->set(RSAVE_L, 1);
                                eng->set(RSAVE_R, 1);
                                eacc = 0;
                                blocks_since_check = 0;
                                wbits = 0;
                                in_lot = true;
                            }
                        } else {
                            in_lot = false;
                            gl_checkpass = 0;
                            eacc = 0;
                            blocks_since_check = 0;
                            wbits = 0;
                        }
                        eng->copy(RSTART, RSTATE);
                    }
                }

                eng->square_mul(RSTATE);
                int b = mpz_tstbit(E_diff.get_mpz_t(), i - 1) ? 1 : 0;
                if (b) {
                    if (ultralowmem_delta2_extend) {
                        eng->mul(RSTATE, RBASE);
                    } else {
                        eng->set_multiplicand(RTMP, RBASE);
                        eng->mul(RSTATE, RTMP);
                    }
                }

                // Injection d'erreur comme dans la branche normale
                if (options.erroriter > 0 &&
                    (bits - i + 1) == options.erroriter &&
                    !errordone)
                {
                    errordone = true;
                    eng->sub(RSTATE, 33);
                    std::cout << "Injected error at iteration " << (bits - i + 1) << std::endl;
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss << "Injected error at iteration " << (bits - i + 1);
                        guiServer_->appendLog(oss.str());
                    }
                }

                wbits <<= 1;
                if (b) wbits += 1;
                bits_in_block += 1;
                
                bool end_block = (bits_in_block == current_block_len);
                if (end_block) {
                    if (options.gerbiczli && current_block_len == B) {
                        eng->set_multiplicand(RTMP, RSTART);
                        eng->mul(RACC_L, RTMP);
                        eng->set_multiplicand(RTMP, RSTATE);
                        eng->mul(RACC_R, RTMP);

                        eacc += wbits;
                        blocks_since_check += 1;
                        gl_checkpass += 1;

                        bool doCheck = options.gerbiczli && in_lot &&
                                       (gl_checkpass == checkpasslevel || i == 1);
                        if (doCheck) {
                            std::cout << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                                guiServer_->appendLog(oss.str());
                            }

                            eng->copy(RCHK, RACC_L);
                            for (uint64_t k = 0; k < B; ++k)
                                eng->square_mul(RCHK);

                            eng->set(RPOW, 1);
                            size_t eb = mpz_sizeinbase(eacc.get_mpz_t(), 2);
                            for (size_t k = eb; k-- > 0;) {
                                eng->square_mul(RPOW);
                                if (mpz_tstbit(eacc.get_mpz_t(), k)) {
                                    eng->set_multiplicand(RTMP, RBASE);
                                    eng->mul(RPOW, RTMP);
                                }
                            }
                            eng->set_multiplicand(RTMP, RPOW);
                            eng->mul(RCHK, RTMP);

                            const bool ok = eng->is_equal(RCHK, RACC_R);

                            if (!ok) {
                                std::cout << "[Gerbicz Li] Mismatch : Last correct state will be restored\n";
                                if (guiServer_) {
                                    std::ostringstream oss;
                                    oss << "[Gerbicz Li] Mismatch : Last correct state will be restored\n";
                                    guiServer_->appendLog(oss.str());
                                }
                                options.gerbicz_error_count += 1;
                                eng->copy(RSTATE, RSAVE_S);
                                eng->set(RACC_L, 1);
                                eng->set(RACC_R, 1);
                                eng->copy(RSTART, RSTATE);
                                i = (mp_bitcnt_t)(i + blocks_since_check * B);
                                eacc = 0;
                                blocks_since_check = 0;
                                wbits = 0;
                                gl_checkpass = 0;
                                bits_in_block = 0;
                                continue;
                            } else {
                                std::cout << "[Gerbicz Li] Check passed\n";
                                if (guiServer_) {
                                    std::ostringstream oss;
                                    oss << "[Gerbicz Li] Check passed\n";
                                    guiServer_->appendLog(oss.str());
                                }
                                eng->copy(RSAVE_S, RSTATE);
                                eng->set(RACC_L, 1);
                                eng->set(RACC_R, 1);
                                eacc = 0;
                                blocks_since_check = 0;
                                gl_checkpass = 0;
                            }
                        }
                    } else {
                        if (options.gerbiczli) {
                            std::cout << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                                guiServer_->appendLog(oss.str());
                            }
                            eng->copy(RCHK, RSTART);
                            for (uint64_t k = 0; k < current_block_len; ++k)
                                eng->square_mul(RCHK);

                            eng->set(RPOW, 1);
                            size_t wb = mpz_sizeinbase(wbits.get_mpz_t(), 2);
                            for (size_t k = wb; k-- > 0;) {
                                eng->square_mul(RPOW);
                                if (mpz_tstbit(wbits.get_mpz_t(), k)) {
                                    eng->set_multiplicand(RTMP, RBASE);
                                    eng->mul(RPOW, RTMP);
                                }
                            }
                            eng->set_multiplicand(RTMP, RPOW);
                            eng->mul(RCHK, RTMP);

                            const bool ok0 = eng->is_equal(RCHK, RSTATE);

                            if (!ok0) {
                                std::cout << "[Gerbicz Li] Mismatch : Last correct state will be restored\n";
                                if (guiServer_) {
                                    std::ostringstream oss;
                                    oss << "[Gerbicz Li] Mismatch : Last correct state will be restored\n";
                                    guiServer_->appendLog(oss.str());
                                }
                                options.gerbicz_error_count += 1;
                                eng->copy(RSTATE, RSTART);
                                i = (mp_bitcnt_t)(i + current_block_len);
                                wbits = 0;
                                bits_in_block = 0;
                                continue;
                            } else {
                                std::cout << "[Gerbicz Li] Check passed\n";
                                if (guiServer_) {
                                    std::ostringstream oss;
                                    oss << "[Gerbicz Li] Check passed\n";
                                    guiServer_->appendLog(oss.str());
                                }
                                eng->copy(RSAVE_S, RSTATE);
                                eng->set(RACC_L, 1);
                                eng->set(RACC_R, 1);
                                eacc = 0;
                                blocks_since_check = 0;
                                gl_checkpass = 0;
                            }
                        }
                    }
                    bits_in_block = 0;
                    wbits = 0;
                }

                auto now2 = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now2 - lastDisplay).count() >= 10) {
                    std::string res64_x_ext;
                    uint64_t bitsDoneNow = processed_bits_ext_base + (bits - i + 1);
                    spinner.displayProgress2(
                        bitsDoneNow,
                        bits,
                        timer.elapsed() + restored_time,
                        timer2.elapsed(),
                        options.exponent,
                        bitsDoneNow,
                        processed_bits_ext_base,
                        res64_x_ext,
                        guiServer_ ? guiServer_.get() : nullptr,
                        1,
                        1,
                        (bits - i + 1),
                        bits,
                        false
                    );
                    timer2.start();
                    lastDisplay = now2;
                }
            }

            if (bits_in_block != 0 && options.gerbiczli) {
                mpz_class wtail = wbits;
                uint64_t bt = bits_in_block;

                eng->copy(RCHK, RSTART);
                for (uint64_t k = 0; k < bt; ++k) eng->square_mul(RCHK);

                eng->set(RPOW, 1);
                size_t wbl = mpz_sizeinbase(wtail.get_mpz_t(), 2);
                for (size_t k = wbl; k-- > 0;) {
                    eng->square_mul(RPOW);
                    if (mpz_tstbit(wtail.get_mpz_t(), k)) {
                        eng->set_multiplicand(RTMP, RBASE);
                        eng->mul(RPOW, RTMP);
                    }
                }
                eng->set_multiplicand(RTMP, RPOW);
                eng->mul(RCHK, RTMP);

                const bool ok_tail = eng->is_equal(RCHK, RSTATE);

                if (!ok_tail) {
                    eng->copy(RSTATE, RSTART);
                    eng->copy(RCHK, RSTART);
                    for (uint64_t k = 0; k < bt; ++k) eng->square_mul(RCHK);
                    eng->set(RPOW, 1);
                    size_t wbl2 = mpz_sizeinbase(wtail.get_mpz_t(), 2);
                    for (size_t k = wbl2; k-- > 0;) {
                        eng->square_mul(RPOW);
                        if (mpz_tstbit(wtail.get_mpz_t(), k)) {
                            eng->set_multiplicand(RTMP, RBASE);
                            eng->mul(RPOW, RTMP);
                        }
                    }
                    eng->set_multiplicand(RTMP, RPOW);
                    eng->mul(RSTATE, RTMP);
                }
            }

            // Progress final pour l'extension
            {
                std::string res64_done_ext;
                uint64_t bitsDoneAll = processed_bits_ext_base + (uint64_t)bits;
                spinner.displayProgress2(
                    bitsDoneAll,
                    bits,
                    timer.elapsed() + restored_time,
                    timer2.elapsed(),
                    options.exponent,
                    bitsDoneAll,
                    processed_bits_ext_base,
                    res64_done_ext,
                    guiServer_ ? guiServer_.get() : nullptr,
                    1,
                    1,
                    1,
                    1,
                    true
                );
            }

            std::cout << "\nExtension exponentiation done.\n";
        }

        // Now RSTATE = X_old ^ E_diff = 3^{E(B1_new)}.

        mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
        mpz_class X  = compute_X_with_dots(eng, static_cast<engine::Reg>(RSTATE), Mp);

        std::string resume_save_path;
        std::string resume_p95_path;
        std::string ds;
        std::string de;
        const bool need_pm1_resume = options.resume || (p95_stage2_enabled && options.B2 > 0 && !(options.nmax > 0 && options.K > 0));
        if (need_pm1_resume) {
            auto now_sys = std::chrono::system_clock::now();
            auto fmt = [](const std::chrono::system_clock::time_point& tp){
                using namespace std::chrono;
                auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
                std::time_t tt = system_clock::to_time_t(tp);
                std::tm tmv{};
            #if defined(_WIN32)
                gmtime_s(&tmv, &tt);
            #else
                std::tm* tmp = std::gmtime(&tt);
                if (tmp) tmv = *tmp;
            #endif
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
                std::ostringstream s;
                s << buf << '.' << std::setw(3) << std::setfill('0') << (int)(ms.count());
                return s.str();
            };
            ds = fmt(now_sys);
            de = ds;
            resume_save_path = "resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(B1_new) + ".save";
            resume_p95_path  = "resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(B1_new) + ".p95";

            writeEcmResumeLine(resume_save_path, B1_new, options.exponent, X);
            convertEcmResumeToPrime95(resume_save_path, resume_p95_path, ds, de);
        }

        // Do gcd and results file exactly as in the normal Stage1 end:
        X -= 1;
        mpz_class g = gcd_with_dots(X, Mp);
        if (options.exponent == 2147483647ULL) {
            mpz_class q1("295257526626031");
            mpz_class q2("87054709261955177");
            mpz_class r1, r2;
            mpz_mod(r1.get_mpz_t(), X.get_mpz_t(), q1.get_mpz_t());
            mpz_mod(r2.get_mpz_t(), X.get_mpz_t(), q2.get_mpz_t());
            std::cout << "[DBG MM31] (X-1) mod q1 = " << r1 << "\n";
            std::cout << "[DBG MM31] (X-1) mod q2 = " << r2 << "\n";
            std::cout << "[DBG MM31] gcd bits = " << mpz_sizeinbase(g.get_mpz_t(), 2) << "\n";
        }
        bool factorFound = (g != 1) && (g != Mp);

        std::string filename = "stage1_result_B1_" + std::to_string(B1_new) +
                               "_p_" + std::to_string(options.exponent) + ".txt";

        if (factorFound) {
            char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
            writeStageResult(filename, "B1=" + std::to_string(B1_new) +
                                        "  factor=" + std::string(fstr));
            std::cout << "\nP-1 factor stage 1 found: " << fstr << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "\nP-1 factor stage 1 found: " << fstr << std::endl;
                guiServer_->appendLog(oss.str());
            }
            options.knownFactors.push_back(std::string(fstr));
            std::free(fstr);
            std::cout << "\n";
        } else {
            writeStageResult(filename, "No factor up to B1=" + std::to_string(B1_new));
            std::cout << "\nNo P-1 (stage 1) factor up to B1=" + std::to_string(B1_new) << "\n" << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "\nNo P-1 (stage 1) factor up to B1=" + std::to_string(B1_new) << "\n";
                guiServer_->appendLog(oss.str());
            }
        }
        uint64_t B2save = options.B2; 
        options.B2 = 0;
        // JSON + worktodo removal same as usual ...
        std::string json = io::JsonBuilder::generate(
            options,
            static_cast<int>(context.getTransformSize()),
            false,
            "",
            ""
        );
        std::cout << "Manual submission JSON:\n" << json << "\n";
        io::WorktodoManager wm(options);
        
        wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage1_ext", json);
        wm.appendToResultsTxt(json);
        options.B2 = B2save;
        if(options.B2 > 0){
            const double elapsed_time_ck =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count()
                + restored_time;

            save_ckpt(
                0,
                elapsed_time_ck,
                0,
                0,
                0,
                0,
                0,
                mpz_class(0),
                mpz_class(0),
                chunkIndex,
                startPrime,
                firstChunk ? 1 : 0,
                processed_total_bits,
                0
            );

            bool external_used = false;
            if (p95_stage2_enabled && !(options.nmax > 0 && options.K > 0) && !resume_save_path.empty()) {
                bool ext_found = false;
                external_used = run_pm1_stage2_external(resume_save_path, ds, de, ext_found);
                factorFound = ext_found || factorFound;
            }
            if (!external_used) {
                factorFound = runPM1Stage2Marin() || factorFound;
            }
        }

        if(options.nmax > 0 && options.K > 0){
            const double elapsed_time_ck =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count()
                + restored_time;

            save_ckpt(
                0,
                elapsed_time_ck,
                0,
                0,
                0,
                0,
                0,
                mpz_class(0),
                mpz_class(0),
                chunkIndex,
                startPrime,
                firstChunk ? 1 : 0,
                processed_total_bits,
                0
            );

            factorFound = runPM1Stage2MarinNKVersion() || factorFound;
        }

        delete_checkpoints(options.exponent, options.wagstaff, true, false);
        { std::error_code ec; fs::remove(pm1_checkpoint_backend_sidecar(ckpt_file), ec); }
        delete eng;
        if (hasWorktodoEntry_) {
                if (worktodoParser_->removeFirstProcessed()) {
                    std::cout << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n";
                    if (guiServer_) { std::ostringstream oss; oss << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n"; guiServer_->appendLog(oss.str()); }
                    std::ifstream f(options.worktodo_path);
                    std::string l; bool more = false; while (std::getline(f, l)) { if (!l.empty() && l[0] != '#') { more = true; break; } }
                    f.close();
                    if (more) { std::cout << "Restarting for next entry in worktodo.txt\n"; if (guiServer_) { std::ostringstream oss; oss << "Restarting for next entry in worktodo.txt\n"; guiServer_->appendLog(oss.str()); } restart_self(argc_, argv_); }
                    else { std::cout << "No more entries in worktodo.txt, exiting.\n"; if (guiServer_) { std::ostringstream oss; oss << "No more entries in worktodo.txt, exiting.\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(0);} }
                } else {
                    std::cerr << "Failed to update " << options.worktodo_path << "\n"; if (guiServer_) { std::ostringstream oss; oss << "Failed to update " << options.worktodo_path << "\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(-1);}
                }
            }
        return factorFound ? 0 : 1;
    }
    else{
        while (true) {
            bool errordone = false;
            bool useFast3Candidate = firstChunk;
            uint64_t nextStart = 0;
            mpz_class Echunk;
            if (useFast3Candidate) {
                uint64_t twoe = 2ULL * (uint64_t)options.exponent;
                uint64_t extra = 0; { uint64_t t = twoe; while (t) { extra++; t >>= 1; } if (extra == 0) extra = 1; }
                uint64_t estBits = (uint64_t)std::ceil(L_est_bits) + extra + 8;
                if (estBits <= MAX_E_BITS) { Echunk = buildE(B1); nextStart = 0; }
                else { Echunk = buildE2(B1, startPrime, MAX_E_BITS, nextStart, firstChunk); }
            } else {
                Echunk = buildE2(B1, startPrime, MAX_E_BITS, nextStart, firstChunk);
            }
            if (firstChunk) Echunk *= mpz_class(2) * mpz_from_u64(options.exponent);
            bool useFast3 = useFast3Candidate && (nextStart == 0) && !aevum_backend;
            if (pm1_ultralowmem_stage1 && !useFast3) {
                throw std::runtime_error("-pm1-ultralowmem requires the fast3 single-chunk path; use a smaller B1 or a larger -maxebits value, or use -pm1-lowmem");
            }
            mp_bitcnt_t bits = mpz_sizeinbase(Echunk.get_mpz_t(), 2);
            if (bits == 0) break;
            if (restored && bits_in_chunk_ck) bits = (mp_bitcnt_t)bits_in_chunk_ck;
            chunkIndex = std::max<uint64_t>(chunkIndex, 1);
            std::cout << "\nChunk " << chunkIndex << "/" << estChunks << "  bits=" << bits << (useFast3 ? " [fast3]" : "") << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "Chunk " << chunkIndex << "/" << estChunks << "  bits=" << bits << (useFast3 ? " [fast3]" : ""); guiServer_->appendLog(oss.str()); }
            uint64_t B = std::max<uint64_t>(1, (uint64_t)std::sqrt((double)bits));
            double desiredIntervalSeconds = 600.0;
            uint64_t checkpass = 0;
            uint64_t checkpasslevel_auto = (uint64_t)((1000 * desiredIntervalSeconds) / (double)B);
            if (checkpasslevel_auto == 0) checkpasslevel_auto = ((uint64_t)bits/B)/((uint64_t)(std::sqrt((double)B)));
            uint64_t checkpasslevel = (options.checklevel > 0)
                ? options.checklevel
                : checkpasslevel_auto;
            if(checkpasslevel==0)
                checkpasslevel=1;
            //uint64_t checkpass = (options.checklevel > 0) ? options.checklevel : 1;
            //auto chunkStart = std::chrono::high_resolution_clock::now();
            //bool tunedCheckpass = false;
            uint64_t resumeI = restored ? (uint64_t)resumeI_ck : (uint64_t)bits;
            uint64_t lastIter = resumeI;
            uint64_t blocks_since_check = restored ? gl_blocks_since_check_ck : 0;
            uint64_t bits_in_block = restored ? gl_bits_in_block_ck : 0;
            uint64_t current_block_len = restored && gl_current_block_len_ck ? gl_current_block_len_ck : (((uint64_t)((resumeI - 1) % B)) + 1);
            mpz_class eacc = restored ? eacc_ck : 0;
            mpz_class wbits = restored ? wbits_ck : 0;
            uint64_t gl_checkpass = restored ? gl_checkpass_ck : 0;
            bool in_lot = restored ? (in_lot_ck != 0) : false;
            spinner.displayProgress2(
                processed_total_bits + (restored ? (bits - resumeI) : 0),
                processed_total_bits + bits,
                timer.elapsed() + restored_time,
                timer2.elapsed(),
                options.exponent,
                processed_total_bits + (restored ? (bits - resumeI) : 0),
                processed_total_bits,
                "",
                guiServer_ ? guiServer_.get() : nullptr,
                chunkIndex,
                estChunks,
                (restored ? (bits - resumeI) : 0),
                bits,
                true
            );
            if (!restored) {
                if (firstChunk) {
                    if (!pm1_ultralowmem_stage1) eng->set(RBASE, 3);
                    eng->set(RSTATE, 1);
                } else {
                    if (pm1_ultralowmem_stage1) {
                        throw std::runtime_error("-pm1-ultralowmem supports only a single fast3 chunk; increase MAX_E_BITS or use -pm1-lowmem");
                    }
                    eng->copy(RBASE, RSTATE);
                    eng->set(RSTATE, 1);
                }
            }
            for (mp_bitcnt_t i = (mp_bitcnt_t)resumeI; i > 0; --i) {
                lastIter = i;
                if (interrupted) {
                    std::cout << "\nInterrupted by user, state saved at iteration " << i << std::endl;
                    if (guiServer_) { std::ostringstream oss; oss << "\nInterrupted signal received\n "; guiServer_->appendLog(oss.str()); }
                    save_ckpt((uint32_t)lastIter, std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time, gl_checkpass, blocks_since_check, bits_in_block, current_block_len, in_lot ? 1 : 0, eacc, wbits, chunkIndex, startPrime, firstChunk ? 1 : 0, processed_total_bits + (bits - i), (uint64_t)bits);
                    delete eng;
                    return 0;
                }
                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - lastBackup).count() >= options.backup_interval) {
                    std::cout << "\nBackup point done at i=" << i << " start...." << std::endl;
                    save_ckpt((uint32_t)lastIter, std::chrono::duration<double>(now - start_clock).count() + restored_time, gl_checkpass, blocks_since_check, bits_in_block, current_block_len, in_lot ? 1 : 0, eacc, wbits, chunkIndex, startPrime, firstChunk ? 1 : 0, processed_total_bits + (bits - i), (uint64_t)bits);
                    std::cout << "\nBackup point done at i=" << i << " done...." << std::endl;
                    lastBackup = now;
                }
                if (bits_in_block == 0) {
                    current_block_len = ((uint64_t)((i - 1) % B)) + 1;
                    if (options.gerbiczli) {
                        if (current_block_len == B) {
                            if (gl_checkpass == 0 && blocks_since_check == 0 && wbits == 0 && eacc == 0) {
                                eng->set(RACC_L, 1);
                                eng->set(RACC_R, 1);
                                eng->copy(RSAVE_S, RSTATE);
                                eng->set(RSAVE_L, 1);
                                eng->set(RSAVE_R, 1);
                                eacc = 0;
                                blocks_since_check = 0;
                                wbits = 0;
                                in_lot = true;
                            }
                        } else {
                            in_lot = false;
                            gl_checkpass = 0;
                            eacc = 0;
                            blocks_since_check = 0;
                            wbits = 0;
                        }
                        eng->copy(RSTART, RSTATE);
                    }
                }
                int b = mpz_tstbit(Echunk.get_mpz_t(), i - 1) ? 1 : 0;
                if (useFast3) { if (b) eng->square_mul(RSTATE, 3); else eng->square_mul(RSTATE); }
                else {
                    if (pm1_ultralowmem_stage1) throw std::runtime_error("internal error: ultra-low-memory entered non-fast3 path");
                    eng->square_mul(RSTATE); if (b) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RSTATE, RTMP); }
                }
                wbits <<= 1; if (b) wbits += 1;
                bits_in_block += 1;
                if (options.erroriter > 0 && (resumeI - i + 1) == options.erroriter && !errordone) { errordone = true; eng->sub(RSTATE, 2); std::cout << "Injected error at iteration " << (resumeI - i + 1) << std::endl; if (guiServer_) { std::ostringstream oss; oss << "Injected error at iteration " << (resumeI - i + 1); guiServer_->appendLog(oss.str()); } }
                bool end_block = (bits_in_block == current_block_len);
                if (end_block) {
                    if (options.gerbiczli && current_block_len == B) {
                        eng->set_multiplicand(RTMP, RSTART);
                        eng->mul(RACC_L, RTMP);
                        eng->set_multiplicand(RTMP, RSTATE);
                        eng->mul(RACC_R, RTMP);
                        eacc += wbits;
                        blocks_since_check += 1;
                        gl_checkpass += 1;
                        /*if (!tunedCheckpass && options.checklevel == 0) {
                            uint64_t processedChunk = bits - i + 1;
                            double elapsedChunk = std::chrono::duration<double>(now - chunkStart).count();
                            if (elapsedChunk > 0.0 && processedChunk >= B) {
                                double sampleIps = (double)processedChunk / elapsedChunk;
                                uint64_t checkpasslevel_auto = (uint64_t)((sampleIps * desiredIntervalSeconds) / (double)B);
                                if (checkpasslevel_auto == 0) checkpasslevel_auto = std::max<uint64_t>(1, (bits / B) / (uint64_t)std::sqrt((double)B));
                                checkpass = checkpasslevel_auto;
                                tunedCheckpass = true;
                            }
                        }*/
                        bool doCheck = in_lot && (gl_checkpass == checkpass || i == 1);
                        if (doCheck) {
                            std::cout << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                            eng->copy(RCHK, RACC_L);
                            for (uint64_t k = 0; k < B; ++k) eng->square_mul(RCHK);
                            eng->set(RPOW, 1);
                            size_t eb = mpz_sizeinbase(eacc.get_mpz_t(), 2);
                            for (size_t k = eb; k-- > 0;) {
                                if (useFast3) { if (mpz_tstbit(eacc.get_mpz_t(), k)) eng->square_mul(RPOW, 3); else eng->square_mul(RPOW); }
                                else { eng->square_mul(RPOW); if (mpz_tstbit(eacc.get_mpz_t(), k)) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RPOW, RTMP); } }
                            }
                            eng->set_multiplicand(RTMP, RPOW);
                            eng->mul(RCHK, RTMP);
                            const bool ok = eng->is_equal(RCHK, RACC_R);
                            if (!ok) { std::cout << "[Gerbicz Li] Mismatch : Last correct state will be restored\n"; if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz Li] Mismatch : Last correct state will be restored\n"; guiServer_->appendLog(oss.str()); } options.gerbicz_error_count += 1; eng->copy(RSTATE, RSAVE_S); eng->set(RACC_L, 1); eng->set(RACC_R, 1); eng->copy(RSTART, RSTATE); i = (mp_bitcnt_t)(i + blocks_since_check * B); eacc = 0; blocks_since_check = 0; wbits = 0; gl_checkpass = 0; bits_in_block = 0; continue; }
                            else { std::cout << "[Gerbicz Li] Check passed\n"; if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz Li] Check passed\n"; guiServer_->appendLog(oss.str()); } eng->copy(RSAVE_S, RSTATE); eng->set(RACC_L, 1); eng->set(RACC_R, 1); eacc = 0; blocks_since_check = 0; gl_checkpass = 0; }
                        }
                    } else if (options.gerbiczli) {
                        {
                            eng->copy(RCHK, RSTART);
                            for (uint64_t k = 0; k < current_block_len; ++k) eng->square_mul(RCHK);
                            eng->set(RPOW, 1);
                            size_t wb = mpz_sizeinbase(wbits.get_mpz_t(), 2);
                            for (size_t k = wb; k-- > 0;) {
                                if (useFast3) { if (mpz_tstbit(wbits.get_mpz_t(), k)) eng->square_mul(RPOW, 3); else eng->square_mul(RPOW); }
                                else { eng->square_mul(RPOW); if (mpz_tstbit(wbits.get_mpz_t(), k)) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RPOW, RTMP); } }
                            }
                            eng->set_multiplicand(RTMP, RPOW);
                            eng->mul(RCHK, RTMP);
                            const bool ok0 = eng->is_equal(RCHK, RSTATE);
                            if (!ok0) { std::cout << "[Gerbicz Li] Mismatch : Last correct state will be restored\n"; if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz Li] Mismatch : Last correct state will be restored\n"; guiServer_->appendLog(oss.str()); } options.gerbicz_error_count += 1; eng->copy(RSTATE, RSTART); i = (mp_bitcnt_t)(i + current_block_len); wbits = 0; bits_in_block = 0; continue; }
                            else { std::cout << "[Gerbicz Li] Check passed\n"; if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz Li] Check passed\n"; guiServer_->appendLog(oss.str()); } eng->copy(RSAVE_S, RSTATE); eng->set(RACC_L, 1); eng->set(RACC_R, 1); eacc = 0; blocks_since_check = 0; gl_checkpass = 0; }
                        }
                    }
                    bits_in_block = 0;
                    wbits = 0;
                }
                auto now2 = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now2 - lastDisplay).count() >= 10) {
                    std::string res64_x;
                    spinner.displayProgress2(
                        processed_total_bits + (bits - i + 1),
                        processed_total_bits + bits,
                        timer.elapsed() + restored_time,
                        timer2.elapsed(),
                        options.exponent,
                        processed_total_bits + (bits - i + 1),
                        processed_total_bits,
                        res64_x,
                        guiServer_ ? guiServer_.get() : nullptr,
                        chunkIndex,
                        estChunks,
                        (bits - i + 1),
                        bits,
                        false
                    );
                    timer2.start();
                    lastDisplay = now2;
                }
            }
            if (bits_in_block != 0 && options.gerbiczli) {
                mpz_class wtail = wbits;
                uint64_t bt = bits_in_block;
                eng->copy(RCHK, RSTART);
                for (uint64_t k = 0; k < bt; ++k) eng->square_mul(RCHK);
                eng->set(RPOW, 1);
                size_t wbl = mpz_sizeinbase(wtail.get_mpz_t(), 2);
                for (size_t k = wbl; k-- > 0;) {
                    if (useFast3) { if (mpz_tstbit(wtail.get_mpz_t(), k)) eng->square_mul(RPOW, 3); else eng->square_mul(RPOW); }
                    else { eng->square_mul(RPOW); if (mpz_tstbit(wtail.get_mpz_t(), k)) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RPOW, RTMP); } }
                }
                eng->set_multiplicand(RTMP, RPOW);
                eng->mul(RCHK, RTMP);
                const bool ok_tail = eng->is_equal(RCHK, RSTATE);
                if (!ok_tail) { eng->copy(RSTATE, RSTART); eng->copy(RCHK, RSTART); for (uint64_t k = 0; k < bt; ++k) eng->square_mul(RCHK); eng->set(RPOW, 1); size_t wbl2 = mpz_sizeinbase(wtail.get_mpz_t(), 2); for (size_t k = wbl2; k-- > 0;) { if (useFast3) { if (mpz_tstbit(wtail.get_mpz_t(), k)) eng->square_mul(RPOW, 3); else eng->square_mul(RPOW); } else { eng->square_mul(RPOW); if (mpz_tstbit(wtail.get_mpz_t(), k)) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RPOW, RTMP); } } } eng->set_multiplicand(RTMP, RPOW); eng->mul(RSTATE, RTMP); }
                bits_in_block = 0;
                wbits = 0;
            }
            processed_total_bits += bits;
            restored = false;
            firstChunk = false;
            if (nextStart == 0) break;
            startPrime = nextStart | 1ULL;
            chunkIndex += 1;
        }
    }
    const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
    std::string res64_done;
    spinner.displayProgress2(
        processed_total_bits,
        processed_total_bits,
        timer.elapsed() + restored_time,
        timer2.elapsed(),
        options.exponent,
        processed_total_bits,
        processed_total_bits,
        res64_done,
        guiServer_ ? guiServer_.get() : nullptr,
        std::max<uint64_t>(chunkIndex, estChunks),
        estChunks,
        1,
        1,
        true
    );
    std::cout << "Elapsed time = " << std::fixed << std::setprecision(2) << elapsed_time << " s." << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "Elapsed time = " << std::fixed << std::setprecision(2) << elapsed_time << " s." << std::endl; guiServer_->appendLog(oss.str()); }
    //engine::digit d(eng, RSTATE);
    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    mpz_class X  = compute_X_with_dots(eng, static_cast<engine::Reg>(RSTATE), Mp);
    auto end_sys = std::chrono::system_clock::now();
    auto fmt = [](const std::chrono::system_clock::time_point& tp){
        using namespace std::chrono;
        auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
        std::time_t tt = std::chrono::system_clock::to_time_t(tp);
        std::tm tmv{};
        #if defined(_WIN32)
        gmtime_s(&tmv, &tt);
        #else
        std::tm* tmp = std::gmtime(&tt);
        if (tmp) tmv = *tmp;
        #endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
        std::ostringstream s;
        s << buf << '.' << std::setw(3) << std::setfill('0') << (int)(ms.count());
        return s.str();
    };
    std::string ds = fmt(start_sys);
    std::string de = fmt(end_sys);
    std::string resume_save_path;
    std::string resume_p95_path;
    const bool need_pm1_resume = options.resume || (p95_stage2_enabled && options.B2 > 0 && !(options.nmax > 0 && options.K > 0));
    if (need_pm1_resume) {
        resume_save_path = "resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(options.B1) + ".save";
        resume_p95_path  = "resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(options.B1) + ".p95";
        writeEcmResumeLine(resume_save_path, options.B1, options.exponent, X);
        convertEcmResumeToPrime95(resume_save_path, resume_p95_path, ds, de);
    }
    bool factorFound = false;
    std::string filename = "stage1_result_B1_" + std::to_string(B1) + "_p_" + std::to_string(options.exponent) + ".txt";
    if (options.pm1_no_stage1_gcd) {
        writeStageResult(filename, "Stage 1 GCD skipped at B1=" + std::to_string(B1));
        std::cout << "\n[PM1] Stage 1 ordinary GCD skipped by -nogcd-stage1.\n";
        std::cout << "[PM1] PM1 resume/checkpoint was still written; continuing to Stage 2 if requested.\n\n";
        if (guiServer_) guiServer_->appendLog("[PM1] Stage 1 ordinary GCD skipped by -nogcd-stage1.");
    } else {
        X -= 1;
        mpz_class g = gcd_with_dots(X, Mp);
        factorFound = (g != 1) && (g != Mp);
        if (factorFound) {
            char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
            writeStageResult(filename, "B1=" + std::to_string(B1) + "  factor=" + std::string(fstr));
            std::cout << "\nP-1 factor stage 1 found: " << fstr << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "\nP-1 factor stage 1 found: " << fstr << std::endl; guiServer_->appendLog(oss.str()); }
            options.knownFactors.push_back(std::string(fstr));
            std::free(fstr);
            std::cout << "\n";
        } else {
            writeStageResult(filename, "No factor up to B1=" + std::to_string(B1));
            std::cout << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl; guiServer_->appendLog(oss.str()); }
        }
    }
    uint64_t B2save = options.B2; 
    options.B2 = 0;
    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
    std::cout << "Manual submission JSON:\n" << json << "\n";
    io::WorktodoManager wm(options);
    options.B2 = 0;
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage1", json);
    wm.appendToResultsTxt(json);
    options.B2 = B2save;

    if(options.B2 > 0){
        {
            const double elapsed_time_ck =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count()
                + restored_time;

            save_ckpt(
                0,                 // i
                elapsed_time_ck,   // et
                0,                 // chk
                0,                 // blks
                0,                 // bib
                0,                 // cbl
                0,                 // inlot
                mpz_class(0),      // ceacc
                mpz_class(0),      // cwbits
                chunkIndex,        // chunkIdx
                startPrime,        // startP
                firstChunk ? 1 : 0,// first
                processed_total_bits, // processedBits
                0                  // bitsInChunk
            );
        }
        bool external_used = false;
        if (p95_stage2_enabled && !(options.nmax > 0 && options.K > 0) && !resume_save_path.empty()) {
            bool ext_found = false;
            external_used = run_pm1_stage2_external(resume_save_path, ds, de, ext_found);
            factorFound = ext_found || factorFound;
        }
        if (!external_used) {
            // Stage 2 may need to allocate a fresh GPU engine.  In PM1 low-memory
            // modes we perform an aggressive and observable GPU handoff: finish queues,
            // release kernels/buffers/program/context, clear host-side large vectors,
            // then leave a short delay for NVIDIA/ROCm drivers to actually retire the
            // freed VRAM before allocating the Stage 2 engine.  Normal PM1 mode keeps
            // the original simple destruction path to avoid regressions.
            if (options.pm1_lowmem) {
                std::cout << "[PM1] Low-memory GPU handoff: explicitly releasing Stage 1 engine before Stage 2..." << std::flush;
                if (eng != nullptr) {
                    eng->release_gpu_resources_for_lowmem_handoff();
                    delete eng;
                    eng = nullptr;
                }
                std::cout << " done.\n";
                std::cout << "[PM1] Low-memory GPU handoff: waiting briefly for driver VRAM retirement before Stage 2 allocation..." << std::flush;
                std::this_thread::sleep_for(std::chrono::milliseconds(options.pm1_ultralowmem ? 2500 : 1000));
                std::cout << " done.\n";
            } else {
                delete eng;
                eng = nullptr;
            }
            factorFound = runPM1Stage2Marin() || factorFound;
        }
    }
   if(options.nmax > 0 && options.K > 0){
        {
            std::cout << "P-1 STAGE 2 IN **** n^K variant  n=" << options.nmax << " K=" << options.K << "******\n";
    
            const double elapsed_time_ck =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count()
                + restored_time;

            save_ckpt(
                0,                 // i
                elapsed_time_ck,   // et
                0,                 // chk
                0,                 // blks
                0,                 // bib
                0,                 // cbl
                0,                 // inlot
                mpz_class(0),      // ceacc
                mpz_class(0),      // cwbits
                chunkIndex,        // chunkIdx
                startPrime,        // startP
                firstChunk ? 1 : 0,// first
                processed_total_bits, // processedBits
                0                  // bitsInChunk
            );
        }
        //options.B2 = 214439;
        if (eng != nullptr) {
            delete eng;
            eng = nullptr;
        }
        factorFound = runPM1Stage2MarinNKVersion() || factorFound;
    }
    //else{
    delete_checkpoints(options.exponent, options.wagstaff, true, false);
    { std::error_code ec; fs::remove(pm1_checkpoint_backend_sidecar(ckpt_file), ec); }
    if (eng != nullptr) delete eng;
    if (hasWorktodoEntry_) {
        if (worktodoParser_->removeFirstProcessed()) {
            std::cout << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n";
            if (guiServer_) { std::ostringstream oss; oss << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n"; guiServer_->appendLog(oss.str()); }
            std::ifstream f(options.worktodo_path);
            std::string l; bool more = false; while (std::getline(f, l)) { if (!l.empty() && l[0] != '#') { more = true; break; } }
            f.close();
            if (more) { std::cout << "Restarting for next entry in worktodo.txt\n"; if (guiServer_) { std::ostringstream oss; oss << "Restarting for next entry in worktodo.txt\n"; guiServer_->appendLog(oss.str()); } restart_self(argc_, argv_); }
            else { std::cout << "No more entries in worktodo.txt, exiting.\n"; if (guiServer_) { std::ostringstream oss; oss << "No more entries in worktodo.txt, exiting.\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(0);} }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n"; if (guiServer_) { std::ostringstream oss; oss << "Failed to update " << options.worktodo_path << "\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(-1);}
        }
    }
    //}
    //delete eng;
    return factorFound ? 0 : 1;
}

/*
2025 - Cherubrock
P-1 Stage 3 — Paired Multiplicative Offsets (k = 0, +/-1, +/-2, …)
*/

int App::runPM1Stage3Marin() {
    using namespace std::chrono;

    if (guiServer_) { std::ostringstream oss; oss << "P-1 factoring stage 3"; guiServer_->setStatus(oss.str()); }
    const uint64_t B3u = options.B3;
    if (B3u == 0) { std::cout << "Stage 3 skipped (B3=0)\n"; if (guiServer_) { std::ostringstream oss; oss << "Stage 3 skipped (B3=0)"; guiServer_->appendLog(oss.str()); } return 1; }

    const uint32_t pexp = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;

    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    bool foundIntermediate = false;
    mpz_class midFactor;

    static constexpr size_t baseRegsStage1 = 11;
    // +2 registres pour H^k et H^{-k}
    const size_t RSTATE=0, RACC=1, RSQ=2, RMINUS=3, RPLUS=4, RPOW=5, RHINV=6, RHK=7, RHIK=8, RHINV_2=9, RHINV_3=10, RHK_2=11, RHK_3=12, RSTART_SQ=13, RCHK_SQ=14;
    static constexpr size_t regsS3 = 15;

    std::ostringstream ck3; ck3 << "pm1_s3_m_" << pexp << ".ckpt";
    const std::string ckpt_file_s3 = ck3.str();

    auto read_ckpt_s3 = [&](engine* e, const std::string& file, uint64_t& saved_b, double& et, uint64_t& sB3)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 2) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != pexp) return -2;
        if (!f.read(reinterpret_cast<char*>(&sB3), sizeof(sB3))) return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_b), sizeof(saved_b))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = e->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!e->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };
    auto save_ckpt_s3 = [&](engine* e, uint64_t cur_b, double et){
        const std::string oldf = ckpt_file_s3 + ".old", newf = ckpt_file_s3 + ".new";
        {
            File f(newf, "wb");
            int version = 2;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&pexp), sizeof(pexp))) return;
            if (!f.write(reinterpret_cast<const char*>(&B3u), sizeof(B3u))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_b), sizeof(cur_b))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;
            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!e->get_checkpoint(data)) return;
            if (!f.write(data.data(), cksz)) return;
            f.write_crc32();
        }
        std::remove(oldf.c_str());
        struct stat s;
        if ((stat(ckpt_file_s3.c_str(), &s) == 0) && (std::rename(ckpt_file_s3.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file_s3.c_str());
    };

    engine* eng = engine::create_gpu(pexp, regsS3, static_cast<size_t>(options.device_id), verbose);

    // Charger H (RSTATE) et H^{-1} (RHINV) depuis checkpoint S1
    {
        mpz_t Hs1; mpz_init(Hs1);

        if (options.s3only) {
            uint64_t B1_file = 0;
            uint32_t p_file  = 0;
            mpz_class X_s1;

            std::string basePath = options.pm1_extend_save_path;
            if (basePath.empty()) {
                basePath = "resume_p" + std::to_string(options.exponent) +
                           "_B1_" + std::to_string(options.B1);
            }

            std::string resumeSave = basePath;
            std::string resumeP95  = basePath;

            if (resumeSave.size() >= 5 &&
                resumeSave.substr(resumeSave.size() - 5) == ".save")
            {
                resumeP95 = resumeSave.substr(resumeSave.size() - 5) + ".p95";
            }
            else if (resumeSave.size() >= 4 &&
                     resumeSave.substr(resumeSave.size() - 4) == ".p95")
            {
                resumeP95  = resumeSave;
                resumeSave = resumeSave.substr(resumeSave.size() - 4) + ".save";
            }
            else {
                resumeSave += ".save";
                resumeP95  += ".p95";
            }

            std::string usedPath;

            if (load_pm1_s1_from_save(resumeSave, B1_file, p_file, X_s1)) {
                usedPath = resumeSave;
            }
            else if (load_pm1_s1_from_p95(resumeP95, B1_file, p_file, X_s1)) {
                usedPath = resumeP95;
            }
            else {
                std::cerr << "Cannot load PM1 S1 state from \"" << resumeSave
                          << "\" nor from \"" << resumeP95 << "\"\n";
                mpz_clear(Hs1);
                delete eng;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Stage 3: cannot load PM1 S1 state from " << resumeSave
                        << " nor " << resumeP95;
                    guiServer_->appendLog(oss.str());
                }
                return -2;
            }

            if (p_file != pexp) {
                std::cerr << "Mismatch between S1 resume file (p=" << p_file
                          << ") and options (p=" << pexp << ")\n";
                mpz_clear(Hs1);
                delete eng;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Stage 3: mismatch between S1 resume file (p=" << p_file
                        << ") and options (p=" << pexp << ")";
                    guiServer_->appendLog(oss.str());
                }
                return -2;
            }

            std::cout << "Stage 3: using PM1 S1 state from " << usedPath << "\n";
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Stage 3: using PM1 S1 state from " << usedPath;
                guiServer_->appendLog(oss.str());
            }

            mpz_set(Hs1, X_s1.get_mpz_t());
        } else {
            engine* eng_load = engine::create_gpu(pexp, baseRegsStage1, static_cast<size_t>(options.device_id), verbose);
            std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
            const std::string ckpt_file = ck.str();

            auto read_ckpt_s1 = [&](engine* e, const std::string& file)->int{
                File f(file);
                if (!f.exists()) return -1;
                int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
                if (version != 3) return -2;
                uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
                if (rp != pexp) return -2;
                uint32_t ri = 0; double et = 0.0;
                if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
                const size_t cksz = e->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!f.read(data.data(), cksz)) return -2;
                if (!e->set_checkpoint(data)) return -2;
                uint64_t tmp64;
                if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
                if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
                if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
                if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
                uint8_t inlot=0; if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
                uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
                if (eacc_len) { std::string skip; skip.resize(eacc_len); if (!f.read(skip.data(), eacc_len)) return -2; }
                uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
                if (wbits_len) { std::string skip; skip.resize(wbits_len); if (!f.read(skip.data(), wbits_len)) return -2; }
                uint64_t chunkIdx=0, startP=0; uint8_t first=0; uint64_t processedBits=0, bitsInChunk=0;
                if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
                if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
                if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
                if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
                if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
                if (!f.check_crc32()) return -2;
                return 0;
            };

            int rr = read_ckpt_s1(eng_load, ckpt_file);
            if (rr < 0) rr = read_ckpt_s1(eng_load, ckpt_file + ".old");
            if (rr != 0) { delete eng_load; delete eng; std::cerr << "Stage 3: cannot load pm1 stage1 checkpoint.\n"; if (guiServer_) { std::ostringstream oss; oss << "Stage 3: cannot load pm1 stage1 checkpoint.\n"; guiServer_->appendLog(oss.str()); } mpz_clear(Hs1); return -2; }

            eng_load->get_mpz(Hs1, static_cast<engine::Reg>(RSTATE));
            delete eng_load;
        }

        mpz_t Mp_local, Hinv;
        mpz_init(Mp_local); mpz_init(Hinv);
        mpz_ui_pow_ui(Mp_local, 2, pexp); mpz_sub_ui(Mp_local, Mp_local, 1);
        if (mpz_invert(Hinv, Hs1, Mp_local) == 0) {
            mpz_clear(Mp_local); mpz_clear(Hinv); mpz_clear(Hs1);
            delete eng;
            std::cerr << "Stage 3: H not invertible mod Mp.\n";
            if (guiServer_) { std::ostringstream oss; oss << "Stage 3: H not invertible mod Mp."; guiServer_->appendLog(oss.str()); }
            return -3;
        }
        eng->set_mpz(static_cast<engine::Reg>(RSTATE), Hs1);     // H
        eng->set_mpz(static_cast<engine::Reg>(RHINV),  Hinv);    // H^{-1}
        mpz_clear(Mp_local); mpz_clear(Hinv); mpz_clear(Hs1);
    }

    eng->set(static_cast<engine::Reg>(RACC), 1);
    eng->copy(static_cast<engine::Reg>(RSQ),  static_cast<engine::Reg>(RSTATE));

    bool resumed_s3 = false;
    uint64_t resume_b = 0, s3B3 = 0;
    double restored_time = 0.0;
    {
        int rs3 = read_ckpt_s3(eng, ckpt_file_s3, resume_b, restored_time, s3B3);
        if (rs3 == 0 && s3B3 == B3u) {
            resumed_s3 = true;
            if (guiServer_) { std::ostringstream oss; oss << "Resuming Stage 3 at b=" << resume_b << "/" << B3u; guiServer_->appendLog(oss.str()); }
            std::cout << "Resuming Stage 3 at b=" << resume_b << "/" << B3u << "\n";
        }
    }

    auto t0 = high_resolution_clock::now();
    auto lastBackup  = t0;
    auto lastDisplay = t0;

    const uint64_t Kmax = std::min<uint64_t>(3, std::max<uint64_t>(1, options.B4));
    uint64_t b = resumed_s3 ? resume_b : 0;
    if (!resumed_s3) {
        std::cout << "Start P-1 Stage 3 up to B3=" << B3u << " [+/-k, k<=" << Kmax << "]\n";
        if (guiServer_) { std::ostringstream oss; oss << "Start P-1 Stage 3 up to B3=" << B3u << " [+/-k, k<=" << Kmax << "]"; guiServer_->appendLog(oss.str()); }
    }

    // --- Inverses : RHINV = H^{-1}, RHINV_2 = H^{-2}, RHINV_3 = H^{-3}
    eng->copy((engine::Reg)RHIK, (engine::Reg)RHINV);              // RHIK = H^{-1}
    eng->set_multiplicand((engine::Reg)RHINV, (engine::Reg)RHIK);  // mul(..., RHINV) = *H^{-1}
    eng->square_mul((engine::Reg)RHIK);                            // RHIK = (H^{-1})^2 = H^{-2}
    eng->set_multiplicand((engine::Reg)RHINV_2, (engine::Reg)RHIK);// RHINV_2 = H^{-2}
    eng->mul((engine::Reg)RHIK, (engine::Reg)RHINV);               // RHIK *= H^{-1} -> H^{-3}
    eng->set_multiplicand((engine::Reg)RHINV_3, (engine::Reg)RHIK);// RHINV_3 = H^{-3}

    // --- Puissances positives : RHK = H, RHK_2 = H^2, RHK_3 = H^3
    eng->copy((engine::Reg)RPOW, (engine::Reg)RSTATE);             // RPOW = H
    eng->set_multiplicand((engine::Reg)RHK, (engine::Reg)RPOW);    // RHK = H

    eng->square_mul((engine::Reg)RPOW);                            // RPOW = H^2
    eng->set_multiplicand((engine::Reg)RHK_2, (engine::Reg)RPOW);  // RHK_2 = H^2

    eng->mul((engine::Reg)RPOW, (engine::Reg)RHK);                 // RPOW *= H -> H^3
    eng->set_multiplicand((engine::Reg)RHK_3, (engine::Reg)RPOW);  // RHK_3 = H^3

    eng->square_mul(static_cast<engine::Reg>(RSQ));
    eng->square_mul(static_cast<engine::Reg>(RSQ));
    eng->square_mul(static_cast<engine::Reg>(RSQ));
    eng->square_mul(static_cast<engine::Reg>(RSQ));
    eng->square_mul(static_cast<engine::Reg>(RSQ));

    uint64_t total_iters = B3u;
    uint64_t B = std::max<uint64_t>(1, (uint64_t)std::sqrt((double)std::max<uint64_t>(1, total_iters)));
    double desiredIntervalSeconds = 600.0;
    uint64_t checkpasslevel_auto = (uint64_t)((1000.0 * desiredIntervalSeconds) / (double)B);
    if (checkpasslevel_auto == 0) {
        uint64_t tmpB = (uint64_t)std::sqrt((double)B);
        if (tmpB == 0) tmpB = 1;
        checkpasslevel_auto = (total_iters / B) / tmpB;
    }
    uint64_t checkpasslevel = (options.checklevel > 0)
        ? options.checklevel
        : checkpasslevel_auto;
    if (checkpasslevel == 0) checkpasslevel = 1;

    uint64_t steps_in_block = 0;
    uint64_t blocks_since_last_check = 0;
    bool errordone = false;
   // uint64_t originalB3 = options.B3;

    for (; b < B3u; ++b) {
        if (interrupted) {
            double et = duration<double>(high_resolution_clock::now() - t0).count() + restored_time;
            save_ckpt_s3(eng, b, et);
            delete eng;
            std::cout << "\nInterrupted by user, Stage 3 state saved at b=" << b << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "Interrupted, Stage 3 saved at b=" << b; guiServer_->appendLog(oss.str()); }
            return 0;
        }

        if (steps_in_block == 0) {
            eng->copy((engine::Reg)RSTART_SQ, (engine::Reg)RSQ);
        }

        // RSQ := RSQ^2  (RSQ = H^{2^b} -> H^{2^{b+1}})
        eng->square_mul(static_cast<engine::Reg>(RSQ));

        if (options.erroriter > 0 &&
            (b + 1) == options.erroriter &&
            !errordone)
        {
            errordone = true;
            eng->sub(static_cast<engine::Reg>(RSQ), 33);
            std::cout << "Injected error at Stage 3 iteration " << (b + 1) << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Injected error at Stage 3 iteration " << (b + 1);
                guiServer_->appendLog(oss.str());
            }
        }

        // Init H^k et H^{-k}
        //eng->copy(static_cast<engine::Reg>(RHK),  static_cast<engine::Reg>(RSTATE)); // H^1
        //eng->copy(static_cast<engine::Reg>(RHIK), static_cast<engine::Reg>(RHINV));  // H^{-1}
        
        
        //for (uint64_t k = 1; k <= Kmax; ++k) {
            // (RSQ * H^{-k} - 1)

        for (uint64_t k = 1; k <= Kmax; ++k) {
            engine::Reg regHplus, regHminus;
            if (k == 1) {
                regHplus  = (engine::Reg)RHK;
                regHminus = (engine::Reg)RHINV;
            } else if (k == 2) {
                regHplus  = (engine::Reg)RHK_2;
                regHminus = (engine::Reg)RHINV_2;
            } else { // k == 3
                regHplus  = (engine::Reg)RHK_3;
                regHminus = (engine::Reg)RHINV_3;
            }

            // (RSQ * H^{-k} - 1)
            eng->copy((engine::Reg)RMINUS, (engine::Reg)RSQ);
            eng->mul ((engine::Reg)RMINUS, regHminus);
            eng->sub ((engine::Reg)RMINUS, 1);
            eng->set_multiplicand((engine::Reg)RPOW, (engine::Reg)RMINUS);
            eng->mul ((engine::Reg)RACC, (engine::Reg)RPOW);

            // (RSQ * H^{+k} - 1)
            eng->copy((engine::Reg)RPLUS, (engine::Reg)RSQ);
            eng->mul ((engine::Reg)RPLUS, regHplus);
            eng->sub ((engine::Reg)RPLUS, 1);
            eng->set_multiplicand((engine::Reg)RPOW, (engine::Reg)RPLUS);
            eng->mul ((engine::Reg)RACC, (engine::Reg)RPOW);
        }

        

/*
            // (RSQ * H^{+k} - 1)
            eng->copy(static_cast<engine::Reg>(RPLUS), static_cast<engine::Reg>(RSQ));
            eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RHK));
            eng->mul (static_cast<engine::Reg>(RPLUS), static_cast<engine::Reg>(RPOW));
            eng->sub (static_cast<engine::Reg>(RPLUS), 1);
            eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RPLUS));
            eng->mul (static_cast<engine::Reg>(RACC),  static_cast<engine::Reg>(RPOW));

            // k <- k+1 : H^k *= H ; H^{-k} *= H^{-1}
            eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RSTATE));
            eng->mul (static_cast<engine::Reg>(RHK),  static_cast<engine::Reg>(RPOW));
            eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RHINV));
            eng->mul (static_cast<engine::Reg>(RHIK), static_cast<engine::Reg>(RPOW));*/
        //}

        steps_in_block++;

        bool end_block = (steps_in_block == B) || (b + 1 == B3u);
        if (end_block) {
            blocks_since_last_check++;

            bool doCheck = options.gerbiczli &&
                           (blocks_since_last_check >= checkpasslevel || (b + 1 == B3u));
            if (doCheck) {
                std::cout << "[Gerbicz Li] Stage 3 check start\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "[Gerbicz Li] Stage 3 check start";
                    guiServer_->appendLog(oss.str());
                }

                eng->copy((engine::Reg)RCHK_SQ, (engine::Reg)RSTART_SQ);
                for (uint64_t k = 0; k < steps_in_block; ++k) {
                    eng->square_mul((engine::Reg)RCHK_SQ);
                }

                mpz_t z0, z1;
                mpz_inits(z0, z1, nullptr);
                eng->get_mpz(z0, (engine::Reg)RCHK_SQ);
                eng->get_mpz(z1, (engine::Reg)RSQ);
                bool ok = (mpz_cmp(z0, z1) == 0);
                mpz_clears(z0, z1, nullptr);

                if (!ok) {
                    std::cout << "[Gerbicz Li] Stage 3 mismatch : last correct RSQ will be restored\n";
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss << "[Gerbicz Li] Stage 3 mismatch : last correct RSQ will be restored";
                        guiServer_->appendLog(oss.str());
                    }
                    options.gerbicz_error_count += 1;
                    eng->copy((engine::Reg)RSQ, (engine::Reg)RSTART_SQ);
                    eng->set(static_cast<engine::Reg>(RACC), 1);
                } else {
                    std::cout << "[Gerbicz Li] Stage 3 check passed\n";
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss << "[Gerbicz Li] Stage 3 check passed";
                        guiServer_->appendLog(oss.str());
                    }

                    mpz_class X_acc = compute_X_with_dots(eng, static_cast<engine::Reg>(RACC), Mp);
                    mpz_class g_mid = gcd_with_dots(X_acc, Mp);
                    bool found_mid = g_mid != 1 && g_mid != Mp;
                    if (found_mid) {
                        char* s = mpz_get_str(nullptr, 10, g_mid.get_mpz_t());
                        std::string fname_mid = "stage3_result_B3_" + std::to_string(b + 1) + "_p_" + std::to_string(options.exponent) + ".txt";
                        writeStageResult(fname_mid, "B3=" + std::to_string(b + 1) + "  factor=" + std::string(s));
                        std::cout << "\n>>>  Factor P-1 (stage 3, intermediate) found : " << s << '\n';
                        if (guiServer_) {
                            std::ostringstream oss;
                            oss << "\n>>>  Factor P-1 (stage 3, intermediate) found : " << s << "\n";
                            guiServer_->appendLog(oss.str());
                        }
                        options.knownFactors.push_back(std::string(s));
                        midFactor = g_mid;
                        foundIntermediate = true;
                        uint64_t tmpB3 = options.B3;
                        options.B3 = b + 1;
                        std::string json_mid = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
                        std::cout << "Manual submission JSON (stage 3 intermediate):\n" << json_mid << "\n";
                        io::WorktodoManager wm_mid(options);
                        wm_mid.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage3_intermediate", json_mid);
                        wm_mid.appendToResultsTxt(json_mid);
                        options.B3 = tmpB3;
                        std::free(s);
                    }

                    eng->set(static_cast<engine::Reg>(RACC), 1);
                    blocks_since_last_check = 0;
                }
            }

            steps_in_block = 0;
        }

        auto now = high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastDisplay).count() >= 3) {
            double percent = B3u ? (100.0 * double(b + 1) / double(B3u)) : 100.0;
            double elapsedSec= std::chrono::duration<double>(now - t0).count() + restored_time;
            double etaSec = (b + 1) ? elapsedSec * (double(B3u - (b + 1)) / double(b + 1)) : 0.0;
            int days = int(etaSec) / 86400;
            int hours = (int(etaSec) % 86400) / 3600;
            int minutes = (int(etaSec) % 3600) / 60;
            int seconds = int(etaSec) % 60;
            std::cout << "Stage3: " << std::fixed << std::setprecision(2) << percent
                      << "% | b=" << (b + 1) << "/" << B3u
                      << " | K=" << Kmax
                      << " | Elapsed " << elapsedSec << "s | ETA "
                      << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r" << std::endl;
            if (guiServer_) {
                std::ostringstream oss; oss << "Stage3: " << std::fixed << std::setprecision(2) << percent
                                            << "% | b=" << (b + 1) << "/" << B3u
                                            << " | K=" << Kmax;
                guiServer_->appendLog(oss.str());
                guiServer_->setProgress(double(b + 1), double(B3u), "Stage 3");
            }
            lastDisplay = now;
        }
        if (now - lastBackup >= std::chrono::seconds(options.backup_interval)) {
            double et = duration<double>(now - t0).count() + restored_time;
            std::cout << "\nBackup Stage 3 at b=" << (b + 1) << " start...\n";
            save_ckpt_s3(eng, b + 1, et);
            lastBackup = now;
            std::cout << "Backup Stage 3 done.\n";
            if (guiServer_) { std::ostringstream oss; oss << "Backup Stage 3 at b=" << (b + 1); guiServer_->appendLog(oss.str()); }
        }
    }

    auto t1 = high_resolution_clock::now();
    double elapsed = duration<double>(t1 - t0).count() + restored_time;

    mpz_class X  = compute_X_with_dots(eng, static_cast<engine::Reg>(RACC), Mp);
    mpz_class g  = gcd_with_dots(X, Mp);
    bool found_final = g != 1 && g != Mp;
    bool found = found_final || foundIntermediate;

    std::cout << "\nElapsed time (stage 3) = " << std::fixed << std::setprecision(2) << elapsed << " s." << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "Elapsed time (stage 3) = " << std::fixed << std::setprecision(2) << elapsed << " s."; guiServer_->appendLog(oss.str()); }

    std::string filename = "stage3_result_B3_" + std::to_string(B3u) + "_p_" + std::to_string(options.exponent) + ".txt";
    if (found_final) {
        char* s = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "B3=" + std::to_string(B3u) + "  factor=" + std::string(s));
        std::cout << "\n>>>  Factor P-1 (stage 3) found : " << s << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\n>>>  Factor P-1 (stage 3) found : " << s << "\n"; guiServer_->appendLog(oss.str()); }
        options.knownFactors.push_back(std::string(s));
        std::free(s);
    } else if (!foundIntermediate) {
        writeStageResult(filename, "No factor P-1 up to B3=" + std::to_string(B3u));
        std::cout << "\nNo factor P-1 (stage 3) until B3 = " << B3u << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\nNo factor P-1 (stage 3) until B3 = " << B3u << '\n'; guiServer_->appendLog(oss.str()); }
    } else {
        char* s = mpz_get_str(nullptr, 10, midFactor.get_mpz_t());
        writeStageResult(filename, "B3=" + std::to_string(B3u) + "  factor=" + std::string(s));
        std::cout << "\n>>>  Factor P-1 (stage 3) found (from intermediate) : " << s << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\n>>>  Factor P-1 (stage 3) found (from intermediate) : " << s << "\n"; guiServer_->appendLog(oss.str()); }
        std::free(s);
    }

    std::remove(ckpt_file_s3.c_str());
    std::remove((ckpt_file_s3 + ".old").c_str());
    std::remove((ckpt_file_s3 + ".new").c_str());
    options.computer_name = "STAGE3";
    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
    std::cout << "Manual submission JSON:\n" << json << "\n";
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage3", json);
    wm.appendToResultsTxt(json);

    delete eng;
    if (hasWorktodoEntry_) {
            if (worktodoParser_->removeFirstProcessed()) {
                std::cout << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n";
                if (guiServer_) { std::ostringstream oss; oss << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n"; guiServer_->appendLog(oss.str()); }
                std::ifstream f(options.worktodo_path);
                std::string l; bool more = false; while (std::getline(f, l)) { if (!l.empty() && l[0] != '#') { more = true; break; } }
                f.close();
                if (more) { std::cout << "Restarting for next entry in worktodo.txt\n"; if (guiServer_) { std::ostringstream oss; oss << "Restarting for next entry in worktodo.txt\n"; guiServer_->appendLog(oss.str()); } restart_self(argc_, argv_); }
                else { std::cout << "No more entries in worktodo.txt, exiting.\n"; if (guiServer_) { std::ostringstream oss; oss << "No more entries in worktodo.txt, exiting.\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(0);} }
            } else {
                std::cerr << "Failed to update " << options.worktodo_path << "\n"; if (guiServer_) { std::ostringstream oss; oss << "Failed to update " << options.worktodo_path << "\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(-1);}
            }
        }
    return found ? 0 : 1;
}

/*
2026 - Cherubrock
P-1 Stage 4 (benchmark / stress-test)
- Load an existing Stage-1 state H (from .save/.p95 or stage1 checkpoint)
- Build a random exponent E as a product of ~B3-digit integers until E reaches TARGET_BITS bits
- Compute H^E mod (2^p - 1) using the Marin GPU engine (square-and-multiply)

Re-uses existing option fields to avoid adding new CLI flags:
- B1 == 0 and (0 < B3 <= 128) => run this stage from runPM1Marin()
- B3 = decimal digits of random factors (e.g. 31)
- B4 = RNG seed (0 = time-based seed)
- B1old (or pm1_extend_save_path) must point to the precomputed S1 state to load.
- tbits = target exponent size in bits (default fallback if tbits==0)
*/
int App::runPM1Stage4Marin() {
    using namespace std::chrono;

    if (guiServer_) {
        std::ostringstream oss;
        oss << "P-1 stage 4 (benchmark)";
        guiServer_->setStatus(oss.str());
        guiServer_->appendLog(oss.str());
    }

    const uint32_t pexp = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;

    // ---- Parameters
    uint32_t digits = (options.B3 > 0) ? static_cast<uint32_t>(options.B3) : 31u; // decimal digits of each random factor
    if (digits < 2) digits = 2;
    if (digits > 128) {
        std::cerr << "Stage 4: B3 too large for \"digits\" (got " << digits << ", expected <= 128).\n";
        if (guiServer_) {
            std::ostringstream oss;
            oss << "Stage 4: invalid B3 digits=" << digits;
            guiServer_->appendLog(oss.str());
        }
        return -1;
    }

    uint64_t TARGET_BITS = (options.tbits > 0) ? static_cast<uint64_t>(options.tbits) : 500000ULL;

    uint64_t seed = static_cast<uint64_t>(options.B4);
    if (seed == 0) {
        seed = static_cast<uint64_t>(duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
    }

    std::cout << "Start P-1 Stage 4 (benchmark): p=" << pexp
              << " | digits=" << digits
              << " | seed=" << seed
              << " | targetBits=" << TARGET_BITS
              << "\n";

    // ---- Build E = Π random_odd(digits) until E reaches TARGET_BITS bits
    auto pow10 = [](uint32_t d)->mpz_class{
        mpz_class r = 1;
        for (uint32_t i = 0; i < d; ++i) r *= 10;
        return r;
    };

    mpz_class low  = pow10(digits - 1); // 10^(digits-1)
    mpz_class span = low * 9;           // 9*10^(digits-1)
    mpz_class E    = 1;

    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    {
        mpz_t zseed;
        mpz_init(zseed);
        mpz_set_ui(zseed, static_cast<unsigned long>(seed & 0xFFFFFFFFu));
        mpz_mul_2exp(zseed, zseed, 32);
        mpz_add_ui(zseed, zseed, static_cast<unsigned long>((seed >> 32) & 0xFFFFFFFFu));
        gmp_randseed(rs, zseed);
        mpz_clear(zseed);
    }

    uint64_t nFactors = 0;
    auto tE0 = high_resolution_clock::now();
    while (mpz_sizeinbase(E.get_mpz_t(), 2) < TARGET_BITS) {
        mpz_class r;
        mpz_urandomm(r.get_mpz_t(), rs, span.get_mpz_t()); // 0..span-1
        r += low;                                          // low..(low+span-1) -> digits decimal digits
        if (mpz_even_p(r.get_mpz_t())) r += 1;             // keep it odd
        E *= r;
        ++nFactors;
    }
    auto tE1 = high_resolution_clock::now();
    const uint64_t Ebits = static_cast<uint64_t>(mpz_sizeinbase(E.get_mpz_t(), 2));
    const double buildSec = duration<double>(tE1 - tE0).count();
    gmp_randclear(rs);

    std::cout << "Stage4 exponent built: bits=" << Ebits
              << " | factors=" << nFactors
              << " | build=" << std::fixed << std::setprecision(2) << buildSec << "s\n";

    // ---- Engine regs (reuse Stage-1 layout for convenience)
    const size_t RSTATE = 0;
    const size_t RACC_L = 1;
    const size_t RACC_R = 2;
    const size_t RCHK   = 3;
    const size_t RPOW   = 4;
    const size_t RTMP   = 5;
    const size_t RSTART = 6;
    const size_t RSAVE_S= 7;
    const size_t RSAVE_L= 8;
    const size_t RSAVE_R= 9;
    const size_t RBASE  = 10;
    static constexpr size_t regsS4 = 11;

    engine* eng = engine::create_gpu(pexp, regsS4, static_cast<size_t>(options.device_id), verbose);

    // ---- Load H (Stage-1 output) from resume file (.save/.p95) or from stage1 checkpoint (if still present)
    mpz_t Hs1;
    mpz_init(Hs1);
    bool loaded = false;
    {
        uint64_t B1_file = 0;
        uint32_t p_file  = 0;
        mpz_class X_s1;

        std::string basePath = options.pm1_extend_save_path;
        uint64_t B1_load = options.B1old;
        if (basePath.empty() && B1_load > 0) {
            basePath = "resume_p" + std::to_string(options.exponent) +
                       "_B1_" + std::to_string(B1_load);
        }

        if (!basePath.empty()) {
            std::string resumeSave = basePath;
            std::string resumeP95  = basePath;

            if (resumeSave.size() >= 5 && resumeSave.substr(resumeSave.size() - 5) == ".save") {
                resumeP95 = resumeSave.substr(0, resumeSave.size() - 5) + ".p95";
            } else if (resumeSave.size() >= 4 && resumeSave.substr(resumeSave.size() - 4) == ".p95") {
                resumeP95  = resumeSave;
                resumeSave = resumeSave.substr(0, resumeSave.size() - 4) + ".save";
            } else {
                resumeSave += ".save";
                resumeP95  += ".p95";
            }

            std::string usedPath;
            if (load_pm1_s1_from_save(resumeSave, B1_file, p_file, X_s1)) { usedPath = resumeSave; loaded = true; }
            else if (load_pm1_s1_from_p95(resumeP95, B1_file, p_file, X_s1)) { usedPath = resumeP95; loaded = true; }

            if (loaded) {
                if (p_file != pexp) {
                    std::cerr << "Stage 4: mismatch between S1 resume file (p=" << p_file
                              << ") and options (p=" << pexp << ")\n";
                    mpz_clear(Hs1);
                    delete eng;
                    return -2;
                }
                std::cout << "Stage 4: using PM1 S1 state from " << usedPath << "\n";
                if (guiServer_) { std::ostringstream oss; oss << "Stage 4: using PM1 S1 state from " << usedPath; guiServer_->appendLog(oss.str()); }
                mpz_set(Hs1, X_s1.get_mpz_t());
            }
        }

        // Fallback: stage1 checkpoint (only if it still exists)
        if (!loaded) {
            engine* eng_load = engine::create_gpu(pexp, 11, static_cast<size_t>(options.device_id), verbose);
            std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
            const std::string ckpt_file = ck.str();

            auto read_ckpt_s1 = [&](engine* e, const std::string& file)->int{
                File f(file);
                if (!f.exists()) return -1;
                int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
                if (version != 3) return -2;
                uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
                if (rp != pexp) return -2;
                uint32_t ri = 0; double et = 0.0;
                if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
                const size_t cksz = e->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!f.read(data.data(), cksz)) return -2;
                if (!e->set_checkpoint(data)) return -2;
                uint64_t tmp64;
                if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
                if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
                if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
                if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
                uint8_t inlot=0; if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
                uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
                if (eacc_len) { std::string skip; skip.resize(eacc_len); if (!f.read(skip.data(), eacc_len)) return -2; }
                uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
                if (wbits_len) { std::string skip; skip.resize(wbits_len); if (!f.read(skip.data(), wbits_len)) return -2; }
                uint64_t chunkIdx=0, startP=0; uint8_t first=0; uint64_t processedBits=0, bitsInChunk=0;
                if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
                if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
                if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
                if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
                if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
                if (!f.check_crc32()) return -2;
                return 0;
            };

            int rr = read_ckpt_s1(eng_load, ckpt_file);
            if (rr < 0) rr = read_ckpt_s1(eng_load, ckpt_file + ".old");
            if (rr == 0) {
                eng_load->get_mpz(Hs1, static_cast<engine::Reg>(0 /*RSTATE*/));
                loaded = true;
                std::cout << "Stage 4: using PM1 S1 state from " << ckpt_file << "\n";
                if (guiServer_) { std::ostringstream oss; oss << "Stage 4: using PM1 S1 state from " << ckpt_file; guiServer_->appendLog(oss.str()); }
            }
            delete eng_load;
        }
    }

    if (!loaded) {
        std::cerr << "Stage 4: cannot load PM1 S1 state. Provide -pm1_extend_save_path or set B1old.\n";
        if (guiServer_) guiServer_->appendLog("Stage 4: cannot load PM1 S1 state. Provide -pm1_extend_save_path or set B1old.");
        mpz_clear(Hs1);
        delete eng;
        return -2;
    }

    // Base = H
    eng->set_mpz(static_cast<engine::Reg>(RBASE), Hs1);
    mpz_clear(Hs1);

    // Result = 1
    eng->set(static_cast<engine::Reg>(RSTATE), 1);

    // ---- Stage4 checkpointing (engine state + bit index)
    std::ostringstream ck4; ck4 << "pm1_s4_m_" << pexp << ".ckpt";
    const std::string ckpt_file_s4 = ck4.str();

    auto read_ckpt_s4 = [&](engine* e, const std::string& file, uint64_t& saved_i, double& et)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 1) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != pexp) return -2;
        uint32_t dig = 0; if (!f.read(reinterpret_cast<char*>(&dig), sizeof(dig))) return -2;
        uint64_t s   = 0; if (!f.read(reinterpret_cast<char*>(&s), sizeof(s))) return -2;
        uint64_t tb  = 0; if (!f.read(reinterpret_cast<char*>(&tb), sizeof(tb))) return -2;
        if (dig != digits || s != seed || tb != TARGET_BITS) return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_i), sizeof(saved_i))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = e->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!e->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };

    auto save_ckpt_s4 = [&](engine* e, uint64_t cur_i, double et){
        const std::string oldf = ckpt_file_s4 + ".old", newf = ckpt_file_s4 + ".new";
        {
            File f(newf, "wb");
            int version = 1;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&pexp), sizeof(pexp))) return;
            if (!f.write(reinterpret_cast<const char*>(&digits), sizeof(digits))) return;
            if (!f.write(reinterpret_cast<const char*>(&seed), sizeof(seed))) return;
            uint64_t tb = TARGET_BITS;
            if (!f.write(reinterpret_cast<const char*>(&tb), sizeof(tb))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_i), sizeof(cur_i))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;
            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!e->get_checkpoint(data)) return;
            if (!f.write(data.data(), cksz)) return;
            f.write_crc32();
        }
        std::remove(oldf.c_str());
        struct stat s;
        if ((stat(ckpt_file_s4.c_str(), &s) == 0) && (std::rename(ckpt_file_s4.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file_s4.c_str());
    };

    bool resumed = false;
    uint64_t resume_i = 0;
    double restored_time = 0.0;
    {
        int rs = read_ckpt_s4(eng, ckpt_file_s4, resume_i, restored_time);
        if (rs == 0) {
            resumed = true;
            std::cout << "Resuming Stage 4 at i=" << resume_i << "/" << Ebits << "\n";
            if (guiServer_) { std::ostringstream oss; oss << "Resuming Stage 4 at i=" << resume_i << "/" << Ebits; guiServer_->appendLog(oss.str()); }
        }
    }

    // ---- Loop parameters
    const uint64_t bits = Ebits;
    uint64_t B = std::max<uint64_t>(1, static_cast<uint64_t>(std::sqrt(static_cast<double>(std::max<uint64_t>(1, bits)))));
    double desiredIntervalSeconds = 600.0;
    uint64_t checkpasslevel_auto = static_cast<uint64_t>((1000.0 * desiredIntervalSeconds) / static_cast<double>(B));
    if (checkpasslevel_auto == 0) {
        uint64_t tmpB = static_cast<uint64_t>(std::sqrt(static_cast<double>(B)));
        if (tmpB == 0) tmpB = 1;
        checkpasslevel_auto = (bits / B) / tmpB;
    }
    uint64_t checkpass = (options.checklevel > 0) ? options.checklevel : checkpasslevel_auto;
    if (checkpass == 0) checkpass = 1;

    uint64_t resumeI = resumed ? resume_i : bits;

    // Gerbicz-Li state (same logic as Stage-1, but no fast3 path)
    uint64_t blocks_since_check = 0;
    uint64_t bits_in_block = 0;
    uint64_t current_block_len = ((resumeI - 1) % B) + 1;
    mpz_class eacc = 0;
    mpz_class wbits = 0;
    uint64_t gl_checkpass = 0;
    bool in_lot = false;

    bool errordone = false;

    auto t0 = high_resolution_clock::now();
    auto lastBackup  = t0;
    auto lastDisplay = t0;

    for (uint64_t i = resumeI; i > 0; --i) {
        if (interrupted) {
            double et = duration<double>(high_resolution_clock::now() - t0).count() + restored_time;
            save_ckpt_s4(eng, i, et);
            delete eng;
            std::cout << "\nInterrupted by user, Stage 4 state saved at i=" << i << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "Interrupted, Stage 4 saved at i=" << i; guiServer_->appendLog(oss.str()); }
            return 0;
        }

        auto now = high_resolution_clock::now();
        if (now - lastBackup >= std::chrono::seconds(options.backup_interval)) {
            double et = duration<double>(now - t0).count() + restored_time;
            std::cout << "\nBackup Stage 4 at i=" << i << " start...\n";
            save_ckpt_s4(eng, i, et);
            lastBackup = now;
            std::cout << "Backup Stage 4 done.\n";
            if (guiServer_) { std::ostringstream oss; oss << "Backup Stage 4 at i=" << i; guiServer_->appendLog(oss.str()); }
        }

        if (bits_in_block == 0) {
            current_block_len = ((i - 1) % B) + 1;
            if (current_block_len == B) {
                if (gl_checkpass == 0 && blocks_since_check == 0 && wbits == 0 && eacc == 0) {
                    eng->set(static_cast<engine::Reg>(RACC_L), 1);
                    eng->set(static_cast<engine::Reg>(RACC_R), 1);
                    eng->copy(static_cast<engine::Reg>(RSAVE_S), static_cast<engine::Reg>(RSTATE));
                    eng->set(static_cast<engine::Reg>(RSAVE_L), 1);
                    eng->set(static_cast<engine::Reg>(RSAVE_R), 1);
                    eacc = 0;
                    blocks_since_check = 0;
                    wbits = 0;
                    in_lot = true;
                }
            } else {
                in_lot = false;
                gl_checkpass = 0;
                eacc = 0;
                blocks_since_check = 0;
                wbits = 0;
            }
            eng->copy(static_cast<engine::Reg>(RSTART), static_cast<engine::Reg>(RSTATE));
        }

        int b = mpz_tstbit(E.get_mpz_t(), static_cast<mp_bitcnt_t>(i - 1)) ? 1 : 0;

        // state = state^2 ; if (b) state *= H
        eng->square_mul(static_cast<engine::Reg>(RSTATE));
        if (b) {
            eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RBASE));
            eng->mul(static_cast<engine::Reg>(RSTATE), static_cast<engine::Reg>(RTMP));
        }

        wbits <<= 1;
        if (b) wbits += 1;
        bits_in_block += 1;

        if (options.erroriter > 0 && (resumeI - i + 1) == (uint64_t)options.erroriter && !errordone) {
            errordone = true;
            eng->sub(static_cast<engine::Reg>(RSTATE), 33);
            std::cout << "Injected error at Stage 4 iteration " << (resumeI - i + 1) << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "Injected error at Stage 4 iteration " << (resumeI - i + 1); guiServer_->appendLog(oss.str()); }
        }

        bool end_block = (bits_in_block == current_block_len);
        if (end_block) {
            if (current_block_len == B) {
                eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RSTART));
                eng->mul(static_cast<engine::Reg>(RACC_L), static_cast<engine::Reg>(RTMP));
                eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RSTATE));
                eng->mul(static_cast<engine::Reg>(RACC_R), static_cast<engine::Reg>(RTMP));
                eacc += wbits;
                blocks_since_check += 1;
                gl_checkpass += 1;

                bool doCheck = options.gerbiczli && in_lot && (gl_checkpass == checkpass || i == 1);
                if (doCheck) {
                    std::cout << "[Gerbicz Li] Stage 4 check start\n";
                    if (guiServer_) guiServer_->appendLog("[Gerbicz Li] Stage 4 check start");

                    eng->copy(static_cast<engine::Reg>(RCHK), static_cast<engine::Reg>(RACC_L));
                    for (uint64_t k = 0; k < B; ++k) eng->square_mul(static_cast<engine::Reg>(RCHK));

                    eng->set(static_cast<engine::Reg>(RPOW), 1);
                    size_t eb = mpz_sizeinbase(eacc.get_mpz_t(), 2);
                    for (size_t k = eb; k-- > 0;) {
                        eng->square_mul(static_cast<engine::Reg>(RPOW));
                        if (mpz_tstbit(eacc.get_mpz_t(), k)) {
                            eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RBASE));
                            eng->mul(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RTMP));
                        }
                    }

                    eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RPOW));
                    eng->mul(static_cast<engine::Reg>(RCHK), static_cast<engine::Reg>(RTMP));

                    mpz_t z0, z1; mpz_inits(z0, z1, nullptr);
                    eng->get_mpz(z0, static_cast<engine::Reg>(RCHK));
                    eng->get_mpz(z1, static_cast<engine::Reg>(RACC_R));
                    bool ok = (mpz_cmp(z0, z1) == 0);
                    mpz_clears(z0, z1, nullptr);

                    if (!ok) {
                        std::cout << "[Gerbicz Li] Stage 4 mismatch: restoring last correct state\n";
                        if (guiServer_) guiServer_->appendLog("[Gerbicz Li] Stage 4 mismatch: restoring last correct state");
                        options.gerbicz_error_count += 1;
                        eng->copy(static_cast<engine::Reg>(RSTATE), static_cast<engine::Reg>(RSAVE_S));
                        eng->set(static_cast<engine::Reg>(RACC_L), 1);
                        eng->set(static_cast<engine::Reg>(RACC_R), 1);
                        eng->copy(static_cast<engine::Reg>(RSTART), static_cast<engine::Reg>(RSTATE));
                        i = i + blocks_since_check * B;
                        eacc = 0;
                        blocks_since_check = 0;
                        wbits = 0;
                        gl_checkpass = 0;
                        bits_in_block = 0;
                        continue;
                    } else {
                        std::cout << "[Gerbicz Li] Stage 4 check passed\n";
                        if (guiServer_) guiServer_->appendLog("[Gerbicz Li] Stage 4 check passed");
                        eng->copy(static_cast<engine::Reg>(RSAVE_S), static_cast<engine::Reg>(RSTATE));
                        eng->set(static_cast<engine::Reg>(RACC_L), 1);
                        eng->set(static_cast<engine::Reg>(RACC_R), 1);
                        eacc = 0;
                        blocks_since_check = 0;
                        gl_checkpass = 0;
                    }
                }
            } else {
                if (options.gerbiczli) {
                    eng->copy(static_cast<engine::Reg>(RCHK), static_cast<engine::Reg>(RSTART));
                    for (uint64_t k = 0; k < current_block_len; ++k) eng->square_mul(static_cast<engine::Reg>(RCHK));

                    eng->set(static_cast<engine::Reg>(RPOW), 1);
                    size_t wb = mpz_sizeinbase(wbits.get_mpz_t(), 2);
                    for (size_t k = wb; k-- > 0;) {
                        eng->square_mul(static_cast<engine::Reg>(RPOW));
                        if (mpz_tstbit(wbits.get_mpz_t(), k)) {
                            eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RBASE));
                            eng->mul(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RTMP));
                        }
                    }

                    eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RPOW));
                    eng->mul(static_cast<engine::Reg>(RCHK), static_cast<engine::Reg>(RTMP));

                    mpz_t z0, z1; mpz_inits(z0, z1, nullptr);
                    eng->get_mpz(z0, static_cast<engine::Reg>(RCHK));
                    eng->get_mpz(z1, static_cast<engine::Reg>(RSTATE));
                    bool ok0 = (mpz_cmp(z0, z1) == 0);
                    mpz_clears(z0, z1, nullptr);

                    if (!ok0) {
                        std::cout << "[Gerbicz Li] Stage 4 mismatch: restoring last block start\n";
                        if (guiServer_) guiServer_->appendLog("[Gerbicz Li] Stage 4 mismatch: restoring last block start");
                        options.gerbicz_error_count += 1;
                        eng->copy(static_cast<engine::Reg>(RSTATE), static_cast<engine::Reg>(RSTART));
                        i = i + current_block_len;
                        wbits = 0;
                        bits_in_block = 0;
                        continue;
                    } else {
                        std::cout << "[Gerbicz Li] Stage 4 check passed\n";
                        if (guiServer_) guiServer_->appendLog("[Gerbicz Li] Stage 4 check passed");
                        eng->copy(static_cast<engine::Reg>(RSAVE_S), static_cast<engine::Reg>(RSTATE));
                        eng->set(static_cast<engine::Reg>(RACC_L), 1);
                        eng->set(static_cast<engine::Reg>(RACC_R), 1);
                        eacc = 0;
                        blocks_since_check = 0;
                        gl_checkpass = 0;
                    }
                }
            }
            bits_in_block = 0;
            wbits = 0;
        }

        auto now2 = high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now2 - lastDisplay).count() >= 3) {
            double done = double(bits - i + 1);
            double percent = bits ? (100.0 * done / double(bits)) : 100.0;
            double elapsedSec = duration<double>(now2 - t0).count() + restored_time;
            double etaSec = (done > 0.0) ? elapsedSec * (double(bits) - done) / done : 0.0;
            int days = int(etaSec) / 86400;
            int hours = (int(etaSec) % 86400) / 3600;
            int minutes = (int(etaSec) % 3600) / 60;
            int seconds = int(etaSec) % 60;

            std::cout << "Stage4: " << std::fixed << std::setprecision(2) << percent
                      << "% | i=" << (bits - i + 1) << "/" << bits
                      << " | Elapsed " << elapsedSec << "s | ETA "
                      << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                      << std::flush;

            if (guiServer_) {
                std::ostringstream oss;
                oss << "Stage4: " << std::fixed << std::setprecision(2) << percent
                    << "% | " << (bits - i + 1) << "/" << bits;
                guiServer_->setProgress(done, double(bits), "Stage 4");
                guiServer_->appendLog(oss.str());
            }

            lastDisplay = now2;
        }
    }

    std::cout << "\n";

    auto t1 = high_resolution_clock::now();
    double elapsed = duration<double>(t1 - t0).count() + restored_time;

    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    mpz_class X  = compute_X_with_dots(eng, static_cast<engine::Reg>(RSTATE), Mp);
    X -= 1;
    mpz_class g  = gcd_with_dots(X, Mp);
    bool found = (g != 1) && (g != Mp);

    std::cout << "Elapsed time (stage 4) = " << std::fixed << std::setprecision(2) << elapsed << " s.\n";
    if (guiServer_) { std::ostringstream oss; oss << "Elapsed time (stage 4) = " << std::fixed << std::setprecision(2) << elapsed << " s."; guiServer_->appendLog(oss.str()); }

    std::string filename = "stage4_result_bits_" + std::to_string(TARGET_BITS) +
                           "_digits_" + std::to_string(digits) +
                           "_seed_" + std::to_string(seed) +
                           "_p_" + std::to_string(options.exponent) + ".txt";

    if (found) {
        char* s = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "stage4: bits=" + std::to_string(TARGET_BITS) +
                                    " digits=" + std::to_string(digits) +
                                    " seed=" + std::to_string(seed) +
                                    " factor=" + std::string(s));
        std::cout << "\n>>>  Factor (stage 4) found : " << s << '\n';
        if (guiServer_) { std::ostringstream oss; oss << ">>>  Factor (stage 4) found : " << s; guiServer_->appendLog(oss.str()); }
        options.knownFactors.push_back(std::string(s));
        std::free(s);
    } else {
        writeStageResult(filename, "No factor (stage 4) for bits=" + std::to_string(TARGET_BITS) +
                                    " digits=" + std::to_string(digits) +
                                    " seed=" + std::to_string(seed));
        std::cout << "No factor (stage 4).\n";
        if (guiServer_) guiServer_->appendLog("No factor (stage 4).");
    }

    std::remove(ckpt_file_s4.c_str());
    std::remove((ckpt_file_s4 + ".old").c_str());
    std::remove((ckpt_file_s4 + ".new").c_str());

    // ---- JSON output / append : same pattern as Stage 3
    options.computer_name = "STAGE4";
    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
    std::cout << "Manual submission JSON:\n" << json << "\n";
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage4", json);
    wm.appendToResultsTxt(json);

    delete eng;

    // ---- worktodo handling: same as Stage 3
    if (hasWorktodoEntry_) {
        if (worktodoParser_->removeFirstProcessed()) {
            std::cout << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n";
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n";
                guiServer_->appendLog(oss.str());
            }

            std::ifstream f(options.worktodo_path);
            std::string l;
            bool more = false;
            while (std::getline(f, l)) {
                if (!l.empty() && l[0] != '#') { more = true; break; }
            }
            f.close();

            if (more) {
                std::cout << "Restarting for next entry in worktodo.txt\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Restarting for next entry in worktodo.txt\n";
                    guiServer_->appendLog(oss.str());
                }
                restart_self(argc_, argv_);
            } else {
                std::cout << "No more entries in worktodo.txt, exiting.\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "No more entries in worktodo.txt, exiting.\n";
                    guiServer_->appendLog(oss.str());
                }
                if (!options.gui) { std::exit(0); }
            }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n";
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Failed to update " << options.worktodo_path << "\n";
                guiServer_->appendLog(oss.str());
            }
            if (!options.gui) { std::exit(-1); }
        }
    }

    return found ? 0 : 1;
}
