#ifndef CORE_ECM_PRIME95_HANDOFF_HPP
#define CORE_ECM_PRIME95_HANDOFF_HPP

// Helpers for the twisted-Edwards ECM Prime95 stage-2 handoff.
//
// A curve handed to Prime95 has finished stage 1 on the GPU; from then on its
// only state is the per-curve resume file (resume_p<p>_ECM_TE_B1_<B1>_c<n>.p95).
// A small "pending" marker next to that file records that the curve still owes
// its stage 2.  The marker is written before the stage-1 checkpoint is dropped
// and removed only once Prime95 has reported a result for the curve, so an
// interrupt, a Prime95 failure or a crash cannot lose the stage 2: the next run
// finds the marker and hands the curve to Prime95 again.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <csignal>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace core {

/// Outcome of one Prime95 results.json.txt line for an ECM stage 2.
struct EcmPrime95Outcome {
    bool ok = false;            // Prime95 searched the range (status NF or F with a factor).
    bool factor_found = false;  // A factor string is present.
    std::string error;          // Set when !ok.
};

/// "NF" means the range was searched without a factor.  "F" must come with a
/// factor: an "F" line whose factor cannot be read is an error, never "no
/// factor", otherwise the curve would be counted as searched while the factor
/// Prime95 found is dropped.
inline EcmPrime95Outcome ecmClassifyPrime95Result(const std::string& status, const std::string& factor) {
    EcmPrime95Outcome o;
    const bool digits = !factor.empty() &&
        std::all_of(factor.begin(), factor.end(), [](unsigned char ch) { return ch >= '0' && ch <= '9'; });
    if (status == "F") {
        if (!digits) {
            o.error = factor.empty()
                ? "Prime95 reported a factor (status F) but the result line has no factor"
                : "Prime95 reported a factor (status F) that cannot be parsed: " + factor;
            return o;
        }
        o.ok = true;
        o.factor_found = true;
        return o;
    }
    if (status == "NF") {
        o.ok = true;
        o.factor_found = !factor.empty();
        return o;
    }
    o.error = "Prime95 returned an unsupported status: " + status;
    return o;
}

/// One curve whose stage 2 has been handed to Prime95 and has no result yet.
struct EcmPrime95Pending {
    uint64_t p = 0;
    uint64_t b1 = 0;
    uint64_t curve_idx = 0;
    std::string resume_path;   // per-curve resume file, relative to the run's directory
    std::string sigma_hex;
    uint64_t curve_seed = 0;
    uint64_t base_seed = 0;
};

inline std::string ecmPrime95PendingMarkerPath(const std::string& resume_path) {
    return resume_path + ".pending";
}

/// Writes the marker atomically (temporary file, then rename).
inline bool ecmWritePrime95Pending(const EcmPrime95Pending& e) {
    const std::string path = ecmPrime95PendingMarkerPath(e.resume_path);
    const std::string tmp = path + ".new";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        if (!out) return false;
        out << "p=" << e.p << '\n'
            << "B1=" << e.b1 << '\n'
            << "curve=" << e.curve_idx << '\n'
            << "resume=" << e.resume_path << '\n'
            << "sigma_hex=" << e.sigma_hex << '\n'
            << "curve_seed=" << e.curve_seed << '\n'
            << "base_seed=" << e.base_seed << '\n';
        out.flush();
        if (!out) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

/// Strict unsigned decimal: digits only (no sign, no blanks), no overflow.
inline bool ecmParseU64(const std::string& s, uint64_t& out) {
    if (s.empty() || s.size() > 20) return false;
    uint64_t v = 0;
    for (const char c : s) {
        const auto ch = static_cast<unsigned char>(c);
        if (ch < '0' || ch > '9') return false;
        const uint64_t d = static_cast<uint64_t>(ch - '0');
        if (v > (UINT64_MAX - d) / 10) return false;
        v = v * 10 + d;
    }
    out = v;
    return true;
}

/// Largest curve index a marker may name (curve counts are 32-bit elsewhere).
constexpr uint64_t kEcmPrime95MaxCurveIdx = 0xFFFFFFFEull;

inline bool ecmReadPrime95Pending(const std::string& marker_path, EcmPrime95Pending& e) {
    std::ifstream in(marker_path);
    if (!in) return false;
    e = EcmPrime95Pending{};
    bool have_p = false, have_b1 = false, have_curve = false, have_resume = false;
    bool have_cs = false, have_bs = false;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq);
        const std::string v = line.substr(eq + 1);
        if (k == "p") { if (!ecmParseU64(v, e.p)) return false; have_p = true; }
        else if (k == "B1") { if (!ecmParseU64(v, e.b1)) return false; have_b1 = true; }
        else if (k == "curve") { if (!ecmParseU64(v, e.curve_idx)) return false; have_curve = true; }
        else if (k == "resume") { e.resume_path = v; have_resume = !v.empty(); }
        else if (k == "sigma_hex") e.sigma_hex = v;
        else if (k == "curve_seed") { if (!ecmParseU64(v, e.curve_seed)) return false; have_cs = true; }
        else if (k == "base_seed") { if (!ecmParseU64(v, e.base_seed)) return false; have_bs = true; }
    }
    if (in.bad()) return false;
    return have_p && have_b1 && have_curve && have_resume && have_cs && have_bs &&
           e.curve_idx <= kEcmPrime95MaxCurveIdx;
}

inline void ecmRemovePrime95Pending(const std::string& resume_path) {
    std::error_code ec;
    std::filesystem::remove(ecmPrime95PendingMarkerPath(resume_path), ec);
}

/// Every pending marker in `dir` for this exponent and B1, sorted by curve.
/// A marker that cannot be read is returned in `unreadable`.
inline std::vector<EcmPrime95Pending> ecmFindPrime95Pending(const std::filesystem::path& dir,
                                                            uint64_t p, uint64_t b1,
                                                            std::vector<std::string>* unreadable = nullptr) {
    std::vector<EcmPrime95Pending> out;
    const std::string prefix = "resume_p" + std::to_string(p) + "_ECM_TE_B1_" + std::to_string(b1) + "_c";
    const std::string suffix = ".p95.pending";
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() <= prefix.size() + suffix.size()) continue;
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        const std::string marker = (dir / name).string();
        // The name carries the 1-based curve number; the contents must agree
        // with it and name the resume file the marker sits next to.
        uint64_t curve_no = 0;
        const std::string num = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        const std::string resume_name = name.substr(0, name.size() - std::string(".pending").size());
        EcmPrime95Pending e;
        if (!ecmParseU64(num, curve_no) || !ecmReadPrime95Pending(marker, e) || e.p != p || e.b1 != b1 ||
            e.curve_idx + 1 != curve_no || std::filesystem::path(e.resume_path).filename().string() != resume_name) {
            if (unreadable) unreadable->push_back(marker);
            continue;
        }
        e.resume_path = (dir / resume_name).string();
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(),
              [](const EcmPrime95Pending& a, const EcmPrime95Pending& b) { return a.curve_idx < b.curve_idx; });
    return out;
}

#ifndef _WIN32
/// Runs `sh -lc <script>` and waits for it, polling every 100 ms.  Unlike
/// std::system(), the calling process keeps its SIGINT/SIGTERM handlers while
/// the child runs, so a Ctrl-C still reaches the process-wide interrupt flag.
/// When `should_stop()` turns true the child gets SIGTERM once (Prime95 saves
/// and exits) and SIGKILL after `grace_seconds`; the child is always reaped.
/// `script` should end in `exec <program> ...` so the signal reaches the program
/// rather than the shell.  `stopped` is true when the run ended while a stop
/// was requested.  Returns the raw wait status, or -1 if the shell cannot start.
inline int ecmRunShellInterruptible(const std::string& script,
                                    const std::function<bool()>& should_stop,
                                    bool& stopped,
                                    int grace_seconds = 30) {
    stopped = false;
    const char* argv[] = {"sh", "-lc", script.c_str(), nullptr};
    pid_t pid = 0;
    if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, const_cast<char* const*>(argv), environ) != 0) return -1;
    bool term_sent = false;
    bool killed = false;
    auto term_at = std::chrono::steady_clock::now();
    for (;;) {
        int status = 0;
        const pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            // A Ctrl-C at the terminal reaches Prime95 directly (same process
            // group) and it may exit before the flag is polled here.
            if (should_stop()) stopped = true;
            return status;
        }
        if (r < 0 && errno != EINTR) return -1;
        if (should_stop()) {
            stopped = true;
            if (!term_sent) {
                kill(pid, SIGTERM);
                term_sent = true;
                term_at = std::chrono::steady_clock::now();
            } else if (!killed &&
                       std::chrono::steady_clock::now() - term_at > std::chrono::seconds(grace_seconds)) {
                kill(pid, SIGKILL);
                killed = true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
#endif

} // namespace core

#endif
