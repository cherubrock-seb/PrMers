#pragma once

// Completed-curve counter for the Gaussian-Mersenne ECM drivers.
//
// The per-curve checkpoints are named by curve index, but a driver that is
// restarted would still begin at curve 0 and redo every finished curve.  This
// tiny file records how many curves are complete.  The key ties the counter to
// the job (driver, family, exponent, bounds and curve stream) so a counter from
// a different job is ignored.

#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace core::gm_ecm_progress {

inline const char* magic() { return "PRMERS-GM-ECM-CURVES 1"; }

// Number of completed curves recorded for `key`, or 0.
inline std::uint64_t load(const std::filesystem::path& file, const std::string& key) {
    std::ifstream in(file);
    std::string head, stored_key, done;
    if (!std::getline(in, head) || head != magic()) return 0;
    if (!std::getline(in, stored_key) || stored_key != key) return 0;
    if (!std::getline(in, done)) return 0;
    try { return std::stoull(done); } catch (...) { return 0; }
}

inline void save(const std::filesystem::path& file, const std::string& key, std::uint64_t done) {
    const std::filesystem::path tmp = file.string() + ".new";
    {
        std::ofstream out(tmp, std::ios::trunc);
        out << magic() << '\n' << key << '\n' << done << '\n';
        if (!out) return;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, file, ec);
}

inline void clear(const std::filesystem::path& file) {
    std::error_code ec;
    std::filesystem::remove(file, ec);
    std::filesystem::remove(file.string() + ".new", ec);
}

// Removes the counter when the driver returns because the job ended (factor
// found or all curves done).  It is kept when the run was interrupted or is
// unwinding an exception, so the next run can resume.
class Guard {
public:
    Guard(std::filesystem::path file, const std::atomic<bool>& interrupted)
        : file_(std::move(file)), interrupted_(interrupted),
          uncaught_(std::uncaught_exceptions()) {}
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    ~Guard() {
        if (interrupted_.load(std::memory_order_relaxed)) return;
        if (std::uncaught_exceptions() > uncaught_) return;
        clear(file_);
    }
private:
    std::filesystem::path file_;
    const std::atomic<bool>& interrupted_;
    int uncaught_;
};

} // namespace core::gm_ecm_progress
