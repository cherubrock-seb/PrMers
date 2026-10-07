#pragma once

// Completed-phase record for the Gaussian-Mersenne GMCHAIN pipeline.
//
// A chain runs P-1, optional ECM and the Proth/PRP test for each requested
// family.  Each phase checkpoints itself, but a restarted chain would begin
// again at the first phase of the first family and redo everything that had
// already finished.  This small file lists the phases (and families) that are
// complete.  The key ties it to the job (the raw worktodo line, or the chain
// parameters for a command-line chain) so a record from a different job is
// ignored.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace core::gm_chain_progress {

inline const char* magic() { return "PRMERS-GM-CHAIN 1"; }

// Tokens recorded for `key`: "<FAMILY> <phase>" for a finished phase and
// "<FAMILY> done <rc>" for a finished family.  Empty if the file is missing,
// damaged or belongs to another job.
inline std::vector<std::string> load(const std::filesystem::path& file, const std::string& key) {
    std::vector<std::string> tokens;
    std::ifstream in(file);
    std::string head, stored_key, line;
    if (!std::getline(in, head) || head != magic()) return tokens;
    if (!std::getline(in, stored_key) || stored_key != key) return tokens;
    while (std::getline(in, line)) {
        if (!line.empty()) tokens.push_back(line);
    }
    return tokens;
}

inline void save(const std::filesystem::path& file, const std::string& key,
                 const std::vector<std::string>& tokens) {
    const std::filesystem::path tmp = file.string() + ".new";
    {
        std::ofstream out(tmp, std::ios::trunc);
        out << magic() << '\n' << key << '\n';
        for (const auto& token : tokens) out << token << '\n';
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

inline std::string phase_token(const std::string& family, const std::string& phase) {
    return family + " " + phase;
}

inline std::string done_token(const std::string& family, int rc) {
    return family + " done " + std::to_string(rc);
}

class Progress {
public:
    Progress(std::filesystem::path file, std::string key)
        : file_(std::move(file)), key_(std::move(key)), tokens_(load(file_, key_)) {}

    bool has(const std::string& token) const {
        for (const auto& t : tokens_) if (t == token) return true;
        return false;
    }

    // Return code of a family recorded as finished.
    std::optional<int> family_rc(const std::string& family) const {
        const std::string prefix = family + " done ";
        for (const auto& t : tokens_) {
            if (t.compare(0, prefix.size(), prefix) != 0) continue;
            try { return std::stoi(t.substr(prefix.size())); } catch (...) { return std::nullopt; }
        }
        return std::nullopt;
    }

    void mark(const std::string& token) {
        if (has(token)) return;
        tokens_.push_back(token);
        save(file_, key_, tokens_);
    }

    void clear() {
        tokens_.clear();
        core::gm_chain_progress::clear(file_);
    }

    const std::filesystem::path& file() const { return file_; }

private:
    std::filesystem::path file_;
    std::string key_;
    std::vector<std::string> tokens_;
};

} // namespace core::gm_chain_progress
