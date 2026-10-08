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

// A progress file is a few short lines; anything larger is not ours.
inline constexpr std::uintmax_t max_file_bytes() { return 1u << 20; }

inline bool valid_family(const std::string& family) { return family == "GM" || family == "GQ"; }

// The only tokens this code ever writes: "<FAMILY> pm1", "<FAMILY> ecm" for a
// finished phase and "<FAMILY> done <rc>" for a finished family.  The chain
// records a family only when it was not interrupted and the pipeline returned
// 0 or 1; rc 2 (error) is never persisted, so rc is exactly one of "0" and "1".
// The match is exact: no sign, no leading zeros, no whitespace, no trailing
// characters.
inline bool valid_token(const std::string& t) {
    if (t.size() < 4 || t[2] != ' ' || !valid_family(t.substr(0, 2))) return false;
    const std::string rest = t.substr(3);
    return rest == "pm1" || rest == "ecm" || rest == "done 0" || rest == "done 1";
}

// Return code recorded by `token` if it is a well-formed finished-family token
// for `family`, otherwise nullopt.
inline std::optional<int> parse_family_rc(const std::string& token, const std::string& family) {
    if (!valid_family(family) || !valid_token(token)) return std::nullopt;
    const std::string prefix = family + " done ";
    if (token.size() != prefix.size() + 1 || token.compare(0, prefix.size(), prefix) != 0)
        return std::nullopt;
    return token.back() - '0';
}

// Tokens recorded for `key`.  Empty if the file is missing, damaged or belongs
// to another job.  A file with any line that is not a well-formed token, an
// unterminated (possibly truncated) last line, or two different results for
// one family is treated as damaged as a whole, so the chain redoes the work
// rather than trusting a partly corrupt record.  Duplicate identical lines are
// collapsed.
inline std::vector<std::string> load(const std::filesystem::path& file, const std::string& key) {
    std::vector<std::string> tokens;
    std::error_code ec;
    const auto size = std::filesystem::file_size(file, ec);
    if (ec || size > max_file_bytes()) return tokens;
    std::ifstream in(file, std::ios::binary);
    std::string head, stored_key, line;
    if (!std::getline(in, head) || in.eof() || head != magic()) return tokens;
    if (!std::getline(in, stored_key) || in.eof() || stored_key != key) return tokens;
    std::string done_gm, done_gq;
    while (std::getline(in, line)) {
        if (in.eof() || !valid_token(line)) return {};   // truncated or malformed
        if (line.compare(3, 4, "done") == 0) {
            std::string& seen = line[1] == 'M' ? done_gm : done_gq;
            if (!seen.empty() && seen != line) return {};  // contradictory results
            seen = line;
        }
        bool dup = false;
        for (const auto& t : tokens) if (t == line) { dup = true; break; }
        if (!dup) tokens.push_back(line);
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

    // Return code of a family recorded as finished (0 or 1), if any.
    std::optional<int> family_rc(const std::string& family) const {
        for (const auto& t : tokens_) {
            if (const auto rc = parse_family_rc(t, family)) return rc;
        }
        return std::nullopt;
    }

    // Records a well-formed token; anything else (for example a done token for
    // rc 2) is ignored, as is a second result for an already finished family.
    void mark(const std::string& token) {
        if (!valid_token(token) || has(token)) return;
        if (token.compare(3, 4, "done") == 0 && family_rc(token.substr(0, 2))) return;
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
