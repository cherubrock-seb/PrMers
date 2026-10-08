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
#include <cstddef>
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

// Why a progress record yielded no tokens.  Missing is the normal first run and
// DifferentJob is a leftover from another assignment; Damaged means the file
// exists for this job but cannot be trusted.
enum class Status { Ok, Missing, DifferentJob, Damaged };

struct LoadResult {
    std::vector<std::string> tokens;
    Status status = Status::Missing;
    std::string reason;   // short human-readable cause; empty for Ok and Missing
};

namespace reason {
inline const char* not_regular_file() { return "not a regular file"; }
inline const char* unreadable() { return "cannot be read"; }
inline const char* too_large() { return "larger than 1 MiB"; }
inline const char* incomplete_header() { return "incomplete header"; }
inline const char* bad_header() { return "unrecognized header"; }
inline const char* incomplete_key() { return "truncated before the job key"; }
inline const char* crlf_key() { return "line ends with a carriage return"; }
inline const char* other_job() { return "record belongs to a different assignment"; }
} // namespace reason

inline LoadResult damaged(std::string why) {
    LoadResult r;
    r.status = Status::Damaged;
    r.reason = std::move(why);
    return r;
}

// Tokens recorded for `key`, with the reason when there are none.  A file with
// any line that is not a well-formed token, an unterminated (possibly
// truncated) last line, or two different results for one family is treated as
// damaged as a whole, so the chain redoes the work rather than trusting a
// partly corrupt record.  Duplicate identical lines are collapsed.  Nothing is
// printed here; the caller decides how to tell the user.
inline LoadResult load_checked(const std::filesystem::path& file, const std::string& key) {
    std::error_code ec;
    const auto type = std::filesystem::symlink_status(file, ec).type();
    if (type == std::filesystem::file_type::not_found) return {};   // Missing, no reason
    const auto st = std::filesystem::status(file, ec);
    if (ec && st.type() == std::filesystem::file_type::not_found) return {};  // dangling link
    if (st.type() != std::filesystem::file_type::regular) return damaged(reason::not_regular_file());
    const auto size = std::filesystem::file_size(file, ec);
    if (ec) return damaged(reason::unreadable());
    if (size > max_file_bytes()) return damaged(reason::too_large());
    std::ifstream in(file, std::ios::binary);
    if (!in) return damaged(reason::unreadable());
    std::string head, stored_key, line;
    if (!std::getline(in, head) || in.eof()) return damaged(reason::incomplete_header());
    if (head != magic()) return damaged(reason::bad_header());
    if (!std::getline(in, stored_key) || in.eof()) return damaged(reason::incomplete_key());
    if (stored_key != key) {
        if (stored_key == key + "\r") return damaged(reason::crlf_key());
        LoadResult r;
        r.status = Status::DifferentJob;
        r.reason = reason::other_job();
        return r;
    }
    LoadResult r;
    r.status = Status::Ok;
    std::string done_gm, done_gq;
    std::size_t line_no = 2;
    while (std::getline(in, line)) {
        ++line_no;
        if (in.eof()) return damaged("line " + std::to_string(line_no) + " is incomplete");
        if (!valid_token(line)) return damaged("line " + std::to_string(line_no) + " is not a valid entry");
        if (line.compare(3, 4, "done") == 0) {
            std::string& seen = line[1] == 'M' ? done_gm : done_gq;
            if (!seen.empty() && seen != line)
                return damaged("line " + std::to_string(line_no) + " contradicts an earlier result");
            seen = line;
        }
        bool dup = false;
        for (const auto& t : r.tokens) if (t == line) { dup = true; break; }
        if (!dup) r.tokens.push_back(line);
    }
    return r;
}

// Tokens recorded for `key`.  Empty if the file is missing, damaged or belongs
// to another job.
inline std::vector<std::string> load(const std::filesystem::path& file, const std::string& key) {
    LoadResult r = load_checked(file, key);
    if (r.status != Status::Ok) r.tokens.clear();
    return std::move(r.tokens);
}

// One-line message for a record that was found but not used, or "" when there
// is nothing to tell the user (no file, or a usable record).
inline std::string notice(const std::filesystem::path& file, const LoadResult& r) {
    if (r.status == Status::Damaged)
        return "Warning: GMCHAIN progress record " + file.string() + " is damaged (" + r.reason +
               "); ignoring it, so the chain starts over.";
    if (r.status == Status::DifferentJob)
        return "Note: GMCHAIN progress record " + file.string() +
               " belongs to a different assignment; ignoring it, so the chain starts over.";
    return std::string();
}

inline void save(const std::filesystem::path& file, const std::string& key,
                 const std::vector<std::string>& tokens) {
    const std::filesystem::path tmp = file.string() + ".new";
    std::error_code ec;
    {
        std::ofstream out(tmp, std::ios::trunc | std::ios::binary);
        out << magic() << '\n' << key << '\n';
        for (const auto& token : tokens) out << token << '\n';
        out.close();   // flush now so a full disk is seen before the rename
        if (!out) { std::filesystem::remove(tmp, ec); return; }
    }
    std::filesystem::rename(tmp, file, ec);
    if (ec) std::filesystem::remove(tmp, ec);
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
        : file_(std::move(file)), key_(std::move(key)) {
        LoadResult r = load_checked(file_, key_);
        status_ = r.status;
        reason_ = std::move(r.reason);
        if (r.status == Status::Ok) tokens_ = std::move(r.tokens);
    }

    // Why the record found at construction was not used (Damaged or
    // DifferentJob), fixed at construction so a caller reports it once per run.
    Status status() const { return status_; }
    const std::string& reason() const { return reason_; }
    std::string notice() const {
        LoadResult r;
        r.status = status_;
        r.reason = reason_;
        return core::gm_chain_progress::notice(file_, r);
    }

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
    Status status_ = Status::Missing;
    std::string reason_;
};

} // namespace core::gm_chain_progress
