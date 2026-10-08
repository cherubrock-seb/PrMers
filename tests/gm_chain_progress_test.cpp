// Host test for the Gaussian-Mersenne GMCHAIN completed-phase record.
#include "core/GmChainProgress.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>

namespace gp = core::gm_chain_progress;
namespace fs = std::filesystem;

#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; return 1; } } while (0)

namespace {

const std::string kMagic = "PRMERS-GM-CHAIN 1";

void write_raw(const fs::path& f, const std::string& content) {
    std::ofstream out(f, std::ios::binary | std::ios::trunc);
    out << content;
}

// A progress file for `key` whose token section is `body` (written verbatim).
std::string file_with(const std::string& key, const std::string& body) {
    return kMagic + "\n" + key + "\n" + body;
}

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "prmers_gm_chain_progress_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path f = dir / "gm_chain_p13_phases.done";
    const std::string key = "GMCHAIN=13,100,200,50,50,2,0,262144,proth,BOTH";

    {
        gp::Progress fresh(f, key);                       // no file yet
        CHECK(!fresh.has(gp::phase_token("GM", "pm1")));
        CHECK(!fresh.family_rc("GM").has_value());
        fresh.mark(gp::phase_token("GM", "pm1"));
        fresh.mark(gp::phase_token("GM", "pm1"));         // idempotent
        fresh.mark(gp::done_token("GM", 1));
        fresh.mark(gp::phase_token("GQ", "pm1"));
    }
    {
        gp::Progress resumed(f, key);                     // a restart sees the same phases
        CHECK(resumed.has(gp::phase_token("GM", "pm1")));
        CHECK(!resumed.has(gp::phase_token("GM", "ecm")));
        CHECK(resumed.has(gp::phase_token("GQ", "pm1")));
        CHECK(resumed.family_rc("GM").value_or(-1) == 1);
        CHECK(!resumed.family_rc("GQ").has_value());
        resumed.mark(gp::done_token("GQ", 0));
    }
    {
        gp::Progress again(f, key);
        CHECK(again.family_rc("GQ").value_or(-1) == 0);
    }
    {
        gp::Progress other(f, "GMCHAIN=13,100,200,50,50,2,0,262144,factor,BOTH");
        CHECK(!other.has(gp::phase_token("GM", "pm1")));  // a different job is ignored
        CHECK(!other.family_rc("GM").has_value());
    }
    {
        gp::Progress done(f, key);
        done.clear();
        CHECK(!fs::exists(f));
        CHECK(!gp::Progress(f, key).has(gp::phase_token("GM", "pm1")));
    }


    // Hostile input.  Every body below is written verbatim after a valid header
    // and key; a malformed record must never yield a finished family.
    {
        const std::string k = "job";
        auto rc_of = [&](const std::string& body) {
            write_raw(f, file_with(k, body));
            return gp::Progress(f, k).family_rc("GM");
        };
        auto tokens_of = [&](const std::string& body) {
            write_raw(f, file_with(k, body));
            return gp::load(f, k);
        };

        // Trailing junk after the result must not resume as a finished family.
        CHECK(!rc_of("GM done 1junk\n").has_value());
        CHECK(!rc_of("GM done 0junk\n").has_value());

        // Well-formed results.
        CHECK(rc_of("GM done 0\n").value_or(-1) == 0);
        CHECK(rc_of("GM done 1\n").value_or(-1) == 1);
        CHECK(rc_of("GM pm1\nGM ecm\nGM done 1\n").value_or(-1) == 1);
        CHECK(rc_of("GM done 1\nGM done 1\n").value_or(-1) == 1);   // identical duplicate
        CHECK(tokens_of("GM pm1\nGM pm1\n").size() == 1);
        CHECK(!rc_of("GQ done 1\n").has_value());                    // other family
        CHECK(rc_of("").has_value() == false);

        // Malformed completed-family tokens.
        const char* bad_done[] = {
            "GM done 1junk\n", "GM done 1 \n", "GM done  1\n", "GM done 1\r\n",
            "GM done \n", "GM done\n", "GM done \n", "GM done -0\n", "GM done -1\n",
            "GM done +1\n", "GM done 01\n", "GM done 00\n", "GM done 0x1\n", "GM done 1.0\n",
            "GM done 1e0\n", "GM done 2\n", "GM done 3\n", "GM done 10\n", "GM done 255\n",
            "GM done 4294967296\n", "GM done 4294967297\n", "GM done 2147483648\n",
            "GM done 99999999999999999999999999\n", "GM  done 1\n", " GM done 1\n",
            "GM done 1\t\n", "GM done\t1\n", "gm done 1\n", "GM DONE 1\n", "GM Done 1\n",
            "GMdone 1\n", "GM done 1 1\n", "GM done a\n", "GM done \xef\xbc\x91\n",
        };
        for (const char* body : bad_done) {
            if (rc_of(body).has_value()) {
                std::cerr << "FAIL: accepted malformed token: " << body;
                return 1;
            }
            CHECK(tokens_of(body).empty());
        }
        {
            std::string with_nul = "GM done 1";
            with_nul.push_back('\0');
            with_nul += "\n";
            CHECK(!rc_of(with_nul).has_value());
        }

        // Unknown families, phases and stray lines make the whole record damaged.
        CHECK(tokens_of("XX pm1\n").empty());
        CHECK(tokens_of("GM pm1\nXX pm1\n").empty());
        CHECK(tokens_of("GM pm2\n").empty());
        CHECK(tokens_of("GM proth\n").empty());
        CHECK(tokens_of("GM\n").empty());
        CHECK(tokens_of("GM \n").empty());
        CHECK(tokens_of("GM pm1 \n").empty());
        CHECK(tokens_of("BOTH done 1\n").empty());
        CHECK(tokens_of("GM pm1\n\nGM ecm\n").empty());            // empty line inside
        CHECK(tokens_of("GM pm1\n\n").empty());                     // empty trailing line
        CHECK(!rc_of("GM done 1\ngarbage\n").has_value());          // good line then junk
        CHECK(!rc_of("garbage\nGM done 1\n").has_value());
        CHECK(!rc_of("GM done 1\nGM done 0\n").has_value());        // contradictory results
        CHECK(!rc_of("GM done 0\nGM done 1\n").has_value());

        // CRLF file (foreign or mangled): header, key and tokens all fail.
        write_raw(f, kMagic + "\r\n" + k + "\r\nGM done 1\r\n");
        CHECK(gp::load(f, k).empty());
        write_raw(f, kMagic + "\n" + k + "\r\nGM done 1\n");
        CHECK(gp::load(f, k).empty());

        // Truncated file: last line has no newline, or the header/key is cut.
        CHECK(!rc_of("GM done 1").has_value());
        CHECK(tokens_of("GM pm1\nGM ec").empty());
        CHECK(tokens_of("GM pm1\nGM ecm").empty());
        write_raw(f, kMagic);
        CHECK(gp::load(f, k).empty());
        write_raw(f, kMagic + "\n");
        CHECK(gp::load(f, k).empty());
        write_raw(f, kMagic + "\n" + k);                              // key cut, no newline
        CHECK(gp::load(f, k).empty());
        write_raw(f, kMagic.substr(0, 10));
        CHECK(gp::load(f, k).empty());
        write_raw(f, "");
        CHECK(gp::load(f, k).empty());

        // Wrong version / foreign header / wrong key.
        write_raw(f, "PRMERS-GM-CHAIN 2\n" + k + "\nGM done 1\n");
        CHECK(gp::load(f, k).empty());
        write_raw(f, "something else\n" + k + "\nGM done 1\n");
        CHECK(gp::load(f, k).empty());
        write_raw(f, file_with("job2", "GM done 1\n"));
        CHECK(gp::load(f, k).empty());
        write_raw(f, file_with("job ", "GM done 1\n"));
        CHECK(gp::load(f, k).empty());

        // Oversized file is ignored rather than read.
        write_raw(f, file_with(k, std::string(2u << 20, 'A') + "\n"));
        CHECK(gp::load(f, k).empty());
        write_raw(f, file_with(k, "GM pm1\n" + std::string(2u << 20, ' ')));
        CHECK(gp::load(f, k).empty());

        // A directory or a missing file in place of the record.
        fs::remove(f);
        fs::create_directory(f);
        CHECK(gp::load(f, k).empty());
        fs::remove(f);
        CHECK(gp::load(f, k).empty());

        // A damaged record does not poison later progress: marking rewrites it.
        write_raw(f, file_with(k, "GM done 1junk\n"));
        {
            gp::Progress p(f, k);
            CHECK(!p.family_rc("GM").has_value());
            p.mark(gp::phase_token("GM", "pm1"));
        }
        {
            gp::Progress p(f, k);
            CHECK(p.has(gp::phase_token("GM", "pm1")));
            CHECK(!p.family_rc("GM").has_value());
        }
        fs::remove(f);

        // Only legitimate tokens are ever persisted.
        {
            gp::Progress p(f, k);
            p.mark(gp::done_token("GM", 2));         // error / interrupted result
            p.mark(gp::done_token("GM", -1));
            p.mark(gp::done_token("GM", 7));
            p.mark(gp::phase_token("XX", "pm1"));
            p.mark(gp::phase_token("GM", "proth"));
            p.mark("GM done 1junk");
            p.mark("");
            CHECK(!fs::exists(f));                    // nothing valid, nothing written
            p.mark(gp::done_token("GM", 1));
            p.mark(gp::done_token("GM", 0));          // second result is ignored
            CHECK(p.family_rc("GM").value_or(-1) == 1);
        }
        CHECK(gp::Progress(f, k).family_rc("GM").value_or(-1) == 1);
        fs::remove(f);

        // Unwritable location: mark must not throw and leaves no stray file.
        {
            gp::Progress p(dir / "no_such_dir" / "x.done", k);
            p.mark(gp::phase_token("GM", "pm1"));
            CHECK(!fs::exists(dir / "no_such_dir"));
        }
        // Rename onto a directory fails; the temp file is cleaned up.
        {
            const fs::path blocked = dir / "blocked.done";
            fs::create_directories(blocked / "child");
            gp::Progress p(blocked, k);
            p.mark(gp::phase_token("GM", "pm1"));
            CHECK(!fs::exists(blocked.string() + ".new"));
            fs::remove_all(blocked);
        }

        // Write failure (full disk): the temp file is /dev/full, so the buffered
        // data fails on flush.  The record must not be installed.
        if (fs::exists("/dev/full")) {
            const fs::path full = dir / "full.done";
            std::error_code ec;
            fs::create_symlink("/dev/full", full.string() + ".new", ec);
            if (!ec) {
                gp::Progress p(full, k);
                p.mark(gp::phase_token("GM", "pm1"));
                CHECK(!fs::exists(full));
                CHECK(!fs::is_symlink(full.string() + ".new"));
            }
        }
    }

    // Why a record was not used: a missing file is silent, a record for another
    // job gets a note, and every kind of damage gets a warning with its reason.
    {
        const std::string k = "job";
        namespace rs = gp::reason;
        const auto D = gp::Status::Damaged;
        auto check = [&](const std::string& content, gp::Status status, const std::string& why) {
            write_raw(f, content);
            const gp::LoadResult r = gp::load_checked(f, k);
            if (r.status != status || r.reason != why) {
                std::cerr << "FAIL: expected reason '" << why << "', got '" << r.reason << "' for:\n"
                          << content << "\n";
                return false;
            }
            const gp::Progress p(f, k);
            return p.status() == status && p.reason() == why &&
                   p.notice().empty() == (status == gp::Status::Ok);
        };

        // Missing file: no reason, no message.
        fs::remove(f);
        {
            const gp::LoadResult r = gp::load_checked(f, k);
            CHECK(r.status == gp::Status::Missing && r.reason.empty() && r.tokens.empty());
            const gp::Progress p(f, k);
            CHECK(p.status() == gp::Status::Missing && p.reason().empty() && p.notice().empty());
        }
        // Dangling symlink behaves like a missing file.
        {
            std::error_code ec;
            fs::create_symlink(dir / "nowhere", f, ec);
            if (!ec) {
                CHECK(gp::Progress(f, k).notice().empty());
                CHECK(gp::load_checked(f, k).status == gp::Status::Missing);
                fs::remove(f);
            }
        }
        // A usable record has no message, with or without tokens.
        CHECK(check(file_with(k, ""), gp::Status::Ok, ""));
        CHECK(check(file_with(k, "GM pm1\nGM done 1\n"), gp::Status::Ok, ""));

        // A record for another job: a note that names the file, not a damage warning.
        CHECK(check(file_with("job2", "GM done 1\n"), gp::Status::DifferentJob, rs::other_job()));
        {
            const std::string n = gp::Progress(f, k).notice();
            CHECK(n.rfind("Note: ", 0) == 0);
            CHECK(n.find(f.string()) != std::string::npos);
            CHECK(n.find("different assignment") != std::string::npos);
            CHECK(n.find("starts over") != std::string::npos);
            CHECK(n.find("damaged") == std::string::npos);
        }
        CHECK(check(file_with("job ", "GM done 1\n"), gp::Status::DifferentJob, rs::other_job()));
        CHECK(check(file_with("", "GM done 1\n"), gp::Status::DifferentJob, rs::other_job()));

        // Damage: each case reports its reason.
        CHECK(check(file_with(k, "GM done 0junk\n"), D, "line 3 is not a valid entry"));
        CHECK(check(file_with(k, "GM pm1\nGM done 1junk\n"), D, "line 4 is not a valid entry"));
        CHECK(check(file_with(k, "GM pm1\n\nGM ecm\n"), D, "line 4 is not a valid entry"));
        CHECK(check(file_with(k, "XX pm1\n"), D, "line 3 is not a valid entry"));
        CHECK(check(file_with(k, "GM done 1\nGM done 0\n"), D, "line 4 contradicts an earlier result"));
        CHECK(check(file_with(k, "GM done 0\nGM done 1\n"), D, "line 4 contradicts an earlier result"));
        CHECK(check(file_with(k, "GM done 1"), D, "line 3 is incomplete"));
        CHECK(check(file_with(k, "GM pm1\nGM ec"), D, "line 4 is incomplete"));
        CHECK(check(file_with(k, "GM done 1\r\n"), D, "line 3 is not a valid entry"));
        CHECK(check(file_with(k, "GM done 1\n") + "garbage\n", D, "line 4 is not a valid entry"));
        CHECK(check(file_with(k, std::string(2u << 20, 'A') + "\n"), D, rs::too_large()));
        CHECK(check("PRMERS-GM-CHAIN 2\n" + k + "\nGM done 1\n", D, rs::bad_header()));
        CHECK(check("something else\n" + k + "\nGM done 1\n", D, rs::bad_header()));
        CHECK(check(kMagic + "\r\n" + k + "\r\nGM done 1\r\n", D, rs::bad_header()));
        CHECK(check(kMagic + "\n" + k + "\r\nGM done 1\n", D, rs::crlf_key()));
        CHECK(check("", D, rs::incomplete_header()));
        CHECK(check(kMagic, D, rs::incomplete_header()));
        CHECK(check(kMagic.substr(0, 10), D, rs::incomplete_header()));
        CHECK(check(kMagic + "\n", D, rs::incomplete_key()));
        CHECK(check(kMagic + "\n" + k, D, rs::incomplete_key()));
        CHECK(check(std::string(4, '\0'), D, rs::incomplete_header()));
        CHECK(check(std::string("\xff\xfe\n") + k + "\n", D, rs::bad_header()));
        {
            write_raw(f, file_with(k, "GM done 0junk\n"));
            const std::string n = gp::Progress(f, k).notice();
            CHECK(n.rfind("Warning: ", 0) == 0);
            CHECK(n.find(f.string()) != std::string::npos);
            CHECK(n.find("damaged") != std::string::npos);
            CHECK(n.find("line 3 is not a valid entry") != std::string::npos);
            CHECK(n.find("starts over") != std::string::npos);
            CHECK(n.find('\n') == std::string::npos);          // one line
            CHECK(n.find("junk") == std::string::npos);        // file content is not echoed
        }

        // A directory in place of the record is damage, not a missing file.
        fs::remove(f);
        fs::create_directory(f);
        CHECK(gp::load_checked(f, k).status == D);
        CHECK(gp::load_checked(f, k).reason == rs::not_regular_file());
        CHECK(!gp::Progress(f, k).notice().empty());
        fs::remove(f);

        // The reason is fixed at construction: marking progress (which rewrites
        // the file) does not change what an existing Progress reports.
        write_raw(f, file_with(k, "GM done 0junk\n"));
        {
            gp::Progress p(f, k);
            const std::string first = p.notice();
            CHECK(!first.empty());
            p.mark(gp::phase_token("GM", "pm1"));
            CHECK(p.notice() == first);
            CHECK(p.has(gp::phase_token("GM", "pm1")));
            CHECK(!p.family_rc("GM").has_value());
        }
        // The rewritten record is good, so the next run is silent.
        CHECK(gp::Progress(f, k).notice().empty());
        CHECK(gp::Progress(f, k).status() == gp::Status::Ok);
        // load() keeps returning empty tokens for every unusable record.
        write_raw(f, file_with(k, "GM done 0junk\n"));
        CHECK(gp::load(f, k).empty());
        fs::remove(f);
    }

    fs::remove_all(dir);
    std::cout << "GM chain progress test passed\n";
    return 0;
}
