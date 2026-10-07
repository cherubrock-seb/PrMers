// Host test for the Gaussian-Mersenne GMCHAIN completed-phase record.
#include "core/GmChainProgress.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace gp = core::gm_chain_progress;
namespace fs = std::filesystem;

#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; return 1; } } while (0)

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

    fs::remove_all(dir);
    std::cout << "GM chain progress test passed\n";
    return 0;
}
