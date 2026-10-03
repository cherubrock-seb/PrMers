// Host test for the Gaussian-Mersenne ECM completed-curve counter.
#include "core/GmEcmProgress.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace gp = core::gm_ecm_progress;
namespace fs = std::filesystem;

#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; return 1; } } while (0)

int main() {
    const fs::path dir = fs::temp_directory_path() / "prmers_gm_ecm_progress_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path f = dir / "gm_ecm_p13_curves.done";

    CHECK(gp::load(f, "k") == 0);                 // no file
    gp::save(f, "k", 3);
    CHECK(gp::load(f, "k") == 3);
    CHECK(gp::load(f, "other-job") == 0);         // different job is ignored
    gp::save(f, "k", 4);
    CHECK(gp::load(f, "k") == 4);

    std::atomic<bool> interrupted{false};
    {   // interrupted run keeps the counter
        gp::Guard g(f, interrupted);
        interrupted.store(true);
    }
    CHECK(gp::load(f, "k") == 4);
    interrupted.store(false);
    {   // finished run removes it
        gp::Guard g(f, interrupted);
    }
    CHECK(gp::load(f, "k") == 0);
    CHECK(!fs::exists(f));

    gp::save(f, "k", 2);
    try {   // an exception keeps it
        gp::Guard g(f, interrupted);
        throw 1;
    } catch (int) {}
    CHECK(gp::load(f, "k") == 2);

    fs::remove_all(dir);
    std::cout << "GM ECM progress counter test passed\n";
    return 0;
}
