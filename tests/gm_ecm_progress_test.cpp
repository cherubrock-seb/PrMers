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

    // Write failure (full disk): the temp file is /dev/full, so the buffered
    // data fails on flush.  The previous counter must survive and no temp
    // file may be left behind.
    if (fs::exists("/dev/full")) {
        gp::save(f, "k", 5);
        CHECK(gp::load(f, "k") == 5);
        const std::string tmp = f.string() + ".new";
        std::error_code ec;
        fs::create_symlink("/dev/full", tmp, ec);
        if (!ec) {
            gp::save(f, "k", 6);
            CHECK(gp::load(f, "k") == 5);
            CHECK(!fs::is_symlink(f));
            CHECK(!fs::is_symlink(tmp) && !fs::exists(tmp));
        }
    }

    // Rename onto a directory fails; the temp file is cleaned up.
    {
        const fs::path blocked = dir / "blocked.done";
        fs::create_directories(blocked / "child");
        gp::save(blocked, "k", 1);
        CHECK(!fs::exists(blocked.string() + ".new"));
        fs::remove_all(blocked);
    }

    // Unwritable location: save must not throw.
    gp::save(dir / "no_such_dir" / "x.done", "k", 1);
    CHECK(!fs::exists(dir / "no_such_dir"));

    fs::remove_all(dir);
    std::cout << "GM ECM progress counter test passed\n";
    return 0;
}
