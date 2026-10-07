// Host test for the Gaussian-Mersenne factoring checkpoint writer: a write that
// fails (read-only or missing directory, full disk) must raise an error without
// touching the previous checkpoint, and guarded_save must turn it into a
// warning so a periodic backup does not end the run.  No GPU needed.
#include "core/GmFactorCheckpoint.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
namespace gfc = core::gm_factor_ckpt;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; ++g_fail; } } while (0)

static gfc::FactorCheckpointHeader make_header(std::uint64_t token, std::size_t bytes) {
    gfc::FactorCheckpointHeader h{};
    std::copy(gfc::GMF_MAGIC.begin(), gfc::GMF_MAGIC.end(), h.magic);
    h.version = gfc::GMF_VERSION;
    h.mode = 1;
    h.phase = 1;
    h.p = 61;
    h.lift = 64;
    h.B1 = 1000;
    h.B2 = 2000;
    h.token = token;
    h.checkpoint_bytes = bytes;
    return h;
}

static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Reads the file back the way load_factor_checkpoint does.
static bool read_back(const fs::path& p, std::uint64_t& token, std::vector<char>& data, std::size_t bytes) {
    File f(p.string());
    if (!f.exists()) return false;
    gfc::FactorCheckpointHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    data.resize(bytes);
    if (!f.read(data.data(), data.size()) || !f.check_crc32()) return false;
    token = h.token;
    return true;
}

int main(int argc, char** argv) {
    const fs::path dir = fs::path(argc > 1 ? argv[1] : ".") / "gmckpt";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path ckpt = dir / "gm_pm1_p61_stage1.ckpt";
    const std::vector<char> data1(4096, 'a');
    const std::vector<char> data2(4096, 'b');

    // A good write round-trips through the reader, and the next one rotates .old.
    gfc::write_checkpoint_file(ckpt, make_header(11, data1.size()), data1);
    std::uint64_t token = 0;
    std::vector<char> back;
    CHECK(read_back(ckpt, token, back, data1.size()) && token == 11 && back == data1);
    gfc::write_checkpoint_file(ckpt, make_header(22, data2.size()), data2);
    CHECK(read_back(ckpt, token, back, data2.size()) && token == 22 && back == data2);
    CHECK(read_back(fs::path(ckpt.string() + ".old"), token, back, data1.size()) && token == 11);
    CHECK(!fs::exists(ckpt.string() + ".new"));
    const std::string good = slurp(ckpt);
    const std::string good_old = slurp(ckpt.string() + ".old");

    // Missing directory: an error, not a crash (the file used to be written
    // through a null FILE*).
    bool threw = false;
    try {
        gfc::write_checkpoint_file(dir / "no" / "such" / "x.ckpt", make_header(1, data1.size()), data1);
    } catch (const std::exception& ex) {
        threw = true;
        std::cout << "missing dir: " << ex.what() << "\n";
    }
    CHECK(threw);

    // Read-only directory (not enforced for root): the new write fails and the
    // existing checkpoint and its .old are untouched.
    if (geteuid() != 0) {
        chmod(dir.c_str(), 0555);
        threw = false;
        try {
            gfc::write_checkpoint_file(ckpt, make_header(33, data2.size()), data2);
        } catch (const std::exception& ex) {
            threw = true;
            std::cout << "read-only dir: " << ex.what() << "\n";
        }
        CHECK(threw);
        CHECK(slurp(ckpt) == good);
        CHECK(slurp(ckpt.string() + ".old") == good_old);
        CHECK(!fs::exists(ckpt.string() + ".new"));

        // guarded_save: warning, false, no exception; success path returns true.
        std::ostringstream cerr_capture;
        auto* old_buf = std::cerr.rdbuf(cerr_capture.rdbuf());
        const bool ok_fail = gfc::guarded_save(ckpt, [&]() {
            gfc::write_checkpoint_file(ckpt, make_header(44, data2.size()), data2);
        });
        std::cerr.rdbuf(old_buf);
        CHECK(!ok_fail);
        CHECK(cerr_capture.str().find("was not saved") != std::string::npos);
        CHECK(cerr_capture.str().find("previous checkpoint") != std::string::npos);
        std::cout << "guarded_save warning: " << cerr_capture.str();
        chmod(dir.c_str(), 0755);
    } else {
        std::cout << "running as root: read-only directory case skipped\n";
    }
    CHECK(gfc::guarded_save(ckpt, [&]() {
        gfc::write_checkpoint_file(ckpt, make_header(55, data2.size()), data2);
    }));
    CHECK(read_back(ckpt, token, back, data2.size()) && token == 55);

    // Full disk: the temporary file is /dev/full, so every write fails with
    // ENOSPC (at flush/close, which the old writer never checked).
    if (fs::exists("/dev/full")) {
        const std::string before = slurp(ckpt);
        const std::string before_old = slurp(ckpt.string() + ".old");
        fs::create_symlink("/dev/full", ckpt.string() + ".new");
        threw = false;
        try {
            gfc::write_checkpoint_file(ckpt, make_header(66, data2.size()), data2);
        } catch (const std::exception& ex) {
            threw = true;
            std::cout << "full disk: " << ex.what() << "\n";
        }
        CHECK(threw);
        CHECK(slurp(ckpt) == before);
        CHECK(slurp(ckpt.string() + ".old") == before_old);
        CHECK(!fs::exists(fs::symlink_status(ckpt.string() + ".new")));
    }

    fs::remove_all(dir);
    if (g_fail) return 1;
    std::cout << "gm factor checkpoint write-failure test passed\n";
    return 0;
}
