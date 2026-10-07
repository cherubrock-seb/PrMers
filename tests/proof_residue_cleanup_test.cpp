// Host test for the lifetime of the proof residue directory <E>/proof.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "core/ProofSetMarin.hpp"

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto dir = fs::temp_directory_path() /
                     ("prmers-proof-cleanup-" + std::to_string(stamp));
    fs::create_directories(dir);
    const auto oldCwd = fs::current_path();
    fs::current_path(dir);

    constexpr uint32_t E = 191;
    const std::vector<uint32_t> residue{1u, 2u, 3u, 4u, 5u, 6u};

    // Making the set (every test does) creates no directory; saving a residue
    // does.
    {
        core::ProofSetMarin set(E, 2);
        expect(!fs::exists(std::to_string(E)), "no directory before the first residue");
        set.save(E, residue);
        expect(fs::exists(core::ProofSetMarin::proofPath(E) / std::to_string(E)),
               "first residue creates <E>/proof");
        set.save(96, residue);
        expect(set.load(96) == residue, "residue round trip");
    }

    // Clearing removes the residues and the exponent directory.
    core::ProofSetMarin::clearResidues(E);
    expect(!fs::exists(std::to_string(E)), "<E> removed with its residues");

    // Clearing when nothing was ever saved is fine.
    core::ProofSetMarin::clearResidues(E);

    // Other content of <E> is not touched.
    {
        core::ProofSetMarin set(E, 2);
        set.save(E, residue);
        std::ofstream(fs::path(std::to_string(E)) / "keep.txt") << "x";
        core::ProofSetMarin::clearResidues(E);
        expect(!fs::exists(core::ProofSetMarin::proofPath(E)), "<E>/proof removed");
        expect(fs::exists(fs::path(std::to_string(E)) / "keep.txt"),
               "other files in <E> are kept");
    }

    // Another exponent's residues are not touched.
    {
        core::ProofSetMarin other(193, 2);
        other.save(193, std::vector<uint32_t>(7, 1u));
        core::ProofSetMarin::clearResidues(E);
        expect(fs::exists(core::ProofSetMarin::proofPath(193) / "193"),
               "other exponent kept");
    }

    fs::current_path(oldCwd);
    std::error_code ec;
    fs::remove_all(dir, ec);

    if (failures) return 1;
    std::cout << "Proof residue cleanup regression: PASS\n";
    return 0;
}
