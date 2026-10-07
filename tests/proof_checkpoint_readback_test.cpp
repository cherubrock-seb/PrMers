// Host test for proof checkpoint residue writing with read-back verification.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/ProofCheckpoint.hpp"
#include "core/ProofSetMarin.hpp"

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

// A set whose load() returns corrupt data for the first `badLoads` calls.
struct FlakySet {
    int badLoads;
    int saves = 0;
    std::vector<uint32_t> stored;

    void save(uint32_t, const std::vector<uint32_t>& w) {
        ++saves;
        stored = w;
    }
    std::vector<uint32_t> load(uint32_t) {
        auto w = stored;
        if (badLoads > 0) {
            --badLoads;
            w[0] ^= 1u;
        }
        return w;
    }
};

} // namespace

int main() {
    const std::vector<uint32_t> words{1u, 2u, 3u, 4u, 5u, 6u};

    // A read-back mismatch that goes away on the second write: the residue is
    // written again and accepted.
    {
        FlakySet set{1, 0, {}};
        core::saveProofCheckpointVerified(set, 7, words);
        expect(set.saves == 2, "transient mismatch: residue written twice");
    }

    // A persistent mismatch is an error, not a warning.
    {
        FlakySet set{100, 0, {}};
        bool threw = false;
        try {
            core::saveProofCheckpointVerified(set, 7, words);
        } catch (const core::ProofCheckpointError&) {
            threw = true;
        }
        expect(threw, "persistent mismatch throws ProofCheckpointError");
        expect(set.saves == 2, "persistent mismatch: exactly one retry");
    }

    // The real ProofSetMarin, in a scratch directory.
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
                     ("prmers-proof-checkpoint-" + std::to_string(stamp));
    std::filesystem::create_directories(dir);
    const auto oldCwd = std::filesystem::current_path();
    std::filesystem::current_path(dir);

    constexpr uint32_t E = 191;  // 6 words
    core::ProofSetMarin set(E, 2);
    const std::vector<uint32_t> residue{9u, 8u, 7u, 6u, 5u, 4u};

    core::saveProofCheckpointVerified(set, E, residue);
    expect(set.load(E) == residue, "real set: residue round trip");

    // A residue shorter than the exponent needs cannot be read back whole.
    {
        bool threw = false;
        try {
            core::saveProofCheckpointVerified(set, E, std::vector<uint32_t>{1u, 2u});
        } catch (const core::ProofCheckpointError&) {
            threw = true;
        }
        expect(threw, "real set: short residue throws ProofCheckpointError");
    }

    // A residue file that cannot be created (a directory is in the way).
    {
        const auto path = core::ProofSetMarin::proofPath(E) / std::to_string(E);
        std::filesystem::remove(path);
        std::filesystem::create_directories(path);
        bool threw = false;
        try {
            core::saveProofCheckpointVerified(set, E, residue);
        } catch (const core::ProofCheckpointError&) {
            threw = true;
        }
        expect(threw, "real set: unwritable residue throws ProofCheckpointError");
    }

    std::filesystem::current_path(oldCwd);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    if (failures) return 1;
    std::cout << "Proof checkpoint read-back regression: PASS\n";
    return 0;
}
