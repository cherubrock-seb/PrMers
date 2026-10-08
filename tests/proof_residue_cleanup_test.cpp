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

    // When the residues may be deleted: only once the result is saved and any
    // requested proof was made and the worktodo entry (if any) is retired;
    // never for LL or Wagstaff.
    {
        using A = core::ProofSetMarin::ResidueAction;
        auto act = core::ProofSetMarin::residueAction;
        //                 prp    wag    wanted done   saved  retired
        expect(act(true,  false, true,  true,  true, true)  == A::Clear, "proof made: clear");
        expect(act(true,  false, false, false, true, true)  == A::Clear, "proofs disabled: clear");
        expect(act(true,  false, true,  false, true, true)  == A::KeepProofFailed,
               "proof failed or did not verify: keep");
        expect(act(true,  false, true,  true,  false, true) == A::KeepResultNotSaved,
               "result not saved: keep");
        expect(act(true,  false, false, false, false, true) == A::KeepResultNotSaved,
               "result not saved without proof: keep");
        expect(act(true,  false, true,  false, false, true) == A::KeepResultNotSaved,
               "result not saved and proof failed: keep");
        expect(act(true,  false, true,  true,  true, false) == A::KeepEntryNotRetired,
               "result saved but worktodo entry not removed: keep");
        expect(act(true,  false, false, false, true, false) == A::KeepEntryNotRetired,
               "entry not removed without proof: keep");
        expect(act(true,  false, true,  true,  false, false) == A::KeepResultNotSaved,
               "result not saved, entry kept: keep");
        expect(act(false, false, true,  true,  true, true)  == A::NotApplicable, "LL: untouched");
        expect(act(true,  true,  true,  true,  true, true)  == A::NotApplicable, "Wagstaff: untouched");
    }

    // The message says where the residues are and that they can be deleted.
    {
        using A = core::ProofSetMarin::ResidueAction;
        const std::string proofMsg =
            core::ProofSetMarin::residuesKeptMessage(E, A::KeepProofFailed);
        const std::string saveMsg =
            core::ProofSetMarin::residuesKeptMessage(E, A::KeepResultNotSaved);
        const std::string where =
            fs::absolute(core::ProofSetMarin::proofPath(E)).string();
        expect(proofMsg.find(where) != std::string::npos, "message names the directory");
        expect(proofMsg.find("delete that directory by hand") != std::string::npos,
               "message says the residues can be deleted by hand");
        expect(proofMsg.find("retry") != std::string::npos, "proof message gives the reason");
        expect(saveMsg.find(where) != std::string::npos, "save message names the directory");
        expect(saveMsg.find("could not be saved") != std::string::npos, "save message gives the reason");
        const std::string entryMsg =
            core::ProofSetMarin::residuesKeptMessage(E, A::KeepEntryNotRetired);
        expect(entryMsg.find(where) != std::string::npos, "entry message names the directory");
        expect(entryMsg.find("worktodo entry could not be removed") != std::string::npos,
               "entry message gives the reason");
        expect(core::ProofSetMarin::residuesKeptMessage(E, A::Clear).empty(), "no message when clearing");
    }

    fs::current_path(oldCwd);
    std::error_code ec;
    fs::remove_all(dir, ec);

    if (failures) return 1;
    std::cout << "Proof residue cleanup regression: PASS\n";
    return 0;
}
