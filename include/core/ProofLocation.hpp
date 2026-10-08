// core/ProofLocation.hpp
//
// Where a run keeps its proof files. They all live under the save path (-f),
// next to the checkpoints and results.txt, or under "." when it is empty:
//
//   <base>/<E>/proof/<iteration>    proof residues (checkpoint data)
//   <base>/proof/<E>-<power>.proof  finished proof (output)
//   <base>/proof-tmp/               proofs being written
//
// Earlier versions kept the residues in <E>/proof under the working directory
// whatever -f said. A resumed run that finds its residues there and not under
// the save path keeps reading them in place (adoptLegacy), and deletes them
// from there at the end of the job.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace core {

class ProofLocation {
public:
    ProofLocation() : base_(".") {}
    explicit ProofLocation(const std::string& savePath)
        : base_(savePath.empty() ? std::filesystem::path(".") : std::filesystem::path(savePath)) {}

    const std::filesystem::path& base() const { return base_; }

    // Directory of the finished proof files, and of the ones being written.
    std::filesystem::path proofDir() const { return base_ / "proof"; }
    std::filesystem::path proofTmpDir() const { return base_ / "proof-tmp"; }

    // Residues of exponent E.
    std::filesystem::path residueDir(uint32_t E) const {
        return base_ / std::to_string(E) / "proof";
    }
    // Where earlier versions kept them: relative to the working directory.
    static std::filesystem::path legacyResidueDir(uint32_t E) {
        return std::filesystem::path(std::to_string(E)) / "proof";
    }

    // The file a residue is written to: always under the save path.
    std::filesystem::path residueWriteFile(uint32_t E, uint32_t iter) const {
        return residueDir(E) / std::to_string(iter);
    }

    // The file a residue is read from: the one under the save path, or the old
    // location once adoptLegacy has accepted it, when only that holds it.
    std::filesystem::path residueFile(uint32_t E, uint32_t iter) const {
        const auto primary = residueWriteFile(E, iter);
        if (legacyAdopted_) {
            std::error_code ec;
            if (!std::filesystem::exists(primary, ec)) {
                const auto old = legacyResidueDir(E) / std::to_string(iter);
                if (std::filesystem::exists(old, ec)) return old;
            }
        }
        return primary;
    }

    bool legacyAdopted() const { return legacyAdopted_; }

    // True when `a` and `b` name the same place (symlinks, "./", a trailing
    // slash, relative or absolute spelling), whether or not they exist yet.
    static bool samePlace(const std::filesystem::path& a, const std::filesystem::path& b) {
        std::error_code e1, e2, e3, e4;
        const auto ca = std::filesystem::weakly_canonical(std::filesystem::absolute(a, e1), e2);
        const auto cb = std::filesystem::weakly_canonical(std::filesystem::absolute(b, e3), e4);
        if (e1 || e2 || e3 || e4) return a.lexically_normal() == b.lexically_normal();
        return ca.lexically_normal() == cb.lexically_normal();
    }

    // Called by a run that resumed an interrupted test (resumeIter > 0), with
    // the residue iterations it needs: every proof point before the iteration
    // it resumed at. Adopts the old location when some of them are not under
    // the save path (at the size of one residue) but are in the old location:
    // each residue is then read from the save path when it is there, from the
    // old location otherwise. Nothing is moved or copied, since the save path
    // is often on another disk: the old files are only read, and deleted by
    // clear() at the end of the job. Fills `note` with a line to print when it
    // adopts. Whether the residues are enough for a proof, and intact, is for
    // the caller's CRC checks (ProofSetMarin::effectivePower) to say; it calls
    // releaseLegacy() when they are not, so that files this run cannot use
    // are not deleted. A run that did not resume never calls this: it writes
    // every residue itself.
    bool adoptLegacy(uint32_t E, const std::vector<uint32_t>& needed, std::string& note) {
        legacyAdopted_ = false;
        note.clear();
        const auto primary = residueDir(E);
        const auto old = legacyResidueDir(E);
        if (samePlace(primary, old)) return false;
        std::error_code ec;
        if (!std::filesystem::is_directory(old, ec)) return false;
        const uintmax_t want = 4u + 4u * static_cast<uintmax_t>((E + 31u) / 32u);
        auto goodFile = [&](const std::filesystem::path& f) {
            std::error_code e;
            return std::filesystem::is_regular_file(f, e) && std::filesystem::file_size(f, e) == want;
        };
        bool usesOld = false;
        for (uint32_t k : needed) {
            const auto name = std::to_string(k);
            if (!goodFile(primary / name) && goodFile(old / name)) { usesOld = true; break; }
        }
        if (!usesOld) return false;
        legacyAdopted_ = true;
        std::error_code e2;
        auto shown = std::filesystem::absolute(old, e2);
        if (e2) shown = old;
        note = "Proof residues for M" + std::to_string(E) + " were not found under " +
               primary.string() + "; using the ones in the old location " + shown.string() +
               " (deleted from there when the test is done).";
        return true;
    }
    // The old location turned out to be of no use: leave it alone.
    void releaseLegacy() { legacyAdopted_ = false; }

    // Remove the residues of E: under the save path, and in the old location
    // when this run used them. The exponent directory goes too when that
    // leaves it empty; anything else in it stays.
    void clear(uint32_t E) const {
        std::error_code ec;
        std::filesystem::remove_all(residueDir(E), ec);
        std::filesystem::remove(base_ / std::to_string(E), ec);
        if (legacyAdopted_) {
            std::filesystem::remove_all(legacyResidueDir(E), ec);
            std::filesystem::remove(std::filesystem::path(std::to_string(E)), ec);
        }
    }

    // Where the residues of E are, absolute: "<dir>", plus " and <old dir>"
    // when the old location is in use.
    std::string describe(uint32_t E) const {
        auto show = [](const std::filesystem::path& p) {
            std::error_code ec;
            auto a = std::filesystem::absolute(p, ec);
            return (ec ? p : a).string();
        };
        std::string s = show(residueDir(E));
        if (legacyAdopted_) s += " and " + show(legacyResidueDir(E));
        return s;
    }

private:
    std::filesystem::path base_;
    bool legacyAdopted_ = false;
};

} // namespace core
