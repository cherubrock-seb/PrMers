// Host test for where proof residues and proof files live: under the save path
// (-f), not the working directory, and for the use of residues an older version
// left in <E>/proof under the working directory by a run that resumes.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include "core/ProofLocation.hpp"
#include "core/ProofSetMarin.hpp"

namespace fs = std::filesystem;

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

constexpr uint32_t E = 191;                   // 6 words per residue
const std::vector<uint32_t> residueA{1u, 2u, 3u, 4u, 5u, 6u};
const std::vector<uint32_t> residueB{9u, 8u, 7u, 6u, 5u, 4u};

// Entries of a directory, for "nothing appears in the working directory".
std::vector<std::string> listing(const fs::path& dir) {
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(dir))
        names.push_back(e.path().filename().string());
    return names;
}

// The residues an older version leaves for exponent E at power 2: one file per
// proof point, in <E>/proof under the working directory.
void writeOldResidues(uint32_t exponent, const std::vector<uint32_t>& words) {
    core::ProofSetMarin old(exponent, 2);  // default location: the working directory
    for (uint32_t k : {48u, 96u, 144u, exponent})
        old.save(k, words);
}

} // namespace

int main() {
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() / ("prmers-proof-location-" + std::to_string(stamp));
    fs::create_directories(root / "cwd");
    const fs::path oldCwd = fs::current_path();
    fs::current_path(root / "cwd");
    const fs::path cwd = fs::current_path();

    // The proof points of E = 191, power 2, are 48, 96, 144 and 191.
    expect(core::ProofSetMarin::isInPoints(E, 2, 48) && core::ProofSetMarin::isInPoints(E, 2, 144),
           "test assumes the proof points 48, 96, 144, 191");

    // Paths: every proof file is under the save path; "" means ".".
    {
        core::ProofLocation def;
        core::ProofLocation empty{std::string()};
        core::ProofLocation f{(root / "save").string()};
        expect(def.residueDir(E) == fs::path(".") / "191" / "proof", "default residues in ./<E>/proof");
        expect(empty.residueDir(E) == def.residueDir(E), "empty save path is the working directory");
        expect(empty.proofDir() == fs::path(".") / "proof" && empty.proofTmpDir() == fs::path(".") / "proof-tmp",
               "default proof files in ./proof and ./proof-tmp");
        expect(f.residueDir(E) == root / "save" / "191" / "proof", "residues under -f");
        expect(f.proofDir() == root / "save" / "proof", "proof files under -f");
        expect(f.proofTmpDir() == root / "save" / "proof-tmp", "proof-tmp under -f");
    }

    // A run with -f elsewhere writes its residues there and nothing in the
    // working directory.
    {
        const fs::path save = root / "elsewhere";
        core::ProofSetMarin set(E, 2, {}, core::ProofLocation(save.string()));
        expect(listing(cwd).empty() && !fs::exists(save), "no directory before the first residue");
        set.save(48, residueA);
        set.save(E, residueB);
        expect(fs::exists(save / "191" / "proof" / "48"), "residue under -f");
        expect(set.load(48) == residueA && set.load(E) == residueB, "residues read back from -f");
        expect(listing(cwd).empty(), "nothing appears in the working directory");
        set.location().clear(E);
        expect(!fs::exists(save / "191"), "cleared under -f");
        expect(listing(cwd).empty(), "clearing leaves the working directory alone");
    }

    // Spellings of -f: trailing slash, relative, and through a symlink all
    // reach the same directory.
    {
        fs::create_directories(root / "real");
        fs::create_symlink(root / "real", root / "link");
        const std::vector<std::string> spellings{
            (root / "real").string() + "/", "../real", "../real/./", (root / "link").string()};
        for (const auto& s : spellings) {
            core::ProofSetMarin set(E, 2, {}, core::ProofLocation(s));
            set.save(96, residueA);
            expect(fs::exists(root / "real" / "191" / "proof" / "96"), ("-f " + s).c_str());
            expect(set.load(96) == residueA, ("-f " + s + " reads back").c_str());
            fs::remove_all(root / "real" / "191");
        }
        expect(listing(cwd).empty(), "spellings of -f leave the working directory empty");
    }

    // Two runs with different -f in the same working directory do not see
    // each other's residues.
    {
        core::ProofSetMarin a(E, 2, {}, core::ProofLocation((root / "A").string()));
        core::ProofSetMarin b(E, 2, {}, core::ProofLocation((root / "B").string()));
        a.save(96, residueA);
        b.save(96, residueB);
        expect(a.load(96) == residueA && b.load(96) == residueB, "separate -f, separate residues");
        a.location().clear(E);
        expect(b.load(96) == residueB, "clearing one -f leaves the other");
        b.location().clear(E);
        expect(listing(cwd).empty(), "two -f runs leave the working directory empty");
    }

    // An -f that cannot be written: the residue save throws (the driver turns
    // that into "no proof, the PRP continues"), nothing lands in the cwd.
    if (geteuid() != 0) {
        fs::create_directories(root / "ro");
        fs::permissions(root / "ro", fs::perms::owner_read | fs::perms::owner_exec);
        core::ProofSetMarin set(E, 2, {}, core::ProofLocation((root / "ro").string()));
        bool threw = false;
        try { set.save(96, residueA); } catch (const std::runtime_error&) { threw = true; }
        expect(threw, "unwritable -f: save throws");
        expect(listing(cwd).empty(), "unwritable -f: nothing in the working directory");
        fs::permissions(root / "ro", fs::perms::owner_all);
    } else {
        std::cout << "SKIP: unwritable -f (running as root)\n";
    }

    // Resume with residues in the old location (and none under -f).
    {
        const fs::path save = root / "resume";
        writeOldResidues(E, residueA);
        core::ProofSetMarin set(E, 2, {}, core::ProofLocation(save.string()));
        std::string note;

        expect(!set.adoptLegacyResidues(0, note) && !set.location().legacyAdopted(),
               "a run that did not resume never uses the old location");
        expect(set.adoptLegacyResidues(100, note), "resume at 100: old residues adopted");
        expect(note.find("old location") != std::string::npos && note.find(fs::absolute("191/proof").string()) != std::string::npos,
               "one-line note names the old location");
        expect(note.find('\n') == std::string::npos, "the note is one line");
        expect(set.load(48) == residueA && set.load(96) == residueA, "old residues read in place");
        expect(!fs::exists(save), "adopting creates nothing under -f");

        // New residues go under -f; mixed per file, -f first.
        set.save(144, residueB);
        expect(fs::exists(save / "191" / "proof" / "144"), "new residues are written under -f");
        expect(set.load(144) == residueB, "the -f copy wins over the old one");
        expect(set.load(48) == residueA, "residues not yet under -f still come from the old location");

        // A corrupt old residue still fails its CRC check.
        {
            std::fstream f("191/proof/48", std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(8);
            f.put('\x7f');
        }
        bool threw = false;
        try { set.load(48); } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("CRC32 mismatch") != std::string::npos;
        }
        expect(threw, "old residue: CRC still checked");

        // The end of the job deletes from where they were used.
        auto msg = core::ProofSetMarin::residuesKeptMessage(
            set.location(), E, core::ProofSetMarin::ResidueAction::KeepProofFailed);
        expect(msg.find((save / "191" / "proof").string()) != std::string::npos &&
               msg.find(fs::absolute("191/proof").string()) != std::string::npos,
               "kept message names both directories");
        core::ProofSetMarin::clearResidues(set.location(), E);
        expect(!fs::exists("191") && !fs::exists(save / "191"), "both locations cleared");
    }

    // Old residues that cannot make a proof are adopted for reading but let go
    // by the power check, and so are not deleted; partial sets give the lower
    // power they still allow; a run whose residues are all under -f already
    // does not adopt.
    {
        using PS = core::ProofSetMarin;
        const fs::path save = root / "partial";
        core::ProofSetMarin old(E, 2);
        old.save(48, residueA);  // 96 missing
        core::ProofSetMarin set(E, 2, {}, core::ProofLocation(save.string()));
        std::string note;
        expect(set.adoptLegacyResidues(100, note), "a partial old set is adopted for the power check");
        expect(PS::effectivePower(set.location(), E, 2, 100) == 0, "but it makes no proof at 100 (96 is missing)");
        set.releaseLegacyResidues();
        expect(!set.location().legacyAdopted(), "released");
        PS::clearResidues(set.location(), E);
        expect(fs::exists("191/proof/48"), "old residues that were no use are not deleted");
        fs::remove_all("191");

        // Power 3 has the points 24 48 72 96 120 144 168 191, power 2 48 96
        // 144 191: an old set with only 48 and 96 allows power 2 at 100.
        core::ProofSetMarin old3(E, 3);
        old3.save(48, residueA);
        old3.save(96, residueA);
        core::ProofSetMarin set3(E, 3, {}, core::ProofLocation(save.string()));
        expect(set3.adoptLegacyResidues(100, note), "old 48 and 96 adopted");
        expect(PS::effectivePower(set3.location(), E, 3, 100) == 2, "and allow power 2, read from the old location");
        expect(PS::effectivePower(core::ProofLocation(save.string()), E, 3, 100) == 0,
               "while without the old location nothing is possible");
        set3.location().clear(E);
        expect(!fs::exists("191"), "adopted residues deleted from the old location");
        fs::remove_all("191");

        // Wrong size: the file is not usable, and is not looked at as a residue.
        writeOldResidues(E, residueA);
        { std::ofstream("191/proof/96", std::ios::binary | std::ios::trunc) << "short"; }
        expect(PS::effectivePower(core::ProofLocation(), E, 2, 100) == 0, "a residue of the wrong size makes no proof");
        fs::remove_all("191");

        // Everything already under -f: nothing to adopt, old ones untouched.
        writeOldResidues(E, residueB);
        core::ProofSetMarin full(E, 2, {}, core::ProofLocation(save.string()));
        full.save(48, residueA);
        full.save(96, residueA);
        expect(!full.adoptLegacyResidues(100, note) && note.empty(), "residues all under -f: old location ignored");
        core::ProofSetMarin::clearResidues(full.location(), E);
        expect(fs::exists("191/proof/48") && !fs::exists(save / "191"),
               "residues in both locations: only the -f ones are deleted");
        fs::remove_all("191");

        // Another exponent's old residues are never looked at.
        writeOldResidues(193, residueA);
        core::ProofSetMarin other(E, 2, {}, core::ProofLocation(save.string()));
        expect(!other.adoptLegacyResidues(100, note), "other exponent's old residues are not adopted");
        fs::remove_all("193");
    }

    // -f unset or the same place as the working directory (".", "", "./", an
    // absolute spelling, a symlink to it): old and new locations are the same.
    {
        fs::create_symlink(cwd, root / "cwdlink");
        const std::vector<std::string> same{"", ".", "./", cwd.string(), cwd.string() + "/", (root / "cwdlink").string(), "../cwd"};
        for (const auto& s : same) {
            writeOldResidues(E, residueA);
            core::ProofSetMarin set(E, 2, {}, core::ProofLocation(s));
            std::string note;
            expect(!set.adoptLegacyResidues(100, note) && note.empty(),
                   ("-f '" + s + "': nothing to adopt").c_str());
            expect(set.load(48) == residueA, ("-f '" + s + "': residues found").c_str());
            core::ProofSetMarin::clearResidues(set.location(), E);
            expect(!fs::exists("191"), ("-f '" + s + "': cleared").c_str());
        }
    }

    fs::current_path(oldCwd);
    std::error_code ec;
    fs::permissions(root / "ro", fs::perms::owner_all, ec);
    fs::remove_all(root, ec);

    if (failures) return 1;
    std::cout << "Proof location regression: PASS\n";
    return 0;
}
