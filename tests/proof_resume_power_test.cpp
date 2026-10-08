// Host test for the proof power usable when a test is resumed.
#include <algorithm>
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
const core::ProofLocation here; // the working directory: no save path

void expectPower(uint32_t E, uint32_t power, uint32_t k, uint32_t want) {
    const uint32_t got = core::ProofSetMarin::effectivePower(here, E, power, k);
    if (got != want) {
        std::cerr << "FAIL: effectivePower(E=" << E << ", power=" << power
                  << ", k=" << k << ") = " << got << ", want " << want << "\n";
        ++failures;
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto dir = fs::temp_directory_path() /
                     ("prmers-proof-resume-" + std::to_string(stamp));
    fs::create_directories(dir);
    const auto oldCwd = fs::current_path();
    fs::current_path(dir);

    // E = 191, power 3: points 24 48 72 96 120 144 168 191.
    // Power 2 uses 48 96 144 191, power 1 uses 96 191.
    constexpr uint32_t E = 191;
    const std::vector<uint32_t> p3 = core::ProofSetMarin::proofPoints(E, 3);
    if (p3 != std::vector<uint32_t>{24, 48, 72, 96, 120, 144, 168, 191}) {
        std::cerr << "FAIL: proofPoints(191, 3)\n";
        ++failures;
    }

    core::ProofSetMarin set(E, 3);
    const std::vector<uint32_t> residue{1u, 2u, 3u, 4u, 5u, 6u};
    for (uint32_t k : p3) {
        if (k < E) set.save(k, residue);
    }

    // Nothing needed yet before the first point; everything present.
    expectPower(E, 3, 0, 3);
    expectPower(E, 3, 100, 3);
    expectPower(E, 3, 190, 3);

    // A point after the resume iteration is not needed yet.
    fs::remove(core::ProofSetMarin::proofPath(here, E) / "168");
    expectPower(E, 3, 100, 3);
    expectPower(E, 3, 168, 2);  // power 3 needs 168 once resumed at 168
    expectPower(E, 3, 190, 2);  // power 2 does not use 168

    // 72 is only a power-3 point.
    fs::remove(core::ProofSetMarin::proofPath(here, E) / "72");
    expectPower(E, 3, 100, 2);
    expectPower(E, 3, 71, 3);

    // 48 is a power-2 point; power 1 only needs 96.
    fs::remove(core::ProofSetMarin::proofPath(here, E) / "48");
    expectPower(E, 3, 100, 1);
    expectPower(E, 2, 100, 1);
    expectPower(E, 3, 47, 3);

    // 96 is needed by every power.
    fs::remove(core::ProofSetMarin::proofPath(here, E) / "96");
    expectPower(E, 3, 100, 0);
    expectPower(E, 3, 95, 1);  // power 1 needs nothing before 96

    // A residue of the wrong size is as good as missing.
    {
        std::ofstream f(core::ProofSetMarin::proofPath(here, E) / "96", std::ios::binary);
        f << "short";
    }
    expectPower(E, 1, 100, 0);

    // A full-size residue with a bad CRC is rejected when it is the newest.
    set.save(96, residue);
    expectPower(E, 1, 100, 1);
    {
        std::fstream f(core::ProofSetMarin::proofPath(here, E) / "96",
                       std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(8);
        f.put('\x55');
    }
    expectPower(E, 1, 100, 0);

    // Every required residue is CRC-checked, not only the newest: an older
    // one damaged in place (same size) rules out the powers that need it.
    auto saveAll = [&]() {
        for (uint32_t k : p3) {
            if (k < E) set.save(k, residue);
        }
    };
    auto corrupt = [&](uint32_t k) {
        std::fstream f(core::ProofSetMarin::proofPath(here, E) / std::to_string(k),
                       std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(8);
        f.put('\x55');
    };
    saveAll();
    expectPower(E, 3, 190, 3);
    corrupt(24);                // power-3 point, older than the newest (168)
    expectPower(E, 3, 190, 2);
    expectPower(E, 3, 23, 3);   // not needed yet before iteration 24
    expectPower(E, 3, 24, 2);
    saveAll();
    corrupt(48);                // power-2 point
    expectPower(E, 3, 190, 1);
    saveAll();
    corrupt(96);                // needed by every power
    expectPower(E, 3, 190, 0);
    expectPower(E, 3, 95, 3);
    saveAll();
    corrupt(168);               // not needed by a test resumed before it
    expectPower(E, 3, 167, 3);
    expectPower(E, 3, 190, 2);

    // Boundary resume iterations: 0 needs nothing; 2^32-1 and E need every
    // point below E.
    saveAll();
    expectPower(E, 3, 0, 3);
    expectPower(E, 3, 0xFFFFFFFFu, 3);
    expectPower(E, 3, E, 3);
    corrupt(120);
    expectPower(E, 3, 0xFFFFFFFFu, 2);
    expectPower(E, 3, 0, 3);

    // An empty file or a directory in place of a residue counts as missing.
    saveAll();
    {
        std::ofstream f(core::ProofSetMarin::proofPath(here, E) / "72", std::ios::binary | std::ios::trunc);
    }
    expectPower(E, 3, 190, 2);
    fs::remove(core::ProofSetMarin::proofPath(here, E) / "72");
    fs::create_directory(core::ProofSetMarin::proofPath(here, E) / "72");
    expectPower(E, 3, 190, 2);
    fs::remove(core::ProofSetMarin::proofPath(here, E) / "72");

    // A residue one word too long is rejected too.
    saveAll();
    {
        std::ofstream f(core::ProofSetMarin::proofPath(here, E) / "144", std::ios::binary | std::ios::app);
        f.write("\0\0\0\0", 4);
    }
    expectPower(E, 3, 190, 1);  // 144 is a power-2 and power-3 point
    saveAll();

    // The points of a lower power are a subset of those of a higher one
    // (effectivePower and setPower rely on it).
    for (uint32_t e : {3u, 5u, 7u, 191u, 1279u, 11213u, 82589933u}) {
        for (uint32_t pw = 2; pw <= 10; ++pw) {
            const auto hi = core::ProofSetMarin::proofPoints(e, pw);
            for (uint32_t k : core::ProofSetMarin::proofPoints(e, pw - 1)) {
                if (!std::binary_search(hi.begin(), hi.end(), k)) {
                    std::cerr << "FAIL: point " << k << " of power " << pw - 1
                              << " is not a point of power " << pw << " (E=" << e << ")\n";
                    ++failures;
                }
            }
        }
    }

    // setPower lowers the points residues are saved for.
    set.setPower(1);
    if (!set.shouldCheckpoint(96) || set.shouldCheckpoint(48) || set.power != 1) {
        std::cerr << "FAIL: setPower(1)\n";
        ++failures;
    }

    fs::current_path(oldCwd);
    std::error_code ec;
    fs::remove_all(dir, ec);

    if (failures) return 1;
    std::cout << "Proof resume power regression: PASS\n";
    return 0;
}
