// Host test for the proof power usable when a test is resumed.
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

void expectPower(uint32_t E, uint32_t power, uint32_t k, uint32_t want) {
    const uint32_t got = core::ProofSetMarin::effectivePower(E, power, k);
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
    fs::remove(core::ProofSetMarin::proofPath(E) / "168");
    expectPower(E, 3, 100, 3);
    expectPower(E, 3, 168, 2);  // power 3 needs 168 once resumed at 168
    expectPower(E, 3, 190, 2);  // power 2 does not use 168

    // 72 is only a power-3 point.
    fs::remove(core::ProofSetMarin::proofPath(E) / "72");
    expectPower(E, 3, 100, 2);
    expectPower(E, 3, 71, 3);

    // 48 is a power-2 point; power 1 only needs 96.
    fs::remove(core::ProofSetMarin::proofPath(E) / "48");
    expectPower(E, 3, 100, 1);
    expectPower(E, 2, 100, 1);
    expectPower(E, 3, 47, 3);

    // 96 is needed by every power.
    fs::remove(core::ProofSetMarin::proofPath(E) / "96");
    expectPower(E, 3, 100, 0);
    expectPower(E, 3, 95, 1);  // power 1 needs nothing before 96

    // A residue of the wrong size is as good as missing.
    {
        std::ofstream f(core::ProofSetMarin::proofPath(E) / "96", std::ios::binary);
        f << "short";
    }
    expectPower(E, 1, 100, 0);

    // A full-size residue with a bad CRC is rejected when it is the newest.
    set.save(96, residue);
    expectPower(E, 1, 100, 1);
    {
        std::fstream f(core::ProofSetMarin::proofPath(E) / "96",
                       std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(8);
        f.put('\x55');
    }
    expectPower(E, 1, 100, 0);

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
