// Host test: the CPU (GMP) verification of the proof the CPU fallback writes.
//
// Real PRP residues 3^(2^k) mod 2^E - 1 are saved at the proof points, the proof is made by
// ProofSetMarin::computeProof (the CPU fallback), written and read back, and must verify. A proof
// with a flipped bit in any residue, a header of another exponent or power, or a truncated file
// must not.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <gmpxx.h>

#include "core/ProofMarin.hpp"
#include "core/ProofSetMarin.hpp"
#include "core/ProofVerifyCpu.hpp"
#include "util/GmpUtils.hpp"

namespace fs = std::filesystem;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

std::vector<uint32_t> words(const mpz_class& x, uint32_t E) {
    auto w = util::convertFromGMP(x);
    w.resize((E + 31) / 32, 0u);
    return w;
}

// The proof of a PRP of M<E> at this power, made as the CPU fallback makes it.
core::ProofMarin makeProof(uint32_t E, uint32_t power) {
    core::ProofSetMarin set(E, power);
    mpz_class mp = 1;
    mp <<= E;
    mp -= 1;
    mpz_class x = 3;
    for (uint32_t k = 1; k <= E; ++k) {
        x = x * x % mp;
        if (set.shouldCheckpoint(k)) set.save(k, words(x, E));
    }
    return set.computeProof();
}

bool verifies(const core::ProofMarin& p, uint32_t E, uint32_t power, std::string* whyOut = nullptr) {
    std::string why;
    const bool ok = core::verifyProofCpu(p, E, power, why, nullptr);
    if (whyOut) *whyOut = why;
    return ok;
}

std::string readFile(const fs::path& f) {
    std::ifstream in(f, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void writeFile(const fs::path& f, const std::string& s) {
    std::ofstream o(f, std::ios::binary | std::ios::trunc);
    o << s;
}

// Load a proof file and verify it; a file that does not load does not verify.
bool fileVerifies(const fs::path& f, uint32_t E, uint32_t power) {
    try {
        return verifies(core::ProofMarin::load(f), E, power);
    } catch (const std::exception&) {
        return false;
    }
}

void testExponent(uint32_t E, uint32_t power) {
    const std::string tag = "M" + std::to_string(E) + " power " + std::to_string(power);
    const core::ProofMarin proof = makeProof(E, power);
    std::string why;
    expect(verifies(proof, E, power, &why), tag + ": a correct proof verifies (" + why + ")");

    const fs::path f = "p-" + std::to_string(E) + "-" + std::to_string(power) + ".proof";
    proof.save(f);
    expect(fileVerifies(f, E, power), tag + ": a correct proof file verifies");

    // a flipped bit in the final residue, and in each middle
    {
        auto B = proof.B;
        B[B.size() / 2] ^= 1u << 7;
        expect(!verifies(core::ProofMarin(E, B, proof.middles), E, power), tag + ": flipped bit in B rejected");
    }
    for (uint32_t i = 0; i < power; ++i) {
        auto middles = proof.middles;
        middles[i][0] ^= 1u;
        expect(!verifies(core::ProofMarin(E, proof.B, middles), E, power),
               tag + ": flipped bit in middle " + std::to_string(i) + " rejected");
    }
    // a flipped bit in the file, in each residue
    {
        const std::string good = readFile(f);
        const size_t nBytes = (E - 1) / 8 + 1;
        const size_t header = good.size() - nBytes * (power + 1);
        for (uint32_t r = 0; r <= power; ++r) {
            std::string bad = good;
            bad[header + r * nBytes + nBytes / 3] ^= 0x10;
            writeFile(f, bad);
            expect(!fileVerifies(f, E, power), tag + ": flipped bit in residue " + std::to_string(r) + " of the file rejected");
        }
        // truncated
        for (size_t cut : {size_t(1), nBytes / 2, nBytes, good.size() - header}) {
            writeFile(f, good.substr(0, good.size() - cut));
            expect(!fileVerifies(f, E, power), tag + ": truncated by " + std::to_string(cut) + " bytes rejected");
        }
        // header of another exponent or power
        auto edit = [&](const std::string& from, const std::string& to) {
            std::string s = good;
            s.replace(s.find(from), from.size(), to);
            return s;
        };
        writeFile(f, edit("NUMBER=M" + std::to_string(E), "NUMBER=M" + std::to_string(E + 2)));
        expect(!fileVerifies(f, E, power), tag + ": header of another exponent rejected");
        expect(!fileVerifies(f, E + 2, power), tag + ": header of another exponent rejected, as that exponent");
        if (power > 1) {
            writeFile(f, edit("POWER=" + std::to_string(power), "POWER=" + std::to_string(power - 1)));
            expect(!fileVerifies(f, E, power), tag + ": header of a lower power rejected");
            // Read at that lower power the file is the first power-1 middles of this proof, which is
            // a valid proof of power-1: the fallback always verifies at the power it made the proof for.
            expect(fileVerifies(f, E, power - 1), tag + ": the first middles are a proof of the lower power");
        }
        writeFile(f, good);
        expect(fileVerifies(f, E, power), tag + ": restored file verifies");
    }
    // verified as another exponent or power
    expect(!verifies(proof, E + 2, power), tag + ": wrong expected exponent rejected");
    expect(!verifies(proof, E, power + 1), tag + ": wrong expected power rejected");
    // a residue with bits above E
    if (E % 32 != 0) {
        auto B = proof.B;
        B.back() |= 1u << 31;
        expect(!verifies(core::ProofMarin(E, B, proof.middles), E, power), tag + ": residue wider than E rejected");
    }
    fs::remove(f);
}

} // namespace

int main() {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const fs::path dir = fs::temp_directory_path() / ("prmers-proof-verify-cpu-" + std::to_string(stamp));
    fs::create_directories(dir);
    const fs::path oldCwd = fs::current_path();
    fs::current_path(dir);

    testExponent(191, 1);       // composite M191
    testExponent(191, 3);
    testExponent(4423, 5);      // M4423 is prime
    testExponent(4441, 2);      // odd spans
    testExponent(9689, 8);
    testExponent(11213, 6);

    fs::current_path(oldCwd);
    fs::remove_all(dir);
    std::cout << "CPU proof verification test: " << (failures ? "FAIL" : "PASS") << std::endl;
    return failures ? 1 : 0;
}
