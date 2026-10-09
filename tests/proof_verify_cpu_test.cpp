// Host test: the CPU (GMP) verification of the proof the CPU fallback writes.
//
// Real PRP residues 3^(2^k) mod 2^E - 1 are saved at the proof points, the proof is made by
// ProofSetMarin::computeProof (the CPU fallback), written and read back, and must verify. A proof
// with a flipped bit in any residue, a header of another exponent or power, or a truncated file
// must not.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
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

bool contains(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }

// The policy that decides how the CPU fallback proof is verified: the GPU when it works, the CPU
// when that is cheap, else not at all (with a warning).
void testPolicy() {
    using Status = core::FallbackVerifyResult::Status;
    const uint32_t E = 11213, power = 6;
    const core::ProofMarin proof = makeProof(E, power);
    const fs::path f = "policy.proof";
    proof.save(f);
    const double cap = core::kDefaultCpuVerifyMaxSeconds;

    // The estimate grows with E, shrinks with the power, and puts the costs where the fallback has them.
    expect(core::estimateCpuVerifySeconds(E, power) < 1.0, "policy: M11213 is estimated under a second");
    expect(core::estimateCpuVerifySeconds(E, power) <= cap, "policy: M11213 is under the default cap");
    expect(core::estimateCpuVerifySeconds(1000003, 8) < cap, "policy: M1M is under the default cap");
    expect(core::estimateCpuVerifySeconds(10000019, 8) > cap, "policy: M10M is over the default cap");
    expect(core::estimateCpuVerifySeconds(100000007, 9) > 3600.0, "policy: M100M takes hours");
    expect(core::estimateCpuVerifySeconds(20000003, 8) > core::estimateCpuVerifySeconds(10000019, 8),
           "policy: the estimate grows with E");
    expect(core::estimateCpuVerifySeconds(10000019, 9) < core::estimateCpuVerifySeconds(10000019, 7),
           "policy: the estimate shrinks with the power");
    expect(core::formatDuration(45.0) == "45 s" && core::formatDuration(1560.0) == "26 min" &&
               core::formatDuration(45000.0) == "12.5 h",
           "policy: durations are formatted");

    // The cap: the default, or the environment variable when it is a number >= 0.
    unsetenv(core::kCpuVerifyMaxSecondsEnv);
    expect(core::cpuVerifyMaxSeconds() == core::kDefaultCpuVerifyMaxSeconds, "policy: default cap");
    setenv(core::kCpuVerifyMaxSecondsEnv, "7.5", 1);
    expect(core::cpuVerifyMaxSeconds() == 7.5, "policy: cap from the environment");
    setenv(core::kCpuVerifyMaxSecondsEnv, "0", 1);
    expect(core::cpuVerifyMaxSeconds() == 0.0, "policy: cap 0 from the environment");
    for (const char* bad : {"-1", "abc", "", "5x", "nan", "inf"}) {
        setenv(core::kCpuVerifyMaxSecondsEnv, bad, 1);
        expect(core::cpuVerifyMaxSeconds() == core::kDefaultCpuVerifyMaxSeconds,
               std::string("policy: invalid cap '") + bad + "' ignored");
    }
    unsetenv(core::kCpuVerifyMaxSecondsEnv);

    int gpuCalls = 0;
    auto run = [&](const core::GpuProofVerifier& gpu, uint32_t e, uint32_t pw, double cpuCap, std::string& log,
                   const fs::path& file) {
        std::ostringstream out;
        auto r = core::verifyFallbackProof(file, e, pw, gpu, cpuCap, &out);
        log = out.str();
        return r;
    };
    std::string log;

    // GPU available: the GPU path, not the CPU one, whatever the size.
    gpuCalls = 0;
    auto gpuOk = [&](const fs::path& p) { ++gpuCalls; return p == f; };
    auto r = run(gpuOk, E, power, cap, log, f);
    expect(r.status == Status::Verified && r.method == "GPU" && gpuCalls == 1, "policy: GPU available -> GPU verification");
    expect(!contains(log, "CPU proof verification"), "policy: GPU available -> no CPU verification");
    r = run(gpuOk, E, power, 0.0, log, f);
    expect(r.status == Status::Verified && r.method == "GPU", "policy: GPU is used even when the CPU cap is 0");

    // GPU says the proof is bad: failed, without a second opinion from the CPU.
    gpuCalls = 0;
    r = run([&](const fs::path&) { ++gpuCalls; return false; }, E, power, cap, log, f);
    expect(r.status == Status::Failed && r.method == "GPU" && gpuCalls == 1 && !r.message.empty(),
           "policy: GPU rejects the proof -> failed");
    expect(!contains(log, "CPU proof verification"), "policy: a GPU rejection is not rechecked on the CPU");

    // A proof for another exponent or power fails before the GPU sees it.
    gpuCalls = 0;
    r = run(gpuOk, E, power + 1, cap, log, f);
    expect(r.status == Status::Failed && gpuCalls == 0, "policy: wrong power fails before the GPU check");
    r = run(gpuOk, E + 2, power, cap, log, f);
    expect(r.status == Status::Failed && gpuCalls == 0, "policy: wrong exponent fails before the GPU check");
    r = run(gpuOk, E, power, cap, log, "no-such.proof");
    expect(r.status == Status::Failed && gpuCalls == 0, "policy: unreadable proof fails before the GPU check");

    // GPU unusable (throws) and small E: verified on the CPU.
    auto gpuDown = [&](const fs::path&) -> bool { ++gpuCalls; throw std::runtime_error("device lost"); };
    gpuCalls = 0;
    r = run(gpuDown, E, power, cap, log, f);
    expect(r.status == Status::Verified && r.method == "CPU" && gpuCalls == 1, "policy: GPU unusable, small E -> CPU verification");
    expect(contains(log, "device lost") && contains(log, "CPU proof verification: SUCCESS"),
           "policy: the CPU verification says why the GPU was not used");
    // ... and no GPU at all.
    r = run(core::GpuProofVerifier(), E, power, cap, log, f);
    expect(r.status == Status::Verified && r.method == "CPU", "policy: no GPU, small E -> CPU verification");
    // ... and a bad proof is caught there.
    {
        auto B = proof.B;
        B[0] ^= 1u;
        core::ProofMarin(E, B, proof.middles).save("bad.proof");
        r = run(gpuDown, E, power, cap, log, "bad.proof");
        expect(r.status == Status::Failed && r.method == "CPU" && !r.message.empty(), "policy: CPU verification rejects a bad proof");
        fs::remove("bad.proof");
    }

    // GPU unusable and a CPU check over the cap (a large E, or a cap of 0): skipped with a warning,
    // and the CPU does not run. The proof is a well-formed fake: it is not checked.
    {
        const uint32_t bigE = 3000017, bigPower = 4;
        expect(core::estimateCpuVerifySeconds(bigE, bigPower) > cap, "policy: the large E is over the cap");
        auto fake = [&](uint32_t seed) {
            std::vector<uint32_t> w((bigE + 31) / 32);
            for (auto& x : w) x = (seed = seed * 1664525u + 1013904223u) | 1u;
            w.back() &= (1u << (bigE & 31u)) - 1u;
            return w;
        };
        std::vector<std::vector<uint32_t>> middles;
        for (uint32_t i = 0; i < bigPower; ++i) middles.push_back(fake(i + 1));
        core::ProofMarin(bigE, fake(99), middles).save("big.proof");
        gpuCalls = 0;
        r = run(gpuDown, bigE, bigPower, cap, log, "big.proof");
        expect(r.status == Status::Skipped && gpuCalls == 1, "policy: GPU unusable, large E -> skipped");
        expect(contains(r.message, "CPU fallback proof for M3000017 was not verified") &&
                   contains(r.message, "GPU unavailable: device lost") && contains(r.message, "CPU verification would take about ") &&
                   contains(r.message, "the proof is kept") && contains(r.message, core::kCpuVerifyMaxSecondsEnv),
               "policy: the skip warning says what was not done, why, the cost and the setting: " + r.message);
        expect(!contains(log, "CPU proof verification"), "policy: a skipped verification does no CPU work");
        r = run(core::GpuProofVerifier(), bigE, bigPower, cap, log, "big.proof");
        expect(r.status == Status::Skipped && contains(r.message, "no GPU verifier"), "policy: no GPU, large E -> skipped");
        // The same large proof with a working GPU goes to the GPU.
        r = run([&](const fs::path&) { return true; }, bigE, bigPower, cap, log, "big.proof");
        expect(r.status == Status::Verified && r.method == "GPU", "policy: GPU available, large E -> GPU verification");
        fs::remove("big.proof");
    }
    // Hours, for the largest exponents, in the warning.
    expect(contains(core::formatDuration(core::estimateCpuVerifySeconds(100000007, 9)), " h"),
           "policy: M100M is reported in hours");
    r = run(gpuDown, E, power, 0.0, log, f);
    expect(r.status == Status::Skipped && contains(r.message, "disabled by") && !contains(r.message, "over the"),
           "policy: a cap of 0 skips the CPU verification");
    // The cap is inclusive of the estimate.
    r = run(gpuDown, E, power, core::estimateCpuVerifySeconds(E, power), log, f);
    expect(r.status == Status::Verified && r.method == "CPU", "policy: a CPU check at the cap is done");

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

    testPolicy();

    fs::current_path(oldCwd);
    fs::remove_all(dir);
    std::cout << "CPU proof verification test: " << (failures ? "FAIL" : "PASS") << std::endl;
    return failures ? 1 : 0;
}
