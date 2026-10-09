// core/ProofVerifyCpu.cpp
//
// CPU (GMP) verification of a PRP proof, the same check as Proof::verify:
//   A = 3, B = final residue, h_0 = hash(B)
//   for each middle M_i: h_i = hash(h_{i-1}, M_i)
//                        B = M_i^h_i * (span odd ? B^2 : B)
//                        A = A^h_i * M_i
//                        span = ceil(span / 2)
//   valid when A^(2^span) == B (mod 2^E - 1).
#include "core/ProofVerifyCpu.hpp"

#include "util/GmpUtils.hpp"

#include <gmpxx.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace core {

namespace {

// x := x mod (2^E - 1), for 0 <= x < 2^(2E), fully reduced to [0, 2^E - 2].
void reduceInPlace(mpz_class& x, mpz_class& hi, const mpz_class& mp, uint32_t E) {
    while (mpz_sizeinbase(x.get_mpz_t(), 2) > E) {
        mpz_tdiv_q_2exp(hi.get_mpz_t(), x.get_mpz_t(), E);
        mpz_tdiv_r_2exp(x.get_mpz_t(), x.get_mpz_t(), E);
        mpz_add(x.get_mpz_t(), x.get_mpz_t(), hi.get_mpz_t());
    }
    if (x >= mp) x -= mp;
}

mpz_class mulMod(const mpz_class& a, const mpz_class& b, const mpz_class& mp, uint32_t E) {
    mpz_class r, hi;
    mpz_mul(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
    reduceInPlace(r, hi, mp, E);
    return r;
}

// A residue as the proof file holds it: exactly ceil(E/32) words, no bit at or above E.
bool wellFormed(const std::vector<uint32_t>& w, uint32_t E) {
    if (w.size() != (static_cast<size_t>(E) + 31u) / 32u) return false;
    if ((E & 31u) != 0u && (w.back() >> (E & 31u)) != 0u) return false;
    return true;
}

} // namespace

bool checkProofShape(const ProofMarin& proof, uint32_t expectedE, uint32_t expectedPower,
                     std::string& why) {
    const uint32_t E = proof.E;
    const uint32_t power = static_cast<uint32_t>(proof.middles.size());
    if (E != expectedE) {
        why = "the proof is for M" + std::to_string(E) + ", not M" + std::to_string(expectedE);
        return false;
    }
    if (power != expectedPower || power == 0) {
        why = "the proof has power " + std::to_string(power) + ", not " + std::to_string(expectedPower);
        return false;
    }
    if (E < 2) {
        why = "invalid exponent";
        return false;
    }
    if (!wellFormed(proof.B, E) || util::isZeroResidue(proof.B, E)) {
        why = "the final residue is malformed or zero";
        return false;
    }
    for (uint32_t i = 0; i < power; ++i) {
        if (!wellFormed(proof.middles[i], E) || util::isZeroResidue(proof.middles[i], E)) {
            why = "middle residue " + std::to_string(i) + " is malformed or zero";
            return false;
        }
    }
    return true;
}

bool verifyProofCpu(const ProofMarin& proof, uint32_t expectedE, uint32_t expectedPower,
                    std::string& why, std::ostream* log) {
    if (!checkProofShape(proof, expectedE, expectedPower, why)) return false;
    const uint32_t E = proof.E;
    const uint32_t power = static_cast<uint32_t>(proof.middles.size());

    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    auto lastLog = start;

    mpz_class mp = 1;
    mp <<= E;
    mp -= 1;

    mpz_class A = 3;
    mpz_class B = util::mersenneReduce(util::convertToGMP(proof.B), E);
    auto hash = ProofMarin::hashWords(E, proof.B);
    uint32_t span = E;

    if (log) *log << "Verifying the CPU proof of M" << E << " (power " << power << ") on the CPU" << std::endl;

    for (uint32_t i = 0; i < power; ++i, span = (span + 1) / 2) {
        const auto& Mw = proof.middles[i];
        hash = ProofMarin::hashWords(E, hash, Mw);
        const uint64_t h = hash[0];
        const mpz_class M = util::mersenneReduce(util::convertToGMP(Mw), E);

        if (span % 2 != 0) B = mulMod(B, B, mp, E);
        B = mulMod(util::mersennePowMod(M, h, E), B, mp, E);
        A = mulMod(util::mersennePowMod(A, h, E), M, mp, E);
    }

    // A := A^(2^span)
    mpz_class hi;
    for (uint32_t k = 0; k < span; ++k) {
        mpz_mul(A.get_mpz_t(), A.get_mpz_t(), A.get_mpz_t());
        reduceInPlace(A, hi, mp, E);
        if (log && (k & 255u) == 255u) {
            const auto now = clock::now();
            if (now - lastLog >= std::chrono::seconds(10)) {
                lastLog = now;
                const double el = std::chrono::duration<double>(now - start).count();
                const double done = static_cast<double>(k + 1) / span;
                *log << "CPU proof verification: " << (k + 1) << " / " << span << " squarings ("
                     << std::fixed << std::setprecision(1) << 100.0 * done << "%), ETA "
                     << std::setprecision(0) << el / done - el << " s" << std::defaultfloat << std::endl;
            }
        }
    }

    const bool ok = A == B;
    const double el = std::chrono::duration<double>(clock::now() - start).count();
    if (log)
        *log << "CPU proof verification: " << (ok ? "SUCCESS" : "FAIL") << " in " << std::fixed
             << std::setprecision(2) << el << " s" << std::defaultfloat << std::endl;
    if (!ok) why = "the proof does not verify";
    return ok;
}

namespace {

// Squaring-equivalents of one modular exponentiation by a 64-bit hash: 64
// squarings and about 32 multiplications.
constexpr double kExponentiationSquarings = 96.0;
// Seconds of one squaring mod 2^E - 1 at E = 1M with GMP, and its growth with E.
constexpr double kSquaringSecondsAt1M = 2.4e-3;
constexpr double kSquaringSizeExponent = 1.17;

} // namespace

double cpuVerifyMaxSeconds() {
    const char* v = std::getenv(kCpuVerifyMaxSecondsEnv);
    if (v && *v) {
        char* end = nullptr;
        const double x = std::strtod(v, &end);
        if (end != v && *end == '\0' && std::isfinite(x) && x >= 0.0) return x;
    }
    return kDefaultCpuVerifyMaxSeconds;
}

double estimateCpuVerifySeconds(uint32_t E, uint32_t power) {
    const double span = std::ceil(static_cast<double>(E) / std::ldexp(1.0, static_cast<int>(power > 62 ? 62 : power)));
    const double ops = span + 2.0 * kExponentiationSquarings * power + 4.0 * power;
    const double perSquaring = kSquaringSecondsAt1M * std::pow(static_cast<double>(E) / 1.0e6, kSquaringSizeExponent);
    return ops * perSquaring;
}

std::string formatDuration(double seconds) {
    std::ostringstream o;
    o << std::fixed;
    if (seconds < 1.0) {
        o << "under 1 s";
    } else if (seconds < 90.0) {
        o << std::setprecision(0) << seconds << " s";
    } else if (seconds < 5400.0) {
        o << std::setprecision(0) << seconds / 60.0 << " min";
    } else if (seconds < 172800.0) {
        o << std::setprecision(1) << seconds / 3600.0 << " h";
    } else {
        o << std::setprecision(1) << seconds / 86400.0 << " days";
    }
    return o.str();
}

FallbackVerifyResult verifyFallbackProof(const std::filesystem::path& file, uint32_t E, uint32_t power,
                                         const GpuProofVerifier& gpuVerify, double cpuMaxSeconds,
                                         std::ostream* log) {
    using Status = FallbackVerifyResult::Status;
    FallbackVerifyResult r;
    r.estimatedCpuSeconds = estimateCpuVerifySeconds(E, power);

    std::string gpuUnavailable = "no GPU verifier";
    if (gpuVerify) {
        // A proof of another exponent or power, or with a malformed residue, fails whatever
        // the method (and must not reach the GPU check, which trusts the header).
        try {
            std::string why;
            if (!checkProofShape(ProofMarin::load(file), E, power, why)) {
                r.status = Status::Failed;
                r.message = why;
                return r;
            }
        } catch (const std::exception& e) {
            r.status = Status::Failed;
            r.message = e.what();
            return r;
        }
        try {
            if (log) *log << "Verifying the CPU proof of M" << E << " (power " << power << ") on the GPU" << std::endl;
            const bool ok = gpuVerify(file);
            r.method = "GPU";
            r.status = ok ? Status::Verified : Status::Failed;
            if (!ok) r.message = "the proof does not verify (GPU verification)";
            return r;
        } catch (const std::exception& e) {
            gpuUnavailable = e.what();
            if (log)
                *log << "GPU verification of the CPU proof is unavailable (" << gpuUnavailable << ")" << std::endl;
        }
    }

    if (r.estimatedCpuSeconds <= cpuMaxSeconds) {
        r.method = "CPU";
        try {
            const ProofMarin proof = ProofMarin::load(file);
            std::string why;
            const bool ok = verifyProofCpu(proof, E, power, why, log);
            r.status = ok ? Status::Verified : Status::Failed;
            if (!ok) r.message = why;
        } catch (const std::exception& e) {
            r.status = Status::Failed;
            r.message = e.what();
        }
        return r;
    }

    r.status = Status::Skipped;
    r.message = "CPU fallback proof for M" + std::to_string(E) + " was not verified (GPU unavailable: " +
                gpuUnavailable + "; CPU verification would take about " + formatDuration(r.estimatedCpuSeconds) + ", ";
    if (cpuMaxSeconds <= 0.0)
        r.message += std::string("and it is disabled by ") + kCpuVerifyMaxSecondsEnv + "=0";
    else
        r.message += "over the " + formatDuration(cpuMaxSeconds) + " limit set by " + kCpuVerifyMaxSecondsEnv;
    r.message += "); the proof is kept";
    return r;
}

} // namespace core
