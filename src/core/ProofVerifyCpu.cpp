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
#include <iomanip>
#include <ostream>

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

bool verifyProofCpu(const ProofMarin& proof, uint32_t expectedE, uint32_t expectedPower,
                    std::string& why, std::ostream* log) {
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

} // namespace core
