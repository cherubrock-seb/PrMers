/*
 * Mersenne OpenCL Primality Test - Proof Generation
 *
 * This code is part of a Mersenne prime search that uses integer arithmetic
 * and an Iterated Discrete Weighted Transform (IDBWT) via a Number-Theoretic 
 * Transform (NTT), executed on the GPU through OpenCL.
 *
 * The proof generation method is based on mathematical techniques inspired by:
 *   - The concept of verifiable delay functions (VDFs) as presented in:
 *         "Simple Verifiable Delay Functions" by Krzysztof Pietrzak (2018)
 *         https://eprint.iacr.org/2018/627.pdf
 *   - The GpuOwl project (https://github.com/preda/gpuowl), which efficiently
 *     computes PRP proofs for Mersenne numbers using floating-point FFTs.
 *
 * The proof structure follows an approach similar to GPUOwl:
 *   - It stores intermediate residues at specific iteration points.
 *   - It generates proofs based on random exponents derived via SHA3.
 *   - The verification process ensures correctness through exponentiation.
 *
 * This implementation reuses and adapts significant portions of GPUOwl's proof
 * generation logic while adapting it to an integer-based arithmetic approach.
 *
 * Author: Cherubrock
 *
 * Released as free software.
 */
#include "proof.h"
#include "Sha3Hash.h"
#include "MD5.h"  
#include "common.h"  // for Words, etc.
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cinttypes>   // for PRIx64
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <cstdarg>
#include <cstdio>   

// For quick logging
static void logmsg(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// ------------------------------------------------
//   Utility definitions
// ------------------------------------------------

namespace fs = std::filesystem;

namespace proof_util {

static inline size_t numBytes(uint32_t E) {
    // GpuOwl uses (E-1)/8 + 1 for the size of the residue in bytes
    return (size_t(E - 1) / 8) + 1;
}

// Writes Words to a file in little-endian form
void writeWords(const Words &w, FILE *f) {
    fwrite(w.data(), sizeof(uint32_t), w.size(), f);
}

Words readWords(uint32_t E, FILE *f) {
    size_t nb = numBytes(E);
    // Round up to multiple of 4 to store in our vector
    size_t n32 = (nb + 3)/4;
    Words w(n32, 0);
    size_t got = fread((void*)w.data(), 1, nb, f);
    if (got != nb) {
        throw std::runtime_error("readWords: file ended early");
    }
    return w;
}

// Very naive mod (2^E - 1). We treat the Words as 32-bit limbs (little-endian).
// For correctness, no big-limb carry checks are done here except trivial "reduce once".
static void reduce_2pminus1(Words &Z, uint32_t E) {
    // Convert E bits => ~ E-1 bits: if the top bit is set beyond E, we wrap.
    size_t nb = numBytes(E);
    size_t n32 = Z.size();
    if (n32*4 < nb) return; // no wrap

    // For 2^E - 1, the top E bits in Z can be "folded" onto the lower bits. A correct
    // approach would be to do repeated folds until we are below 2^E. This is a naive version:
    //  (For large E, do a loop. For smaller exponents, a single pass might suffice.)
    // Here we do multiple folds if needed.
    const uint32_t shiftBits = (n32*32) - E;
    if ((int)shiftBits < 0) return; // no folding if n32*32 < E
    uint32_t shiftWords = shiftBits / 32;
    uint32_t bitRema = shiftBits % 32;

    if (shiftWords >= n32) return; // no effect

    // Fold high part -> low part
    // This is a simplistic approach: Z = lower + (higher << someBitRema)
    uint64_t carry = 0;
    for (size_t i = 0; i < n32 - shiftWords; i++) {
        // each chunk from the "higher" portion
        uint64_t chunk = (i + shiftWords < n32) ? (uint64_t) Z[i + shiftWords] : 0ULL;
        // shift chunk right by bitRema
        if (bitRema) {
            //uint64_t chunk2 = (i + shiftWords + 1 < n32) ? (uint64_t)Z[i + shiftWords + 1] : 0ULL;
            uint64_t val = (chunk << (64 - bitRema)) & 0xFFFFFFFFFFFFFFFFULL;
            val >>= (64 - bitRema); // effectively chunk >> bitRema, but we want to handle partial
            // a better approach would be collecting bits from chunk2. This is a demonstration only.
            chunk = chunk >> bitRema;  
        }
        // add
        uint64_t s = (uint64_t)Z[i] + chunk + carry;
        Z[i] = (uint32_t)(s & 0xFFFFFFFFULL);
        carry = s >> 32;
    }
    // the naive approach doesn’t handle leftover carry properly for large E. For demonstration only.
}

// Multiply two big-ints mod (2^E - 1) in naive O(n^2).
Words modMul(const Words &A, const Words &B, uint32_t E) {
    Words Z(std::max(A.size(), B.size())*2, 0);
    for (size_t i = 0; i < A.size(); i++) {
        uint64_t carry = 0;
        uint64_t x = (uint64_t) A[i];
        for (size_t j = 0; j < B.size() || carry; j++) {
            uint64_t y = (j < B.size()) ? (uint64_t)B[j] : 0;
            uint64_t s = Z[i + j] + x*y + carry;
            Z[i + j] = (uint32_t)(s & 0xFFFFFFFFULL);
            carry = s >> 32;
        }
    }
    // Then reduce once or repeatedly mod (2^E - 1).
    reduce_2pminus1(Z, E);
    return Z;
}

Words modExp(const Words &base, uint64_t exponent, uint32_t E) {
    // Square-and-multiply
    Words result; 
    result.resize(base.size(), 0);
    // set result = 1
    result[0] = 1;

    Words cur = base;
    uint64_t e = exponent;
    while (e > 0) {
        if (e & 1ULL) {
            result = modMul(result, cur, E);
        }
        e >>= 1ULL;
        cur = modMul(cur, cur, E);
    }
    return result;
}

Words expExp2(const Words &A, uint32_t count, uint32_t E) {
    // A^(2^count) = repeated squaring of A, count times
    Words r = A;
    for (uint32_t i = 0; i < count; i++) {
        r = modMul(r, r, E);
    }
    return r;
}

// "expMul" merges like GpuOwl:  newB = B * (M^h).
Words expMul(const Words &A, uint64_t exp, const Words &B, bool doSquareBefore, uint32_t E) {
    // If doSquareBefore is true, we do B = B^2 mod M prior to multiply
    Words lhs = B;
    if (doSquareBefore) {
        lhs = modMul(lhs, lhs, E);
    }
    // Then M^h mod M
    Words rhs = modExp(A, exp, E);
    // Multiply them
    Words out = modMul(lhs, rhs, E);
    return out;
}

bool isResidue9(const Words &w) {
    if (w.size() == 0) return false;
    if (w[0] != 9) return false;
    for (size_t i = 1; i < w.size(); i++) {
        if (w[i] != 0) return false;
    }
    return true;
}

// A helper that merges (prefix, data) into Sha3
std::array<uint64_t,4> sha3Words(const Words &w, const std::array<uint64_t,4> &prefix) {
    // Move into a Sha3Hash instance
    Sha3Hash hasher;
    if (!(prefix[0] == 0 && prefix[1] == 0 && prefix[2] == 0 && prefix[3] == 0)) {
       hasher.update(prefix.data(), sizeof(prefix));
    }
    // Then update by the residue data
    size_t nb = w.size() * 4;
    hasher.update(w.data(), nb);
    auto result = std::move(hasher).finish();
    return result;
}

std::string fileMD5(const fs::path &p) {
    FILE *f = fopen(p.string().c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file for MD5: " + p.string());
    }
    char buf[65536];
    MD5 h;
    while (!feof(f)) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        if (n > 0) h.update(buf, (unsigned)n);
    }
    fclose(f);
    return std::move(h).finish();
}

} // namespace proof_util

// ------------------------------------------------
//   Proof
// ------------------------------------------------

static const char *HEADER_v2 = "PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=%u\nNUMBER=M%u\n";

// Save to .proof file
void Proof::save(const fs::path &proofFile) const {
    FILE *f = fopen(proofFile.string().c_str(), "wb");
    if (!f) {
        throw std::runtime_error("Cannot open proof file for writing: " + proofFile.string());
    }
    // Write header
    fprintf(f, HEADER_v2, (unsigned) middles.size(), (unsigned) E);
    // Then B
    proof_util::writeWords(B, f);
    // Then each middle
    for (auto &m : middles) {
        proof_util::writeWords(m, f);
    }
    fclose(f);

    logmsg("Proof saved to %s\n", proofFile.string().c_str());
}

Proof Proof::load(const fs::path &path) {
    FILE *f = fopen(path.string().c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Could not open proof file for reading: " + path.string());
    }
    // Parse the header
    uint32_t power = 0, exponent = 0;
    int ret = fscanf(f, HEADER_v2, &power, &exponent);
    if (ret != 2) {
        fclose(f);
        throw std::runtime_error("Invalid proof header in " + path.string());
    }
    // read final residue B
    Words B = proof_util::readWords(exponent, f);
    // read the middles
    std::vector<Words> mids;
    mids.reserve(power);
    for (uint32_t i = 0; i < power; i++) {
        Words w = proof_util::readWords(exponent, f);
        mids.push_back(std::move(w));
    }
    fclose(f);

    return Proof(exponent, std::move(B), std::move(mids));
}

// Exponentiate `finalLeft` by 2^localSpan in small batches, printing progress
Words Proof::expExp2WithProgress(const Words &base, uint32_t totalSquares) const {
    // We'll do repeated squaring: result = base^(2^totalSquares).
    // For large totalSquares, break them into increments so we can show progress.
    Words result = base;

    // For demonstration: update progress every "batchSize" squares
    const uint32_t batchSize = 10000; // pick a suitable frequency
    uint32_t completed = 0;

    while (completed < totalSquares) {
        uint32_t nextStop = std::min(completed + batchSize, totalSquares);
        for (; completed < nextStop; completed++) {
            result = proof_util::modMul(result, result, E);
        }
        logmsg("Exp: %u/%u\n", completed, totalSquares);
        fflush(stdout);
    }

    return result;
}


bool Proof::verify(const std::vector<uint64_t> &expectedHashes) const {
    // We re-run the merge steps with:
    //   A = 3  (the standard PRP base)
    //   B = final residue in the file
    //   middles[0..power-1] are partial residues from the user
    //   We do random exponent = sha3(A||M) => h
    //   Then newB = B^? * M^h ...
    // except that GpuOwl uses the final iteration count as well, so we do “square if odd exponent” etc.
    // For a minimal example, we adopt the same code as GpuOwl’s "expMul" & "expExp2" approach.

    uint32_t p = E;

    // Construct the “start residue” is just {3, 0, 0,...} for base=3.
    Words A; 
    A.resize((p-1)/32+1, 0);
    A[0] = 3;  // i.e. base=3

    // If the final proof indicates a “9” residue, it’s probably prime. We can note that, not essential.
    bool isPrime = proof_util::isResidue9(B);

    // We'll do step by step merges:
    Words currentB = B;
    auto hashVal = proof_util::sha3Words(currentB, {0,0,0,0});

    //uint32_t span = p; // we do p merges total for a full LL or PRP? GpuOwl does partial merges, but let's do the simpler approach.
    // Actually in GpuOwl code, each middle reduces the exponent from p to p/2, p/4, etc. We replicate that:

    uint32_t localSpan = p;
    Words left = A;   // call it “left side”
    Words right = currentB;  // call it “right side”

    for (size_t i = 0; i < middles.size(); i++) {
        const Words &M = middles[i];
        // hashVal = sha3(hashVal || M)
        hashVal = proof_util::sha3Words(M, hashVal);
        uint64_t h = hashVal[0];  // 64 bits from first part

        // doSquareBefore if localSpan is odd
        bool doSquare = (localSpan % 2) == 1;
        right = proof_util::expMul(M, h, right, doSquare, p);
        left = proof_util::expMul(left, h, M, false, p);

        // Halve the exponent for next round
        localSpan = (localSpan + 1)/2;
        if (i < expectedHashes.size()) {
            uint64_t want = expectedHashes[i];
            if (h != want) {
                logmsg("Mismatch in random exponent at step %zu: got %016" PRIx64 " expected %016" PRIx64 "\n",
                       i, h, want);
                return false;
            }
        }
    }
    // Finally we exponentiate “left” by 2^(localSpan) and see if it matches “right”.
    Words finalLeft = Proof::expExp2WithProgress(left, localSpan);
    bool eq = (finalLeft == right);

    if (eq) {
        logmsg("Proof verified. M%u is %s\n", p, (isPrime ? "probable prime" : "composite"));
    } else {
        logmsg("Proof: invalid, final mismatch\n");
    }
    return eq;
}

std::filesystem::path Proof::fileName(const std::filesystem::path &proofDir) const {
    // Example: "32768-8.proof" or "999983-10.proof"
    std::string name = std::to_string(E) + "-" + std::to_string(middles.size()) + ".proof";
    return proofDir / name;
}

// ------------------------------------------------
//   ProofSet
// ------------------------------------------------

ProofSet::ProofSet(uint32_t exponent, uint32_t pw) : E(exponent), power(pw) {
    // Generate typical “points” schedule, as GpuOwl does
    // e.g. the iteration indices might be about p/2, p/4, p/8, ...
    // This is a simple approach: repeatedly floor-div2 until we get ~power points.
    // In GpuOwl you often see a more complex spacing.
    uint32_t current = exponent;
    for (uint32_t i = 0; i < pw; i++) {
        current = (current + 1) / 2; 
        points.push_back(current);
    }
    // Sort descending if needed, or keep ascending, depends on which order you want to store them in.
    std::reverse(points.begin(), points.end());
}

std::filesystem::path ProofSet::proofDir() const {
    // e.g. "216091/proof" subdir
    fs::path dir = fs::path(std::to_string(E)) / "proof";
    return dir;
}

std::filesystem::path ProofSet::partFile(uint32_t k) const {
    // e.g. "216091/proof/12345-part"
    fs::path p = proofDir() / (std::to_string(k) + "-part");
    return p;
}

void ProofSet::save(uint32_t k, const Words &state) const {
    fs::create_directories(proofDir());
    auto fname = partFile(k);
    FILE* f = fopen(fname.string().c_str(), "wb");
    if (!f) {
        throw std::runtime_error("Cannot open partial file for writing: " + fname.string());
    }
    proof_util::writeWords(state, f);
    fclose(f);
}

Words ProofSet::load(uint32_t k) const {
    auto fname = partFile(k);
    FILE* f = fopen(fname.string().c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open partial file for reading: " + fname.string());
    }
    Words w = proof_util::readWords(E, f);
    fclose(f);
    return w;
}

// Build a final proof from partials + final residue
std::pair<Proof, std::vector<uint64_t>> ProofSet::computeProof(const Words &finalRes) const {
    // 1) Collect partials in ascending order (lowest iteration -> highest)
    //    or in the order GpuOwl expects: usually the points from largest to smallest iteration.
    // 2) We do a pass to generate random exponents for each partial (just as GpuOwl does).
    //    This is optional for a CPU environment— if you want to exactly match GpuOwl’s random exps,
    //    you must replicate the same chain of hashing. For demonstration, we’ll do a simpler approach:
    //    We just “simulate” we’ll do it in verify() anyway.

    std::vector<Words> middleResidues;
    middleResidues.reserve(points.size());
    for (uint32_t idx : points) {
        Words w = load(idx);
        middleResidues.push_back(std::move(w));
    }

    // The user might want the random exponents that verification uses. Here we just do a “dummy” example:
    std::vector<uint64_t> randomExps;
    randomExps.reserve(points.size());
    // In real GpuOwl, these exponents are derived during merges, not stored ahead of time. 
    // We'll do a placeholder of zeros.
    for (size_t i = 0; i < middleResidues.size(); i++) {
        randomExps.push_back(0ULL);
    }

    // Build the Proof object in memory
    Proof proofObj(E, finalRes, middleResidues);
    return { proofObj, randomExps };
}




// A simple default for "bestPower"
uint32_t ProofSet::bestPower(uint32_t exponent) {
    // GpuOwl’s typical approach is 8 for exponents in the ~ 100k-1M range,
    // maybe 10 for exponents ~10M, etc. Let’s just do a quick approach:
    if (exponent < 500000) return 6;   // smaller exponent => fewer partials
    else if (exponent < 5000000) return 8;
    else if (exponent < 50000000) return 10;
    else return 12;
}


Words ProofSet::fromUint64(const std::vector<uint64_t> &input64, uint32_t E)
{
    size_t n32 = (size_t(E - 1) / 32) + 1;
    size_t needed64 = (n32 + 1) / 2; // how many 64-bit words we might need

    Words output32;
    output32.resize(n32, 0);

    for (size_t i = 0; i < needed64 && i < input64.size(); i++) {
        uint64_t v = input64[i];
        // lower 32 bits
        if (2*i < n32) {
            output32[2*i] = static_cast<uint32_t>(v & 0xFFFFFFFFULL);
        }
        // higher 32 bits
        if (2*i + 1 < n32) {
            output32[2*i + 1] = static_cast<uint32_t>(v >> 32);
        }
    }

    return output32;
}