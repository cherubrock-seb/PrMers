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
#ifndef MY_PROOF_H
#define MY_PROOF_H

#include <array>
#include <string>
#include <vector>
#include <filesystem>
#include "common.h"  // for Words, etc.

// Minimal "ProofInfo" struct with exponent/power + a textual MD5
struct ProofInfo {
    uint32_t power;
    uint32_t exp;
    std::string md5;
};

/*
  A final proof consists of:
   - Header: "PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=%u\nNUMBER=M%u\n"
   - The final residue B (Words)
   - A series of "middle" residues (Words)

  The proof merges them to check B == A^(2^span) (mod 2^exp - 1),
  using the same random-exponent chaining that GpuOwl does.
 */
class Proof {
public:
    // The Mersenne exponent
    const uint32_t E;
    // The final residue (the one after all PRP iterations), size ~ E bits
    const Words    B;
    // The partial “middle” residues
    const std::vector<Words> middles;

    /*
     * For a typical proof of power==8 (for example),
     * we have 8 partial residues at exponentially-later iteration points.
     */

    // Constructor
    Proof(uint32_t exponent, Words finalRes, std::vector<Words> middlesVec)
        : E(exponent), B(std::move(finalRes)), middles(std::move(middlesVec)) {}

    // Save to a .proof file
    //   e.g. "M216091-8.proof" or "216091-8.proof"
    void save(const std::filesystem::path& proofFile) const;

    // Load from a .proof file on disk
    static Proof load(const std::filesystem::path& path);

    Words expExp2WithProgress(const Words &base, uint32_t totalSquares) const;


    // Verify the final proof. Returns true if correct, false if invalid.
    //  - The optional 'expectedHashes' can be used to check that the random exponents
    //    match GpuOwl exactly. If you do not need that exact match, pass an empty vector.
    bool verify(const std::vector<uint64_t>& expectedHashes = {}) const;

    // For convenience: just returns "M216091-8.proof" style name
    std::filesystem::path fileName(const std::filesystem::path& proofDir) const;

    // A short metadata function that returns whether the final residue B is
    // the "9" residue (which typically indicates PRP probable-prime).
    bool isProbablePrime() const {
        // If the final Words is exactly {9,0,0,...}, it’s the typical PRP result
        // for a base-3 PRP test. Check that quickly:
        if (B.size() == 0) return false;
        if (B[0] != 9) return false;
        for (size_t i = 1; i < B.size(); i++) {
            if (B[i] != 0) return false;
        }
        return true;
    }
};

/*
 * ProofSet manages the intermediate partial residues (the "point files").
 * This is how GpuOwl typically does it: a set of iteration indices
 * is chosen; at those points, the code saves a partial residue. Then
 * after all iterations, these partials combine into a final proof.
 */
class ProofSet {
public:
    // Mersenne exponent
    const uint32_t E;
    // Proof power (how many partials to capture)
    const uint32_t power;

    // Construct
    ProofSet(uint32_t exponent, uint32_t pw);

    // Return the iteration indices that GpuOwl uses for your exponent & power
    // This is typically something like ~ [ E/2, E/4, E/8, ... ] etc.
    std::vector<uint32_t> points;

    // Save one partial residue for iteration k.
    // The file will go into something like: "exponent/proof/k-part"
    void save(uint32_t k, const Words& state) const;

    // Load the partial residue for iteration k
    Words load(uint32_t k) const;

    // Attempt to compute a final proof from all partials + final residue.
    // 1) We read the final residue from the user (the state after iteration E).
    // 2) We read each partial from disk in ascending order.
    // 3) We produce a “Proof” object in memory and also compute a set of random exponents
    //    for each partial residue. Optionally, we can store those exponents in `hashes`.
    std::pair<Proof, std::vector<uint64_t>> computeProof(const Words& finalRes) const;

    // Return a recommended "power" for exponent E (like GpuOwl’s bestPower logic).
    // You can use a fixed integer instead if you prefer.
    static uint32_t bestPower(uint32_t exponent);
    static Words fromUint64(const std::vector<uint64_t> &input64, uint32_t E);
    
private:
    // Return the sub-directory where partial proofs are stored (e.g. "32768/proof/")
    std::filesystem::path proofDir() const;

    // The path for storing a partial residue for iteration k (like "32768/proof/k-part").
    std::filesystem::path partFile(uint32_t k) const;
};

/*
 * Additional “utility” functions for hashing & big-int arithmetic we use in the proof flow.
 */
namespace proof_util {

  // Compute Sha3-256 of a Words buffer
  std::array<uint64_t, 4> sha3Words(const Words &w, const std::array<uint64_t, 4> &prefix = {0,0,0,0});

  // Merge two residues: result <- (A^h * B) mod (2^E - 1).
  // We pick a random exponent h from the previous residue’s Sha3, then do exponentiation mod 2^E - 1,
  // then multiply mod 2^E - 1.
  // This is the CPU version (no GPU).
  Words expMul(const Words &A, uint64_t exp, const Words &B, bool doSquareBefore, uint32_t E);

  // Repeated squaring: returns A^(2^count) mod (2^E - 1).
  // If count is large, we do it by standard repeated squaring in a loop.
  Words expExp2(const Words &A, uint32_t count, uint32_t E);

  // A naive big-int multiply mod (2^E - 1) for demonstration.
  // For real performance with large E, you might want FFT/NTT-based multiplication or e.g. GMP.
  Words modMul(const Words &x, const Words &y, uint32_t E);

  // A naive big-int exponent mod (2^E - 1). Uses square-and-multiply.
  Words modExp(const Words &base, uint64_t exponent, uint32_t E);

  // A quick check that Words is all zero except possibly the first limb, and that first limb=9
  bool isResidue9(const Words &w);

  // Minimal function to read or write a "Words" from a file. The size is (E-1)/8 + 1 bytes.
  void writeWords(const Words& w, FILE* f);
  Words readWords(uint32_t E, FILE* f);

  // For verifying the .proof file’s MD5, or for producing a string signature if needed
  std::string fileMD5(const std::filesystem::path &path);
}

#endif // MY_PROOF_H
