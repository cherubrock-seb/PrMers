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
#ifndef MY_COMMON_H
#define MY_COMMON_H

#include <cstdint>
#include <string>
#include <vector>

// Basic type aliases
using u8  = uint8_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

// A container for our “big integer” limbs in 32-bit little-endian format
using Words = std::vector<u32>;

/*
 * Return a 64-bit integer from the first two 32-bit limbs in `words`,
 * which is a quick human-friendly way to display a partial representation
 * of the big integer. If `words` has fewer than 2 limbs, treat missing limbs as zero.
 */
inline u64 res64(const Words &words) {
    if (words.size() >= 2) {
        return (static_cast<u64>(words[1]) << 32) | words[0];
    } else if (!words.empty()) {
        return words[0];
    } else {
        return 0ULL;
    }
}

/*
 * Number of 32-bit words needed to hold a Mersenne exponent of size `E` bits.
 * GpuOwl typically does: (E - 1)/32 + 1
 */
inline u32 nWords(u32 E) {
    return (E - 1) / 32 + 1;
}

/*
 * Create a Words vector of the right size for exponent `E`, initialize
 * the lowest limb with `value`, and the rest with zeros.
 */
inline Words makeWords(u32 E, u32 value) {
    Words ret(nWords(E), 0);
    ret[0] = value;
    return ret;
}

/*
 * Convert a 64-bit integer to a hexadecimal string (e.g. for logging).
 */
std::string hex(u64 x);

/*
 * Strip trailing newlines or carriage returns from a string.
 */
std::string rstripNewline(std::string s);

/*
 * Basic CRC32 for integrity checks, using a nibble-based approach.
 */
u32 crc32(const void *data, size_t size);

/*
 * Overload for convenience with a `Words` vector.
 */
inline u32 crc32(const Words &words) {
    return crc32(words.data(), words.size() * sizeof(u32));
}

#endif // MY_COMMON_H
