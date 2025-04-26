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
 // GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "common.h"
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstddef> // for size_t

std::string hex(u64 x) {
    std::ostringstream out;
    // Format x as 16 hex digits, padded with 0
    out << std::hex << std::setfill('0') << std::setw(16) << x;
    return out.str();
}

std::string rstripNewline(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

/*
 * A nibble-based CRC32 approach
 * We process each byte in two 4-bit steps.
 */
u32 crc32(const void *data, size_t size) {
    static const u32 table[16] = {
        0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC,
        0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C,
        0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C,
        0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C
    };

    const unsigned char *p = static_cast<const unsigned char*>(data);
    u32 crc = ~0U;

    for (size_t i = 0; i < size; i++) {
        // Process low nibble
        crc = table[(crc ^  p[i]) & 0x0F] ^ (crc >> 4);
        // Process high nibble
        crc = table[(crc ^ (p[i] >> 4)) & 0x0F] ^ (crc >> 4);
    }
    return ~crc;
}
