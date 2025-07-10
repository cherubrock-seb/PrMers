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
// Copyright Mihai Preda

#pragma once

#include "io/sha3.h"
#include "io/Hash.h"

#include <vector>
#include <array>
#include <cstring>
using namespace std;
namespace io{
class Sha3Hash {
  SHA3Context context;
  static const constexpr int SIZE_BITS = 256;

  void clear() { SHA3Init(&context, SIZE_BITS); }
  
public:
  Sha3Hash() { clear(); }

  void update(const void* data, u32 size) { SHA3Update(&context, reinterpret_cast<const unsigned char*>(data), size); }
  
  array<u64, 4> finish() && {
    u64 *p = reinterpret_cast<u64 *>(SHA3Final(&context));
    return {p[0], p[1], p[2], p[3]};
  }
};

using SHA3 = Hash<Sha3Hash>;
} // namespace io