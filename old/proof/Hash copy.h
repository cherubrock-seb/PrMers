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

#include "common.h"
#include <array>
#include <vector>
#include <cstring>
#include <string>

using namespace std;

template <typename H>
class Hash {
  H h;
  
public:
  template <typename... Ts>
  static auto hash(Ts... data) {
    Hash hash;
    (hash.update(data),...);
    return std::move(hash).finish();
  }

  Hash& update(const void* data, u32 size) { h.update(data, size); return *this; }

  template<typename T, std::size_t N>
  Hash&& update(const array<T, N>& v) && { h.update(v.data(), N * sizeof(T)); return std::move(*this); }

  void update(u32 x) { h.update(&x, sizeof(x)); }
  void update(u64 x) { h.update(&x, sizeof(x)); }

  template<typename T>
  void update(const vector<T>& v) { h.update(v.data(), v.size() * sizeof(T)); }

  void update(const string& s) {h.update(s.c_str(), s.size()); }
  
  auto finish() && { return std::move(h).finish(); }
};
