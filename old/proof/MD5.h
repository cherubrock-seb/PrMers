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
#include "common.h"
#include <vector>
#include <array>
#include <cstring>


using namespace std;


struct MD5Context {
  int isInit;
  unsigned buf[4];
  unsigned bits[2];
  unsigned char in[64];
};

void MD5Init(MD5Context*);
void MD5Update(MD5Context*, const unsigned char*, unsigned int);
void MD5Final(unsigned char digest[16], MD5Context*);

class MD5Hash {
  MD5Context context;
  
public:
  MD5Hash() { MD5Init(&context); }
  void update(const void* data, u32 size) { MD5Update(&context, reinterpret_cast<const unsigned char*>(data), size); }
  
  string finish() && {
    unsigned char digest[16];
    MD5Final(digest, &context);
    string s;
    char hex[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) {
      s.push_back(hex[digest[i] >> 4]);
      s.push_back(hex[digest[i] & 0xf]);
    }
    return s;
  }
};

#include "Hash.h"

using MD5 = Hash<MD5Hash>;
