# PrMers: GPU-accelerated Mersenne primality test

## Compilation
To compile:
```bash
g++ -std=c++11 -o prmers prmers.cpp -lOpenCL
```

## Usage
Example of execution:
```bash
 ./prmers 9279 -d 1
PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
Testing exponent: 9279
Using OpenCL device ID: 1

Launching OpenCL kernel (p = 9279); computation may take a while depending on the exponent.
Max global workers possible: 256
Final workers count: 256
Progress: 100.00% | Elapsed: 11.09s | Iterations/sec: 836.78 | ETA: 0.00s       
M9279 is composite.
Kernel execution time: 11.09 seconds
Iterations per second: 836.77 (9277 iterations in total)
```
Or without specifying a device:
```bash
./prmers 216091
PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
Testing exponent: 216091
Using OpenCL device ID: 0

Launching OpenCL kernel (p = 216091); computation may take a while depending on the exponent.
Max global workers possible: 256
Final workers count: 256
Progress: 0.11% | Elapsed: 4.05s | Iterations/sec: 57.50 | ETA: 3754.28s
```

## Output Example
```
PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
Testing exponent: 13
Using OpenCL device ID: 1

Launching OpenCL kernel (p_min_i = 13) without progress display; computation may take a while depending on the exponent.

Lucas-Lehmer test results:
Mp with p = 13 is a Mersenne prime.
Kernel execution time: 0.00150793 seconds
Iterations per second: 7294.76 (11 iterations in total)
```

## Notes
- Currently, there is **no progress display**.
- The code is **not yet fully optimized**.
- It uses **Crandall & Fagin's IBDWT** technique but implemented with **integer arithmetic**.
- The modular reduction is performed **modulo 2^64 - 2^32 + 1**, enabling efficient **NTT and IBDWT** for large integers.

## Background on Integer NTT Modulo 2^64 - 2^32 + 1
This project leverages a **fast modular reduction** method that avoids costly division operations, making it ideal for **integer-based NTTs on GPUs**. This method is inspired by **Nick Craig-Wood's ARM Prime** implementation.

The modulus:
```
p = 2^64 - 2^32 + 1
```
was chosen because:
1. It allows **fast modular reduction** using only bit shifts and additions.
2. It enables **FFT and NTT operations** on integers **up to 2^26** in length.
3. It supports **Integer Discrete Weighted Transforms (IDBWT)** for efficient large-number multiplication.

The modular reduction is based on the following properties:
```
2^64 ≡ 2^32 - 1 mod p
2^96 ≡ -1 mod p
2^128 ≡ -2^32 mod p
2^192 ≡ 1 mod p
```
which allows efficient reduction of large numbers using simple bit manipulations.

### Primitive roots and NTT
- `7` is a **primitive root modulo p**.
- An `n`-th root of unity is given by:
  ```
  7^(5*(p-1)/n) mod p
  ```
- The **IBDWT** requires a power-of-two **root of 2**. Since `2^192 ≡ 1 mod p`, we get:
  ```
  7^(5*(p-1)/192/n) mod p
  ```
  as the required root of 2.

### More details:
For an in-depth explanation, see [Nick Craig-Wood's ARM Prime Math](https://www.craig-wood.com/nick/armprime/math/).

## Resources & Community
- **GIMPS (Great Internet Mersenne Prime Search)** - [https://www.mersenne.org](https://www.mersenne.org)  
  The largest distributed computing project dedicated to finding Mersenne primes.
- **Mersenne Forum** - [https://www.mersenneforum.org](https://www.mersenneforum.org)  
  A great place to discuss mathematical concepts, optimizations, and implementations related to Mersenne primes.

## Inspiration
This project is inspired by:
- [Nick Craig-Wood ioccc2012](https://github.com/ncw/ioccc2012)
- [Armprime project](https://github.com/ncw/)
- [Genefer by Yves Gallot](https://github.com/galloty/genefer22)
- [GPUOwl](https://github.com/preda/gpuowl)

Special thanks to **Yves Gallot** for his explanations regarding **NTT and IBDWT**.

**Author:** Cherubrock
