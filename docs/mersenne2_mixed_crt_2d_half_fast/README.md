# mersenne2 mixed CRT test

Small experimental CPU code based on Yves Gallot mersenne2.
Original code is here:
https://github.com/galloty/mersenne2

This version try to test odd transform size like 9 * 2^m or 21 * 2^m.
The odd part is separated with CRT indexing.
The power of two part keep the half real GF(p^2) transform.

It use GF(M61^2) and GF(M31^2), then reconstruct with CRT/Garner.
Radix 3, 7, 9, 21 and 63 are experimental.
Odd roots are real scalar, so multiplication is cheaper.

Compile on Linux:

```bash
g++ -std=c++17 -O3 -march=native mersenne2_mixed_crt_2d_half_fast_scalarodd.cpp -o mersenne2_mixed_crt_2d_half_fast
```

Compile on Mac:

```bash
clang++ -std=c++17 -O3 -march=native mersenne2_mixed_crt_2d_half_fast_scalarodd.cpp -o mersenne2_mixed_crt_2d_half_fast
```

Run:

```bash
./mersenne2_mixed_crt_2d_half_fast 11213
./mersenne2_mixed_crt_2d_half_fast 11213 --original
./mersenne2_mixed_crt_2d_half_fast 11213 --compare
```

Example:

```text
M86243 use 9*2^8 instead of 2^12
storage is 1152 complex values instead of 2048
```

This is not production code.
It is just a test to see if mixed radix can be usefull.
