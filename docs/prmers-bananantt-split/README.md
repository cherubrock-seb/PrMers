# PrMers BananaNTT Split : mixed CRT/PFA odd-radix POC

This branch is a GPU POC for the mixed CRT/PFA half-real idea first tested in the CPU prototype:

`docs/mersenne2_mixed_crt_2d_half_fast`
https://github.com/cherubrock-seb/PrMers/tree/main/docs/mersenne2_mixed_crt_2d_half_fast

The goal is to test transforms of the form:

```text
N = odd * 2^m
```

The odd axis is handled with CRT/PFA indexing. The `2^m` axis keeps the usual half-real packing:

```text
z[k] = coeff(2*k) + i*coeff(2*k + 1)
```

The current GPU path supports odd radix `3` and `9`. The default CRT mode is the optimized odd-radix 9 path with LDS512 row kernels, tile7 odd head/tail kernels, and preCRT coeffhi output.

## Build

```bash
g++ -O3 -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

`-lgmp` is required for the CPU reference validator.

## Default run

```bash
./prmers_opencl_prp 136279841 --device 1 --iters 5000 --profile-kernels
```

Default mode:

```text
--modulus crt
--crt-odd-radix 9
--crt-center-mode halfreal
--crt-halfreal-flags 48
PRMERS_CRT_MIXED_PACK_TILE7=1
PRMERS_CRT_MIXED_PRECRT_TILE7=1
PRMERS_CRT_MIXED_PRECRT_COEFFHI=1
```

## Field modes

### GF(M61^2) only

```bash
./prmers_opencl_prp 216091 --modulus gf61 --device 1 --iters 1000 --profile-kernels
```

### GF(M31^2) only

```bash
./prmers_opencl_prp 216091 --modulus gf31 --device 1 --iters 1000 --profile-kernels
```

### CRT GF(M61^2) x GF(M31^2), normal power-of-two transform

```bash
./prmers_opencl_prp 136279841 \
  --modulus crt \
  --crt-odd-radix off \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

### CRT GF(M61^2) x GF(M31^2), mixed odd radix 3

```bash
./prmers_opencl_prp 11213 \
  --modulus crt \
  --crt-odd-radix 3 \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 1000 \
  --profile-kernels
```

### CRT GF(M61^2) x GF(M31^2), mixed odd radix 9

```bash
./prmers_opencl_prp 136279841 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

## Main test switches

All switches are optional. They are kept to compare implementations.

| switch | default | purpose |
|---|---:|---|
| `--crt-odd-radix off` | no | normal power-of-two CRT path |
| `--crt-odd-radix 3` | no | mixed CRT/PFA radix 3 path |
| `--crt-odd-radix 9` | yes | mixed CRT/PFA radix 9 path |
| `PRMERS_CRT_MIXED_PACK_TILE7=0/1` | `1` | cooperative LDS odd head for radix 9 |
| `PRMERS_CRT_MIXED_PRECRT_TILE7=0/1` | `1` | cooperative LDS odd tail for radix 9 |
| `PRMERS_CRT_MIXED_PRECRT_COEFFHI=0/1` | `1` | write preCRT coeffhi stream before Garner |
| `PRMERS_CRT_MIXED_PRECRT_OUTPAR=0/1` | `0` | one-output-per-thread tail test |
| `PRMERS_CRT_MIXED_PRECRT_SPLIT=0/1` | `0` | split residue tail before preCRT combine |
| `PRMERS_CRT_MIXED_SHIFT_LUT=0/1` | `0` | shift lookup table test |
| `PRMERS_CRT_MIXED_FUSE_PACK_BOTH=0/1` | `0` | fused GF61/GF31 pack test |
| `PRMERS_CRT_MIXED_FUSE_CENTER_BOTH=0/1` | `0` | fused GF61/GF31 center test |
| `PRMERS_CRT_MIXED_LDS512_DISABLE=0/1` | `0` | disable LDS512 row core |

## Validation

The validator can compare one or more GPU squarings with an exact CPU square. It needs `-lgmp` at build time.

### Small radix 3 validation

```bash
PRMERS_CRT_HALFREAL_DUMP_ALWAYS=1 \
PRMERS_CRT_HALFREAL_CPU_REF_ITERS=4 \
./prmers_opencl_prp 11213 \
  --modulus crt \
  --crt-odd-radix 3 \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 4 \
  --crt-halfreal-dump hr11213_odd3 \
  --crt-halfreal-dump-count 2048 \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

```bash
wc -l hr11213_odd3_diff_iter1.txt
```

Expected output:

```text
0 hr11213_odd3_diff_iter1.txt
```

### Small radix 9 validation

```bash
PRMERS_CRT_HALFREAL_DUMP_ALWAYS=1 \
PRMERS_CRT_HALFREAL_CPU_REF_ITERS=4 \
./prmers_opencl_prp 216091 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 4 \
  --crt-halfreal-dump hr216091_odd9 \
  --crt-halfreal-dump-count 4096 \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

```bash
wc -l hr216091_odd9_diff_iter1.txt
```

### Medium radix 9 validation

```bash
PRMERS_CRT_HALFREAL_DUMP_ALWAYS=1 \
PRMERS_CRT_HALFREAL_CPU_REF_ITERS=1 \
./prmers_opencl_prp 3021377 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 1 \
  --crt-halfreal-dump hr3021377_odd9 \
  --crt-halfreal-dump-count 4096 \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

```bash
wc -l hr3021377_odd9_diff_iter1.txt
```

## Benchmark matrix

### Current default radix 9 POC

```bash
./prmers_opencl_prp 136279841 --device 1 --iters 5000 --profile-kernels
```

### Radix 9 without tile7 head

```bash
PRMERS_CRT_MIXED_PACK_TILE7=0 \
./prmers_opencl_prp 136279841 --device 1 --iters 5000 --profile-kernels
```

### Radix 9 without tile7 tail

```bash
PRMERS_CRT_MIXED_PRECRT_TILE7=0 \
./prmers_opencl_prp 136279841 --device 1 --iters 5000 --profile-kernels
```

### Normal CRT half-real path

```bash
./prmers_opencl_prp 136279841 \
  --modulus crt \
  --crt-odd-radix off \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

### Large boundary case

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

Use the normal path for comparison:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix off \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 1000 \
  --profile-kernels
```

## Notes

This is not a final PRP implementation. 
It is a POC validation and benchmarking branch for the mixed CRT/PFA odd-radix GPU layout.

