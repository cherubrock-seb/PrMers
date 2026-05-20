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

The current GPU path supports odd radix `3` and `9`. The default CRT mode is now `--crt-odd-radix auto`: it keeps the normal power-of-two CRT path when radix-9 is not clearly useful, and selects the optimized odd-radix 9 path when the tested sizes show a win. Explicit `--crt-odd-radix off`, `3`, or `9` still force the old behavior.

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
--crt-odd-radix auto
--crt-center-mode halfreal
--crt-halfreal-flags 48
PRMERS_CRT_MIXED_TILE14=1
PRMERS_CRT_MIXED_SHIFT_LUT=1
PRMERS_CRT_MIXED_TILE14_SHIFT=1
PRMERS_CRT_MIXED_TILE14_LMAT=1
PRMERS_CRT_MIXED_PRECRT_COEFFHI=1
PRMERS_CRT_MIXED_FUSE_PACK_BOTH=0
PRMERS_CRT_MIXED_FUSE_LDS_BOTH=0
PRMERS_CRT_MIXED_FUSE_CENTER_BOTH=0
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
  --crt-mixed-row-core generic \
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
| `--crt-odd-radix auto` | yes | default heuristic: off for cases like M136279841, odd9 for cases like M142606357 and the small/medium tested windows |
| `--crt-odd-radix off` | no | normal power-of-two CRT path |
| `--crt-odd-radix 3` | no | mixed CRT/PFA radix 3 path |
| `--crt-odd-radix 9` | yes | force mixed CRT/PFA radix 9 path |
| `PRMERS_CRT_MIXED_TILE14=0/1` | `1` | cooperative LDS odd head/tail tile14 path for radix 9 |
| `PRMERS_CRT_MIXED_SHIFT_LUT=0/1` | `1` for radix 9 | use precomputed unweight shift table in mixed odd head/tail |
| `PRMERS_CRT_MIXED_TILE14_SHIFT=0/1` | `1` | tile14 variant using the shift LUT |
| `PRMERS_CRT_MIXED_TILE14_LMAT=0/1` | `1` | local-matrix tile14-shift head/tail for radix 9; set `0` to compare the previous tile14-shift kernels |
| `PRMERS_CRT_MIXED_PACK_TILE7=0/1` | fallback | old cooperative LDS odd head for radix 9 |
| `PRMERS_CRT_MIXED_PRECRT_TILE7=0/1` | fallback | old cooperative LDS odd tail for radix 9 |
| `PRMERS_CRT_MIXED_PRECRT_COEFFHI=0/1` | `1` | write preCRT coeffhi stream before Garner |
| `PRMERS_CRT_MIXED_PRECRT_OUTPAR=0/1` | `0` | one-output-per-thread tail test |
| `PRMERS_CRT_MIXED_PRECRT_SPLIT=0/1` | `0` | split residue tail before preCRT combine |
| `PRMERS_CRT_MIXED_FUSE_PACK_BOTH=0/1` | `0` | fused GF61/GF31 edge test; usually slower on RTX 3080 |
| `PRMERS_CRT_MIXED_FUSE_LDS_BOTH=0/1` | `0` | legacy env switch for fused GF61/GF31 LDS forward/inverse tests |
| `PRMERS_CRT_MIXED_FUSE_CENTER_BOTH=0/1` | `0` | legacy env switch for fused GF61/GF31 LDS center tests |
| `--crt-mixed-row-fuse-both off|auto|center|stage|all|force` | `off` | CLI switch for fused GF61/GF31 row stages and/or center, sizes 8..1024 |
| `PRMERS_CRT_MIXED_LDS512_DISABLE=0/1` | `0` | disable LDS512 row core |
| `--crt-mixed-row-core auto|lds512|lds1024|generic` | `auto` | explicitly selects the mixed odd row core for path tests |


## LDS and row-core knobs

There are two different controls:

- `--crt-local-square N` or `--crt-mixed-row-center N` chooses the row center size.
- `--crt-lds-stage N` or `--crt-mixed-row-stage N` chooses the LDS stage used before the center.

For the mixed odd radix path, forced LDS row centers now exist for `8..1024`. The 512 path still has special optimized kernels; other sizes use the generic LDS kernels and are mainly for validation/performance comparison.

Clear forced syntax:

```bash
# stable fast path, center 512, stage 512
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-stage 512 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

```bash
# force center 1024 and a 256 stage before the center
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-center 1024 \
  --crt-mixed-row-stage 256 \
  --device 1 \
  --iters 1000 \
  --profile-kernels
```

Equivalent old names still work:

```bash
--crt-local-square 512      # same as --crt-mixed-row-center 512
--crt-lds-stage 512         # same as --crt-mixed-row-stage 512
```

`--crt-mixed-row-core` values:

```text
auto      default planner
lds       force the LDS row core using --crt-mixed-row-center, supports 8..1024
lds512    shorthand forcing center 512
lds1024   shorthand forcing center 1024
generic   force the generic radix2 row path
```

Important: the program refuses misleading combinations. For example `--crt-mixed-row-core lds1024 --crt-mixed-row-center 256` is rejected instead of silently running the 1024 center.

Useful test grid:

```bash
# compare fast centers
for C in 512 1024; do
  for S in 128 256 512 1024; do
    echo "=== odd9 fast center=$C stage=$S ==="
    ./prmers_opencl_prp 142606357 \
      --modulus crt \
      --crt-odd-radix 9 \
      --crt-center-mode halfreal \
      --crt-halfreal-no-autoprobe \
      --crt-halfreal-flags 48 \
      --crt-mixed-row-core lds \
      --crt-mixed-row-center $C \
      --crt-mixed-row-stage $S \
      --device 1 \
      --iters 1000 \
      --profile-kernels
  done
done
```

```bash
# generic path sanity test for smaller requested center sizes
for C in 8 16 32 64 128 256; do
  echo "=== odd9 generic center=$C ==="
  ./prmers_opencl_prp 3021377 \
    --modulus crt \
    --crt-odd-radix 9 \
    --crt-center-mode halfreal \
    --crt-halfreal-no-autoprobe \
    --crt-halfreal-flags 48 \
    --crt-mixed-row-core generic \
    --crt-mixed-row-center $C \
    --device 1 \
    --iters 5 \
    --profile-kernels
 done
```

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

### Current default radix 9 validation with tile14 shift-LUT

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
  --crt-halfreal-dump hr3021377_odd9_tile14_shift \
  --crt-halfreal-dump-count 4096 \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

Expected output contains:

```text
CRT halfreal validator: OK
```

And the diff file should be empty:

```bash
wc -l hr3021377_odd9_tile14_shift_diff_iter1.txt
```

## Benchmark matrix

### Current default radix 9 POC

```bash
./prmers_opencl_prp 136279841 --device 1 --iters 5000 --profile-kernels
```

### Radix 9 without shift LUT

```bash
PRMERS_CRT_MIXED_SHIFT_LUT=0 \
./prmers_opencl_prp 136279841 --device 1 --iters 5000 --profile-kernels
```

### Radix 9 old tile7 fallback

```bash
PRMERS_CRT_MIXED_SHIFT_LUT=0 \
PRMERS_CRT_MIXED_TILE14=0 \
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

## Manual path matrix tests without scripts

This release can be shared with only these three files:

```text
prmers_opencl_prp.cpp
prmers_opencl_prp.cl
README.md
```

No helper script is required. The commands below reproduce the useful path checks manually.

### Build once

```bash
g++ -O3 -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

### Quick correctness checks

Radix 3 small case:

```bash
PRMERS_CRT_HALFREAL_DUMP_ALWAYS=1 \
PRMERS_CRT_HALFREAL_CPU_REF_ITERS=4 \
./prmers_opencl_prp 11213 \
  --modulus crt \
  --crt-odd-radix 3 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 4 \
  --crt-halfreal-dump hr11213_odd3 \
  --crt-halfreal-dump-count 2048 \
  --device 1 \
  --iters 1
```

Radix 9 small/medium case:

```bash
PRMERS_CRT_HALFREAL_DUMP_ALWAYS=1 \
PRMERS_CRT_HALFREAL_CPU_REF_ITERS=1 \
./prmers_opencl_prp 3021377 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 1 \
  --crt-halfreal-dump hr3021377_odd9 \
  --crt-halfreal-dump-count 4096 \
  --device 1 \
  --iters 1
```

Expected output contains:

```text
CRT halfreal validator: OK
```

### Performance comparisons on M142606357

Normal CRT, no odd radix, generic/default planner:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix off \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 3000 \
  --profile-kernels
```

Normal CRT, force head/tail LDS512:

```bash
PRMERS_CRT_HALFREAL_LDS_SPLIT=1 \
PRMERS_CRT_HALFREAL_LDS_PAIR=1 \
PRMERS_CRT_HALFREAL_LDS_HOTOPT=1 \
PRMERS_CRT_HALFREAL_HEAD_LDS=512 \
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix off \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-local-square 512 \
  --crt-lds-stage 512 \
  --crt-mixed-row-core lds512 \
  --device 1 \
  --iters 3000 \
  --profile-kernels
```

Expected labels when exact 512 is used:

```text
crt_halfreal_head_pack_lds512_61
crt_halfreal_tail_lds512_unpack_61
```

Normal CRT, force head/tail LDS1024:

```bash
PRMERS_CRT_HALFREAL_LDS_SPLIT=1 \
PRMERS_CRT_HALFREAL_LDS_PAIR=1 \
PRMERS_CRT_HALFREAL_LDS_HOTOPT=1 \
PRMERS_CRT_HALFREAL_HEAD_LDS=1024 \
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix off \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-local-square 512 \
  --crt-lds-stage 1024 \
  --device 1 \
  --iters 3000 \
  --profile-kernels
```

Expected labels when exact 1024 is used:

```text
crt_halfreal_head_pack_lds1024_61
crt_halfreal_tail_lds1024_unpack_61
```

Mixed odd radix 9 stable path:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-local-square 512 \
  --crt-lds-stage 512 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

Expected stable odd9 labels:

```text
crt_mixed_lds512_center_61
crt_mixed_lds512_forward_61
crt_mixed_lds512_inverse_61
crt_mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat
crt_garner_first_coeffhi_mask32_anybase_x2
```

Mixed odd radix 9 with generic radix2 row core forced:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 1000 \
  --profile-kernels
```

This is useful only as a fallback comparison. It should be slower and usually shows radix2 fallback kernels.

### Logging all manual tests in one file

```bash
LOG=prmers_manual_matrix_$(date +%Y%m%d_%H%M%S).txt
{
  echo "===== no odd default ====="
  ./prmers_opencl_prp 142606357 --modulus crt --crt-odd-radix off --crt-center-mode halfreal --crt-halfreal-no-autoprobe --crt-halfreal-flags 48 --device 1 --iters 1000 --profile-kernels

  echo "===== no odd head512 ====="
  PRMERS_CRT_HALFREAL_LDS_SPLIT=1 PRMERS_CRT_HALFREAL_LDS_PAIR=1 PRMERS_CRT_HALFREAL_LDS_HOTOPT=1 PRMERS_CRT_HALFREAL_HEAD_LDS=512 \
  ./prmers_opencl_prp 142606357 --modulus crt --crt-odd-radix off --crt-center-mode halfreal --crt-halfreal-no-autoprobe --crt-halfreal-flags 48 --crt-local-square 512 --crt-lds-stage 512 --device 1 --iters 1000 --profile-kernels

  echo "===== no odd head1024 ====="
  PRMERS_CRT_HALFREAL_LDS_SPLIT=1 PRMERS_CRT_HALFREAL_LDS_PAIR=1 PRMERS_CRT_HALFREAL_LDS_HOTOPT=1 PRMERS_CRT_HALFREAL_HEAD_LDS=1024 \
  ./prmers_opencl_prp 142606357 --modulus crt --crt-odd-radix off --crt-center-mode halfreal --crt-halfreal-no-autoprobe --crt-halfreal-flags 48 --crt-local-square 512 --crt-lds-stage 1024 --device 1 --iters 1000 --profile-kernels

  echo "===== odd9 stable ====="
  ./prmers_opencl_prp 142606357 --modulus crt --crt-odd-radix 9 --crt-center-mode halfreal --crt-halfreal-no-autoprobe --crt-halfreal-flags 48 --crt-local-square 512 --crt-lds-stage 512 --device 1 --iters 3000 --profile-kernels
} 2>&1 | tee "$LOG"

echo "saved in $LOG"
```

## Notes

This is not a final PRP implementation. 
It is a POC validation and benchmarking branch for the mixed CRT/PFA odd-radix GPU layout.



### Single-field half-real mode

For standalone `--modulus gf61` or `--modulus gf31`, the default path remains the classic full-size field transform to avoid changing existing results.  The half-real single-field path can be selected explicitly:

```bash
./prmers_opencl_prp 216091 --modulus gf61 --single-center-mode halfreal --crt-halfreal-flags 48
./prmers_opencl_prp 216091 --modulus gf31 --single-center-mode halfreal --crt-halfreal-flags 48
```

With `--crt-halfreal-flags 48`, the single-field half-real path now uses the fast LDS512 route by default when possible:

```text
pack+first 512 DIF in LDS -> residual row NTT -> LDS512 pair center -> residual inverse -> last 512 DIT+unpack in LDS
```



Auto mode is available for standalone fields:

```bash
./prmers_opencl_prp 216091 --modulus gf61 --single-center-mode auto --crt-halfreal-flags 48
./prmers_opencl_prp 756839 --modulus gf31 --single-center-mode auto
```

Current policy:

```text
gf61: normal for very small N, halfreal for N >= PRMERS_SINGLE_HALFREAL_AUTO_MIN_N
gf31: normal
default PRMERS_SINGLE_HALFREAL_AUTO_MIN_N=4096
```

Runtime switches:

```text
PRMERS_SINGLE_HALFREAL_FAST512=0        disable the standalone fast LDS512 head/center/tail path
PRMERS_SINGLE_HALFREAL_LDS_PAIR=0      disable the standalone LDS512 pair center path
PRMERS_SINGLE_HALFREAL_AUTO_MIN_N=65536 auto-mode threshold for gf61
```

Aliases:

```text
--single-halfreal        same as --single-center-mode halfreal
--single-normal          same as --single-center-mode normal
--field-center-mode      alias for --single-center-mode
--single-center-mode auto keeps gf61 normal for small cases like p=216091 and switches to gf61 halfreal from the medium/large window; gf31 stays normal
```

The CRT options `--crt-local-square`, `--crt-lds-stage`, `--crt-lds-tile`, `--crt-edge-radix` still apply to the CRT paths.  `--single-center-mode` is only for standalone `gf61` / `gf31`.

## Odd radix 9 stable default after fused Garner test

The fused preCRT+Garner experiment exists, but it is **off by default** because it was much slower on RTX 3080 for M142606357.

Stable default:

```bash
./prmers_opencl_prp 142606357 --modulus crt --crt-odd-radix 9 --crt-center-mode halfreal --crt-halfreal-flags 48 --device 1 --iters 5000 --profile-kernels
```

Default:

```text
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER=0
```

Old experimental fused tail, only for comparison:

```bash
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER=1 \
./prmers_opencl_prp 142606357 --modulus crt --crt-odd-radix 9 --crt-center-mode halfreal --crt-halfreal-flags 48 --device 1 --iters 5000 --profile-kernels
```

Expected stable profile uses separate kernels:

```text
crt_mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat
crt_garner_first_coeffhi_mask32_anybase_x2
```

If the fused experiment is enabled you will see:

```text
crt_mixed_odd9_precrt_garner64_lmat
```

Keep it disabled unless a later kernel version fixes the register pressure and serialization.

## LDS planner notes

`--crt-local-square` and `--crt-lds-stage` accept `1024` as well as the older sizes.
For the half-real CRT path the head/tail LDS radix can be forced with:

```bash
PRMERS_CRT_HALFREAL_HEAD_LDS=1024 ./prmers_opencl_prp ... --crt-lds-stage 1024
```

To keep the automatic planner from selecting 1024, cap it at 512:

```bash
PRMERS_CRT_HALFREAL_HEAD_LDS_MAX=512 ./prmers_opencl_prp ... --crt-lds-stage 512
```

`PRMERS_CRT_HALFREAL_HEAD_LDS=512` now means exact 512: it will not silently fall back to 256 or 128. `PRMERS_CRT_HALFREAL_HEAD_LDS_MAX=512` is only a cap for auto mode: the planner starts at the largest valid value not above 512. `PRMERS_CRT_HALFREAL_HEAD_LDS=0` disables the head/tail LDS shortcut.

By default the planner allows a mixed residual around the LDS512 middle stage. This is needed for cases such as `ln=23`, where a 512 head leaves a `16*512` residual. Use `PRMERS_CRT_HALFREAL_STRICT_RESIDUAL=1` only if you want the old conservative behaviour that can fall back to 128.

The 1024 head can be useful for sizes such as `ln=23`, where it leaves a clean radix-8 residual before the LDS512 pair center.

In mixed odd-radix mode the default fast row core is still the stable LDS512 path. For the current fast odd9 path use `--crt-local-square 512 --crt-lds-stage 512` or leave the defaults. For path tests use `--crt-mixed-row-core lds512`, `--crt-mixed-row-core lds1024`, or `--crt-mixed-row-core generic`. The `lds1024` core is a real 1024 center path; for example `--crt-mixed-row-core lds1024 --crt-local-square 1024 --crt-lds-stage 256` should show `row-core=LDS1024` and `crt_mixed_lds1024_center_*` in the profile. There is still no optimized odd-row `center256 + stage1024` path. In the startup line, check `row-core=` and `row-core-request=`.

## v7 mixed row LDS matrix

This version lets `--crt-mixed-row-core lds` use a forced row center in `{8,16,32,64,128,256,512,1024}`.
`--crt-mixed-row-stage` can also be `{8,16,32,64,128,256,512,1024}`.

The mixed odd row planner now builds a real LDS plan. Example:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 1024 \
  --crt-mixed-row-center 256 \
  --device 1 \
  --iters 1000 \
  --profile-kernels
```

If the requested LDS stage does not divide the remaining row size down to the requested center, the planner automatically adds the small completion radix. For example, a 256 stage with a 256 center can become `fwd=256+4,center=256,inv=reverse` when a radix-4 completion is needed.

New kernels:

- `gf61_crt_mixed_halfreal_lds_pair_any_61`
- `gf61_crt_mixed_halfreal_lds_pair_any_31`

Use `test_mixed_row_lds_matrix.sh` to validate or benchmark all stage/center combinations.
## v11 mixed GPU validator

Large mixed odd-radix exponents can make the exact CPU validator too slow.  Use
`--crt-mixed-gpu-reference` to validate the selected LDS/fused mixed row path
against the generic mixed GPU path instead.  This keeps the same CRT/PFA digit
order, but uses `--crt-mixed-row-core generic` internally for the reference
square.

Example:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 1 \
  --crt-mixed-gpu-reference \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both stage \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

Equivalent environment switch: `PRMERS_CRT_MIXED_GPU_REFERENCE=1`.
