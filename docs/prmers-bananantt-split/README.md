# PrMers mixed CRT/PFA half-real row path

GPU experimental branch for CRT/PFA mixed-radix Mersenne transforms.  The main test target is a transform of the form:

```text
N = odd * 2^m
```

The odd axis is handled with CRT/PFA indexing, and the power-of-two axis uses the half-real packing:

```text
z[k] = coeff(2*k) + i*coeff(2*k + 1)
```

The current mixed path supports odd radix `3` and `9`.  The most tested path is radix `9`, CRT over `GF(M61^2) x GF(M31^2)`, row half-real mode, and LDS row stages/centers.

## Build

```bash
g++ -O3 -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

`-lgmp` is needed for the exact CPU validator.  Large exponents can instead use the GPU-reference validator.

## Fast odd9 command

```bash
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1 \
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0 \
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61=1 \
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=0 \
PRMERS_CRT_MIXED_CENTER_SPLIT_F48_61=1 \
PRMERS_CRT_MIXED_CENTER_SPLIT_F48_31=1 \
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both off \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

The current default keeps the original F48 scaling because it was slightly faster in the latest 142606357 runs.  The delayed-scale algebra is still available for comparison:

```bash
PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE=1 ./prmers_opencl_prp ...
```

The profile should show labels like:

```text
crt_mixed_lds512_center_1lds_f48_nonself_61
crt_mixed_lds512_center_1lds_f48_self_61
crt_mixed_lds512_forward_1lds_61
crt_mixed_lds512_inverse_1lds_61
```

## Main CLI options

| option | values | purpose |
|---|---|---|
| `--modulus` | `crt`, `gf61`, `gf31` | field/modulus path |
| `--crt-odd-radix` | `off`, `auto`, `3`, `9` | normal power-of-two CRT path or mixed odd CRT/PFA path |
| `--crt-center-mode` | `halfreal`, `normal` | CRT center convention |
| `--crt-halfreal-flags` | usually `48` | half-real convention; `48` is the optimized tested convention |
| `--crt-mixed-row-core` | `auto`, `lds`, `lds512`, `lds1024`, `generic` | row core selection for mixed odd path |
| `--crt-mixed-row-stage` | `8..1024` powers of two | LDS stage size before/after the row center |
| `--crt-mixed-row-center` | `8..1024` powers of two | LDS center/square size |
| `--crt-mixed-row-fuse-both` | `off`, `auto`, `center`, `stage`, `all`, `force` | optional fused GF61/GF31 row LDS kernels |
| `--crt-halfreal-validate-random` | flag | start from a random state and validate square(s) |
| `--crt-halfreal-validate-iters N` | integer | number of validation squares |
| `--crt-mixed-gpu-reference` | flag | compare selected mixed path against generic mixed GPU path |
| `--profile-kernels` | flag | print per-kernel timings |
| `--profile-validator` | flag | include validator kernel timings |

Aliases still accepted:

```text
--crt-local-square N  == --crt-mixed-row-center N
--crt-lds-stage N    == --crt-mixed-row-stage N
```

## Main environment switches

### Row LDS policies

| env | default | purpose |
|---|---:|---|
| `PRMERS_CRT_MIXED_CENTER_SINGLE_LDS` | mixed | global single-LDS center policy for both fields |
| `PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61` | `1` | use one LDS array for GF61 center pair processing |
| `PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31` | `0` | use one LDS array for GF31 center pair processing |
| `PRMERS_CRT_MIXED_STAGE_SINGLE_LDS` | mixed | global single-LDS forward/inverse stage policy |
| `PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61` | `1` | use one LDS array for GF61 stage pair processing |
| `PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31` | `0` | use one LDS array for GF31 stage pair processing |
| `PRMERS_CRT_MIXED_CENTER_FUSE_OVERRIDES_SINGLE_LDS` | `0` | fused GF61/GF31 center overrides single-LDS field policy |
| `PRMERS_CRT_MIXED_STAGE_FUSE_OVERRIDES_SINGLE_LDS` | `0` | fused GF61/GF31 stage overrides single-LDS field policy |

### F48 center shortcuts

| env | default | purpose |
|---|---:|---|
| `PRMERS_CRT_MIXED_CENTER_SPLIT_F48_61` | `1` | split GF61 F48 center into self-pair and non-self-pair kernels |
| `PRMERS_CRT_MIXED_CENTER_SPLIT_F48_31` | `1` | split GF31 F48 center into self-pair and non-self-pair kernels when the matching path is active |
| `PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE` | `0` | global delayed-scale shortcut for F48 center algebra |
| `PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE_61` | inherits global | per-field delayed-scale shortcut for GF61 |
| `PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE_31` | inherits global | per-field delayed-scale shortcut for GF31 |

Compatibility aliases are still accepted:

```text
PRMERS_CRT_MIXED_CENTER_F48_LATE_SCALE
PRMERS_CRT_MIXED_CENTER_F48_LATE_SCALE_61
PRMERS_CRT_MIXED_CENTER_F48_LATE_SCALE_31
```

The delayed-scale formula is only used by the `flags=48` fast center helper.  The default is off because the extra normalisation was not faster on the latest RTX 3080 odd9 tests.  If enabled, all specialized F48 center kernels call the same helper, so the option applies to self-pair, non-self-pair, one-LDS, two-LDS, fused 61x31 center, and generic F48 center calls.

### Mixed odd head/tail and Garner

| env | default | purpose |
|---|---:|---|
| `PRMERS_CRT_MIXED_TILE14` | `1` | cooperative LDS odd head/tail tile14 path |
| `PRMERS_CRT_MIXED_SHIFT_LUT` | `1` | precomputed unweight shift table |
| `PRMERS_CRT_MIXED_TILE14_SHIFT` | `1` | tile14 variant using the shift LUT |
| `PRMERS_CRT_MIXED_TILE14_LMAT` | `1` | local-matrix tile14-shift kernels |
| `PRMERS_CRT_MIXED_PRECRT_COEFFHI` | `1` | write coeffhi stream before Garner |
| `PRMERS_CRT_MIXED_PRECRT_OUTPAR` | `0` | one-output-per-thread tail experiment |
| `PRMERS_CRT_MIXED_PRECRT_SPLIT` | `0` | split residue tail before preCRT combine |
| `PRMERS_CRT_MIXED_FUSE_PACK_BOTH` | `0` | fused GF61/GF31 odd pack/head experiment |
| `PRMERS_CRT_MIXED_FUSE_LDS_BOTH` | `0` | legacy fused GF61/GF31 LDS stage experiment |
| `PRMERS_CRT_MIXED_FUSE_CENTER_BOTH` | `0` | legacy fused GF61/GF31 LDS center experiment |
| `PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER` | `0` | old fused preCRT+Garner experiment, normally slower |

## Validation commands

Small exact CPU validation:

```bash
PRMERS_CRT_HALFREAL_DUMP_ALWAYS=1 \
PRMERS_CRT_HALFREAL_CPU_REF_ITERS=4 \
./prmers_opencl_prp 216091 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 4 \
  --crt-halfreal-dump hr216091_odd9 \
  --crt-halfreal-dump-count 4096 \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

Large GPU-reference validation:

```bash
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1 \
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0 \
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61=1 \
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=0 \
PRMERS_CRT_MIXED_CENTER_SPLIT_F48_61=1 \
PRMERS_CRT_MIXED_CENTER_SPLIT_F48_31=1 \
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
  --crt-mixed-row-fuse-both off \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

Expected validator line:

```text
CRT halfreal validator: OK
```

## Matrix script

The script runs several stage/center/fuse combinations, concatenates all logs, and creates TSV summaries.

Quick small validation:

```bash
DEVICE=1 P=3021377 ITERS=1 VAL_ITERS=2 VALIDATE=1 \
STAGES="256 512 1024" CENTERS="256 512 1024" FUSE_BOTHS="off all" \
SINGLE_LDS_CENTER_61=1 SINGLE_LDS_CENTER_31=0 \
SINGLE_LDS_STAGE_61=1 SINGLE_LDS_STAGE_31=0 \
CENTER_SPLIT_F48_61=1 CENTER_SPLIT_F48_31=1 \
./test_mixed_row_lds_matrix.sh
```

Large GPU-reference validation:

```bash
DEVICE=1 P=142606357 ITERS=1 VAL_ITERS=1 VALIDATE=1 GPU_REF=1 \
STAGES="512 1024" CENTERS="512" FUSE_BOTHS="off all" \
SINGLE_LDS_CENTER_61=1 SINGLE_LDS_CENTER_31=0 \
SINGLE_LDS_STAGE_61=1 SINGLE_LDS_STAGE_31=0 \
CENTER_SPLIT_F48_61=1 CENTER_SPLIT_F48_31=1 \
./test_mixed_row_lds_matrix.sh
```

Benchmark matrix:

```bash
DEVICE=1 P=142606357 ITERS=1000 VALIDATE=0 PROFILE=1 \
STAGES="256 512 1024" CENTERS="256 512 1024" FUSE_BOTHS="off all" \
SINGLE_LDS_CENTER_61=1 SINGLE_LDS_CENTER_31=0 \
SINGLE_LDS_STAGE_61=1 SINGLE_LDS_STAGE_31=0 \
CENTER_SPLIT_F48_61=1 CENTER_SPLIT_F48_31=1 \
./test_mixed_row_lds_matrix.sh
```

Output files:

```text
mixed_row_lds_matrix_logs/all_tests_combined.log
mixed_row_lds_matrix_logs/summary.tsv
mixed_row_lds_matrix_logs/summary_detail.tsv
mixed_row_lds_matrix_logs/kernel_profile.tsv
```

Useful filters:

```bash
column -t -s $'\t' mixed_row_lds_matrix_logs/summary_detail.tsv | head -40
sort -t $'\t' -k6,6nr mixed_row_lds_matrix_logs/summary_detail.tsv | head -20
awk -F'\t' '$6 ~ /center|forward|inverse/ {print}' mixed_row_lds_matrix_logs/kernel_profile.tsv | head -50
```

## Normal CRT comparison

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

## Standalone field modes

```bash
./prmers_opencl_prp 216091 --modulus gf61 --device 1 --iters 1000 --profile-kernels
./prmers_opencl_prp 216091 --modulus gf31 --device 1 --iters 1000 --profile-kernels
```

Standalone half-real can be forced with:

```bash
./prmers_opencl_prp 216091 --modulus gf61 --single-center-mode halfreal --crt-halfreal-flags 48
./prmers_opencl_prp 216091 --modulus gf31 --single-center-mode halfreal --crt-halfreal-flags 48
```

## What to check in profiles

For the current fast mixed odd9 route, the important lines are:

```text
row-core=LDS512
center-single-lds=61:on,31:off
center-split-f48=61:on,31:off
center-f48-delayed-scale=61:off,31:off
stage-single-lds=61:on,31:off
```

Expected kernel families:

```text
crt_mixed_lds512_center_1lds_f48_nonself_61
crt_mixed_lds512_center_1lds_f48_self_61
crt_mixed_lds512_forward_1lds_61
crt_mixed_lds512_inverse_1lds_61
crt_mixed_lds512_center_31
crt_mixed_lds512_forward_31
crt_mixed_lds512_inverse_31
crt_mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat
crt_garner_first_coeffhi_mask32_anybase_x2
```
