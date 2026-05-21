# mixed CRT/PFA odd-row half-real branch

This branch is a GPU proof of concept for the mixed CRT/PFA half-real row path.
It keeps the odd axis outside the power-of-two row transform:

`pack + odd DFT -> half-real row NTT -> odd inverse + preCRT/Garner`

The goal is to make the odd-radix path easy to validate and to compare the LDS
choices per GF field.

## Build

```bash
g++ -O3 -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

## Main validation command

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
  --crt-mixed-row-fuse-both off \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

## Fast benchmark command

```bash
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
  --iters 1000 \
  --profile-kernels
```

## Single-LDS control

The mixed center now has independent controls for GF61 and GF31.
v23 default is the requested test policy:

```bash
# default if unset
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0
```

Global backward-compatible knobs still work:

```bash
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS=0   # force old two-LDS center for both fields
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS=1   # force one-LDS center for both fields
```

Field-specific knobs override the global one:

```bash
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1 \
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0 \
./prmers_opencl_prp ...
```

For the LDS stage before and after the center, v23 adds fixed-size one-LDS
stage kernels for 8, 16, 32, 64, 128, 256, 512 and 1024. GF61 is on by
default, GF31 is opt-in:

```bash
# default if unset
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61=1
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=0
```

When a field single-LDS flag is on, the planner also refuses to silently use a
fused 61x31 LDS stage unless the override below is set.

A fused 61x31 center or stage uses both fields in the same kernel. It can be
forced only when the single-LDS flags are cleared, or with the explicit override:

```bash
PRMERS_CRT_MIXED_CENTER_FUSE_OVERRIDES_SINGLE_LDS=1
PRMERS_CRT_MIXED_STAGE_FUSE_OVERRIDES_SINGLE_LDS=1
# or both at once
PRMERS_CRT_MIXED_FUSE_OVERRIDES_SINGLE_LDS=1
```

## Matrix script

```bash
chmod +x test_mixed_row_lds_matrix.sh
DEVICE=1 P=142606357 GPU_REF=1 VALIDATE=1 VAL_ITERS=1 \
STAGES="512 1024" CENTERS="512" FUSE_BOTHS="off center all" \
./test_mixed_row_lds_matrix.sh
```

Outputs:

- `mixed_row_lds_matrix_logs/all_tests_combined.log`
- `mixed_row_lds_matrix_logs/summary.tsv`
- `mixed_row_lds_matrix_logs/summary_detail.tsv`
- `mixed_row_lds_matrix_logs/kernel_profile.tsv`

The script also accepts:

```bash
SINGLE_LDS_CENTER_61=1
SINGLE_LDS_CENTER_31=0
SINGLE_LDS_STAGE_61=1
SINGLE_LDS_STAGE_31=0
FUSE_OVERRIDES_SINGLE_LDS=0
```

## Kernel names to inspect

Center kernels:

- `gf61_crt_mixed_halfreal_lds512_pair_61`
- `gf61_crt_mixed_halfreal_lds512_pair_31`
- `gf61_crt_mixed_halfreal_lds512_pair_1lds_61`
- `gf61_crt_mixed_halfreal_lds512_pair_1lds_31`
- `gf61_crt_mixed_halfreal_lds_pair_any_1lds_61`
- `gf61_crt_mixed_halfreal_lds_pair_any_1lds_31`

Stage kernels:

- `gf61_crt_lds_stage_dif_pow2_61_1lds_8` ... `gf61_crt_lds_stage_dif_pow2_61_1lds_1024`
- `gf61_crt_lds_stage_dit_pow2_61_1lds_8` ... `gf61_crt_lds_stage_dit_pow2_61_1lds_1024`
- `gf61_crt_lds_stage_dif_pow2_31_1lds_8` ... `gf61_crt_lds_stage_dif_pow2_31_1lds_1024`
- `gf61_crt_lds_stage_dit_pow2_31_1lds_8` ... `gf61_crt_lds_stage_dit_pow2_31_1lds_1024`
- `gf61_crt_lds_stage_dif_pow2_61_512opt`
- `gf61_crt_lds_stage_dit_pow2_61_512opt`
- `gf61_crt_lds_stage_dif_pow2_31_512opt`
- `gf61_crt_lds_stage_dit_pow2_31_512opt`
- `gf61_crt_lds_stage_dif_pow2_61x31_512opt`
- `gf61_crt_lds_stage_dit_pow2_61x31_512opt`
