# mixed CRT/PFA odd-row half-real path

GPU path for odd-radix CRT/PFA with a half-real power-of-two row transform:

```text
pack + odd DFT -> half-real row NTT -> odd inverse + preCRT/Garner
```

## Build

```bash
g++ -O3 -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

## Square validation

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
  --crt-mixed-row-stage 1024 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both all \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

## Benchmark

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 1024 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both all \
  --device 1 \
  --iters 1000 \
  --profile-kernels
```

## Single-LDS controls

Default policy:

```bash
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61=1
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=0
```

Global controls remain available:

```bash
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS=0
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS=1
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS=0
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS=1
```

Field-specific controls override the global value:

```bash
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1 \
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0 \
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61=1 \
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=0 \
./prmers_opencl_prp ...
```

Fused GF61/GF31 kernels can be forced with:

```bash
PRMERS_CRT_MIXED_CENTER_FUSE_OVERRIDES_SINGLE_LDS=1
PRMERS_CRT_MIXED_STAGE_FUSE_OVERRIDES_SINGLE_LDS=1
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

```text
mixed_row_lds_matrix_logs/all_tests_combined.log
mixed_row_lds_matrix_logs/summary.tsv
mixed_row_lds_matrix_logs/summary_detail.tsv
mixed_row_lds_matrix_logs/kernel_profile.tsv
```

Useful script overrides:

```bash
SINGLE_LDS_CENTER_61=1
SINGLE_LDS_CENTER_31=0
SINGLE_LDS_STAGE_61=1
SINGLE_LDS_STAGE_31=0
FUSE_OVERRIDES_SINGLE_LDS=0
```

## Kernels to inspect

Center:

```text
gf61_crt_mixed_halfreal_lds512_pair_61
gf61_crt_mixed_halfreal_lds512_pair_31
gf61_crt_mixed_halfreal_lds512_pair_1lds_61
gf61_crt_mixed_halfreal_lds512_pair_1lds_31
gf61_crt_mixed_halfreal_lds_pair_any_1lds_61
gf61_crt_mixed_halfreal_lds_pair_any_1lds_31
gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_self_61
gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_nonself_61
gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_self_31
gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_nonself_31
```

Stage:

```text
gf61_crt_lds_stage_dif_pow2_61_1lds_8 ... 1024
gf61_crt_lds_stage_dit_pow2_61_1lds_8 ... 1024
gf61_crt_lds_stage_dif_pow2_31_1lds_8 ... 1024
gf61_crt_lds_stage_dit_pow2_31_1lds_8 ... 1024
```


## Split F48 center

For flags 48 and one-LDS center, the code can split the center into non-self and self-pair kernels. This is available for center sizes 8..1024. The fixed 512 kernels are still used for center 512; the generic split kernels are used for the other sizes.

```bash
PRMERS_CRT_MIXED_CENTER_SPLIT_F48_61=1
PRMERS_CRT_MIXED_CENTER_SPLIT_F48_31=1
```

The old names are still accepted:

```bash
PRMERS_CRT_MIXED_CENTER512_SPLIT_F48_61=1
PRMERS_CRT_MIXED_CENTER512_SPLIT_F48_31=1
```

Disable it with `PRMERS_CRT_MIXED_CENTER_SPLIT_F48_61=0` or `PRMERS_CRT_MIXED_CENTER_SPLIT_F48_31=0`.
