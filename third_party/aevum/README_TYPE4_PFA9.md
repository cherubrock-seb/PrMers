# Aevum v0.3.66 force-adaptive FFT323161 + PFA9

FFT type 4 is deliberately not part of automatic selection.  It can be
requested explicitly with:

```text
pfa9:4:512:9:512:202
```

In exp11 that ordinary request is optimized before GPU creation.  At
M175000039 the requested 4.50M shape is only 37.09 bpw, below the measured
38.95-bpw exact GF31+GF61 limit, so it executes:

```text
pfa9:1:512:9:512:202
```

This removes the redundant FP32 transform and recovers the approximately
770-IPS paired-NTT performance seen on RTX 3080.  The genuine three-plane path
is still available for validation or exponents that exceed the paired-NTT
capacity:

```text
pfa9full:4:512:9:512:202
```

## Ubuntu build and validation

```bash
sudo apt update
sudo apt install -y build-essential clang ocl-icd-opencl-dev opencl-headers
make -j"$(nproc)" engine-lib
./scripts/test_type4_pfa9_ubuntu.sh 1 175000039 2
```

The differential test compares the exact paired-NTT path against the forced
full three-plane path word for word.  The host policy test also verifies that
ordinary `pfa9:4` elides FP32 while `pfa9full:4` does not.
