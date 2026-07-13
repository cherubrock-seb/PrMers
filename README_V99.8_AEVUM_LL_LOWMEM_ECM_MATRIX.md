# PrMers v99.8 — Aevum LL, P-1 low-memory, ECM and validation matrix

## Purpose

This update keeps Aevum available wherever the arithmetic path uses the generic
`engine::Reg` operations. It does not globally replace failing paths with Marin.
Instead, backend compatibility and performance selection are modeled separately.

## Backend routing

| Workload | Automatic rule | Forced Aevum |
|---|---|---|
| PRP | Aevum when `Aevum transform / Marin transform <= 1.00` | Supported when an FFT3161 plan exists; otherwise the fallback is logged explicitly |
| LL-safe / LL-unsafe / LL-safe2 | Same rule as PRP | Supported through the common engine interface |
| P-1 normal Stage 1 | Aevum when ratio `<= 0.75` | Supported |
| P-1 multi-register Stage 2 | Aevum when ratio `<= 1.00` | Supported |
| P-1 low-memory, 3 registers | Aevum when ratio `<= 0.75` | Supported |
| P-1 ultra-low-memory, 1 register | Marin compatibility route | Rejected before plugin allocation because this algorithm depends on Marin `fast3` |
| ECM | Aevum when ratio `<= 0.75` | Supported for Edwards and Montgomery paths |

All decisions are printed with the workload, selected engine, transform sizes,
ratio, threshold and FFT3161 plan. The Web GUI receives the same data.

## Corrections

- LL-safe, LL-unsafe and LL-safe2 now report the engine actually used and write
the active engine transform size in their result JSON.
- LL/PRP checkpoints contain mode and backend tags. LL-safe2 and LL-unsafe use
distinct checkpoint filenames.
- P-1 low-memory remains a 3-register generic-engine path and can use Aevum.
- P-1 ultra-low-memory is selected as Marin before engine allocation in Auto mode;
forced Aevum exits cleanly with code 2.
- The tiny Marin radix-4 kernels use `r2_2` and `r2i_2`. This fixes the OpenCL
compile error seen for very small transforms in P-1 low-memory and ECM.
- ECM no-factor exit code 1 is accepted by the GPU routing tests as a valid
completed calculation.
- Vendored Aevum reports `v0.3.4` instead of the parent PrMers commit hash.

## Exhaustive report

```bash
PRMERS_TEST_DEVICE=0 \
PRMERS_MATRIX_PROFILE=standard \
make test-backend-matrix
```

For complete Aevum/Marin result comparisons on medium sizes:

```bash
PRMERS_TEST_DEVICE=0 \
PRMERS_MATRIX_PROFILE=full \
PRMERS_MATRIX_COMPLETE_SECONDS=1200 \
make test-backend-matrix
```

Restrict a rerun to selected cases:

```bash
PRMERS_MATRIX_CASE_FILTER='ll-safe|pm1-lowmem|ecm-medium' \
./tests/run_backend_validation_matrix.sh 0 full
```

The generated `.tar.gz` contains commands, individual logs, system information,
`summary.tsv`, `comparisons.tsv`, `errors.tsv`, `combined.log`, and a manifest.
