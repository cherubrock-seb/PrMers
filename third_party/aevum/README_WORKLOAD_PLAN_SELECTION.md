# Aevum workload-specific FFT plan selection

`throughput:auto` is calibrated for long PRP/LL square chains.  It compares:

- stock Type 1 `FFT3161`;
- power-of-two Type 4 `FFT323161` with the validated lead cache;
- native PFA3/PFA9 candidates when admissible.

A smaller transform is not automatically faster, and the fastest PRP plan is
not automatically the fastest ECM or P-1 plan:

- PRP/LL spend almost all hot-loop time in consecutive `square_mul(reg, 1)`;
- P-1 Stage 1 uses a mixed square / multiply-by-3 bit stream plus Gerbicz work;
- ECM keeps about 51 registers and performs add, subtract, square, prepared
  multiplication and prepared-cache eviction/rebuild operations;
- PrMers P-1 Stage 2 V-trace and classic BSGS currently use Marin, so an Aevum
  FFT choice does not apply to those Stage 2 implementations.

PrMers v99.86 now uses separate workload selectors. The RTX 3080 audit moved
P-1 Stage 1 to the exact faster Type4 plan while ECM remains on Type1. A new
candidate is promoted only when it is both:

1. word-exact against Type 1 for the representative workload operation mix;
2. faster in an actual complete ECM/P-1 run after kernel compilation is removed
   from the timing;
3. free of invariant or Gerbicz errors.

Build the differential benchmark with:

```bash
make workload-plan-audit-build
```

Example comparison:

```bash
build-tests/aevum-workload-plan-audit \
  build-engine/libaevum_engine.so 1 55050557 ecm \
  1:1K:4:256:101 4:256:16:256:202 64 .
```

The command prints the resolved plans, transform sizes, timings and canonical
output hashes, then fails if the two plans do not produce the same words.

The PrMers repository also ships `scripts/audit_aevum_plans.py`, which discovers
all candidates, runs these differential tests, compares canonical mathematical
resume fields, repeats real measurements and audits the built-in selectors
`throughput:prp`, `throughput:ll`, `throughput:pm1` and `throughput:ecm`.

## Apple OpenCL 1.2

On Apple, only staged stock Type 1 FFT3161 is currently accepted for Aevum
PRP/LL.  Type 4/PFA and Aevum ECM/P-1 are rejected before execution because the
legacy Apple OpenCL compiler/runtime has not passed the required numerical
validation for those paths.

## RTX 5090 / GB202

A contributor measured a significant gain near exponents 147–148M with:

```text
-fft 1:512:8:512:202
-use INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0
```

This is valuable device-specific tuning data, but it must not be installed as a
global default until the exact GB202 identification, exponent boundaries and
full tune table have been validated.  It should become a GB202-scoped tune
entry, not a universal override.

Explicit candidate lists can be supplied independently because plan shapes are
exponent-dependent:

```bash
./scripts/audit_aevum_plans_ubuntu.sh \
  --large-plans '1:1K:16:256:101,4:512:8:512:202' \
  --factoring-plans '1:1K:4:256:101,4:256:16:256:202,4:512:4:512:202,pfa3:1:512:3:512:101'
```

Normally, omit both options and let `throughput:auto` report all admissible
candidates for each exponent.
