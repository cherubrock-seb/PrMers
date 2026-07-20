# PrMers v99.73 force-adaptive FFT323161 + PFA9

FFT type 4 remains explicit and is never selected by the normal automatic
policy.  Both commands below now use the optimized capacity-aware policy:

```bash
./prmers 175000039 -d 1 -pfa9-type4 -proof 0
./prmers 175000039 -d 1 -aevum-fft pfa9:4:512:9:512:202 -proof 0
```

At M175000039 they resolve to the exact faster plan:

```text
pfa9:1:512:9:512:202
```

The true FP32 + GF31 + GF61 path is diagnostic-only:

```bash
./prmers 175000039 -d 1 -pfa9-type4-full -proof 0
# equivalent plan: pfa9full:4:512:9:512:202
```

Run the complete Ubuntu build, exact differential validation and isolated
benchmarks with:

```bash
./scripts/test_type4_optimized_ubuntu.sh 1 180 175000039
```
