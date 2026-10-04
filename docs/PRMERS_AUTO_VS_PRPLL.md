# PrMers AUTO vs tuned prpll

This note consolidates the completed PRP benchmark audit and the high-range runtime-selector fix.

Historical benchmark rows come from benchmarks/prmers-vs-prpll-auto/results.tsv. They are preserved measurements and are not rewritten using later runtime-autotuner microbenchmarks.

## Historical benchmark aggregate

- Rows: 8
- Geometric-mean speedup of historical PrMers AUTO over tuned prpll: 1.6405x (64.05% faster).
- Geometric-mean speedup of historical best measured PrMers plan over tuned prpll: 2.2947x (129.47% faster).

The historical selector_miss field means that old normal AUTO was slower than a separately benchmarked PrMers plan. It is distinct from the later sanitized runtime-autotuner validation, where candidate plans are exact-checked and timed inside Aevum.

## Historical benchmark rows

| p | GPU | old AUTO plan | AUTO us/iter | best measured PrMers plan | best us/iter | tuned prpll plan | prpll us/iter | AUTO vs prpll | best vs prpll | correctness |
|---:|---|---|---:|---|---:|---|---:|---:|---:|---|
| 150000001 | Radeon VII | 1:1K:8:256:101 | 1634.361 | 1:512:8:512:101 | 1593.473 | 1:512:8:512:101 | 4425 | 63.065% | 63.989% | PASS_RES64_37cbda4ed677020b |
| 150000001 | RTX 3080 | 1:1K:8:256:101 | 788.121 | 1:512:8:512:101 | 706.030 | 1:512:8:512:101 | 1812 | 56.505% | 61.036% | PASS_RES64_37cbda4ed677020b |
| 170000009 | Radeon VII | 1:1K:16:256:101 | 5165.289 | 4:512:8:512:202 | 2454.048 | 4:512:8:512:202 | 5848 | 11.674% | 58.036% | PASS_RES64_106376418c42599a |
| 170000009 | RTX 3080 | 1:1K:16:256:101 | 1584.359 | 4:512:8:512:202 | 940.725 | 4:512:8:512:202 | 2319 | 31.679% | 59.434% | PASS_RES64_106376418c42599a |
| 197000003 | Radeon VII | 1:1K:16:256:101 | 5176.519 | 4:512:8:512:202 | 2599.901 | 4:512:8:512:202 | 6011 | 13.883% | 56.748% | PASS_RES64_b614a0a6a21c1f4e |
| 197000003 | RTX 3080 | 1:1K:16:256:101 | 1602.872 | 4:512:8:512:202 | 937.902 | 4:512:8:512:202 | 2330 | 31.207% | 59.747% | PASS_RES64_b614a0a6a21c1f4e |
| 210000017 | Radeon VII | 1:1K:8:512:101 | 3960.082 | 1:1K:8:512:101 | 3960.082 | 1:1K:8:512:101 | 6977 | 43.241% | 43.241% | PASS_RES64_aae43412d514e705 |
| 210000017 | RTX 3080 | 1:1K:16:256:101 | 1619.643 | 1:1K:8:512:202 | 1526.112 | 1:1K:8:512:202 | 2769 | 41.508% | 44.886% | PASS_RES64_aae43412d514e705 |

## Sanitized runtime-selector evidence

- p150: runtime AUTO exercised exact candidate comparison on Radeon VII and RTX 3080.
- p170: both GPUs selected the valid 4M Type4 plan 4:512:8:512:202.
- p197: both GPUs selected the valid 4M Type4 plan 4:512:8:512:202.
- p210 exposed a cold-start coverage defect: wall-clock budget termination could occur before all important high-range Type1 candidates were reached.
- 4:512:8:512:202 is correctness-invalid at p210 from established exact validation and remains forbidden as a winner.

## High-range selector fix

- 2796127fc9e9c75f29bc5de7e040fdde5cc55b56 - prioritize strategic high-range autotune seeds.
- cc524af795ef7b651d588aeb37bb3f606f116d6c - scope that promotion to PRP.

For PRP exponents 198M through 230M, the runtime tuner promotes these measured-safe Type1 seeds before ordinary wall-clock termination:

1. 1:1K:8:512:202
2. 1:1K:8:512:101
3. 1:512:16:512:202

This changes candidate ordering, not the winner policy. admissiblePlan and runtime comparePlans remain authoritative. There is no Radeon/NVIDIA name branch.

## p210 post-fix validation

Both post-fix runs used p=210000017, no explicit FFT, a fresh private autotune cache, and the production default candidate cap/budget. All three strategic candidates reached runtime exact comparison before normal budget termination.

| GPU | candidate | speedup vs native in selector |
|---|---|---:|
| Radeon VII | 1:1K:8:512:202 | 1.3712x |
| Radeon VII | 1:1K:8:512:101 | 1.4050x |
| Radeon VII | 1:512:16:512:202 | 0.9950x |
| RTX 3080 | 1:1K:8:512:202 | 1.0376x |
| RTX 3080 | 1:1K:8:512:101 | 1.0414x |
| RTX 3080 | 1:512:16:512:202 | 1.0672x |

- Radeon VII selected 1:1K:8:512:101 at 1.4050x versus native.
- RTX 3080 independently selected 1:512:16:512:202 at 1.0672x versus native.
- All three strategic Type1 candidates were exact on both GPUs.
- The known-invalid p210 Type4 plan was not selected.
- Different Radeon and RTX winners demonstrate generic search coverage rather than a hardcoded device-specific result.

RTX post-fix validation was completed on cherubrock2. The earlier attempt on cherubrock1 was blocked because that host temporarily exposed no NVIDIA PCI/OpenCL device; that was an external host condition, not a PrMers failure.

## Conclusion

1. PrMers is substantially faster than the tuned prpll reference across the retained eight-row benchmark matrix.
2. The p210 defect was bounded-search coverage, not a correctness shortcut. Promoting three safe strategic Type1 seeds closes that gap while preserving exact runtime comparison.
3. The fix remains generic: Radeon VII and RTX 3080 evaluated the same strategic set and independently selected different fastest plans.

Selector/source validation is complete on both target GPU families. Release publication still requires the normal CI and release workflow.
