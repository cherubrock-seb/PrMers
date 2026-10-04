# PrMers AUTO vs tuned prpll

Status

This report records the completed representative comparison and the subsequent
AUTO-selector validation. Completed benchmark measurements are preserved rather
than rerun.

The production comparison used:

PrMers benchmark commit 25e35d155a8593aae1a4ed1f887b0c375c42e365
(vendor: sync Aevum v0.3.93-pass5-prp-max);
tuned prpll commit eb5999cce238f08bfd0f3792bd3e6854ee687d0f
(PRPLL v8.0-397-geb5999c);
Radeon VII as OpenCL device 0;
RTX 3080 as OpenCL device 2.

The later p210 strategic-seed selector fix was validated as
cc524af795ef7b651d588aeb37bb3f606f116d6c, ported without GPU-name
branching, and merged to PrMers main by merge commit
045191c286579d3c88e0fe36968e6f21a9d983d1.

Apples-to-apples prpll audit

The initially large PrMers/prpll gaps were treated as suspicious and audited
before publication.

The persisted evidence establishes that the prpll reference is genuine and
directly comparable:

exact prpll executable:
.cherub-worktrees/prpll-current-r947/build-release/prpll;
executable source commit:
eb5999cce238f08bfd0f3792bd3e6854ee687d0f;
backend: OpenCL on the same physical devices used by PrMers;
requested FFT plans are also the plans reported as executed by prpll;
transform sizes and variants match the requested plans;
Type 4 is FFT323161; for the key p170 comparison both programs execute
4:512:8:512:202, report a 4M FFT, use CARRY64, and produce the same RES64;
setup/JIT is outside the steady-state iteration timing used for the
production comparison;
correctness residues match at all four representative exponents.

The p170 repeated same-plan evidence is especially discriminating:

GPU	PrMers/Aevum samples, us/iter	median	prpll samples, us/iter	median
Radeon VII	2454.048, 2735.903, 2372.817	2454.048	5866, 5848, 5842	5848
RTX 3080	925.840, 940.743, 940.725	940.725	2905, 2318, 2319	2319

Aevum production-loop IPS is consistent with these timings. The comparison is
therefore not a cold-start/JIT artifact and not a superficially similar FFT
string selecting another prpll implementation.

Persisted production throughput matrix

These are the completed steady-state production measurements. The AUTO column
is the production AUTO state at the benchmark point; Best PrMers is the
fastest correctness-valid PrMers plan measured in the same campaign. prpll
values are the persisted tuned-prpll medians.

GPU	p	production AUTO plan	AUTO us/iter	Best PrMers plan	Best us/iter	tuned prpll plan	prpll us/iter	AUTO time reduction vs prpll	Best time reduction vs prpll
Radeon VII	150000001	1:1K:8:256:101	1634.361	1:512:8:512:101	1593.473	1:512:8:512:101	4425	63.065%	63.989%
RTX 3080	150000001	1:1K:8:256:101	788.121	1:512:8:512:101	706.030	1:512:8:512:101	1812	56.505%	61.036%
Radeon VII	170000009	1:1K:16:256:101	5165.289	4:512:8:512:202	2454.048	4:512:8:512:202	5848	11.674%	58.036%
RTX 3080	170000009	1:1K:16:256:101	1584.359	4:512:8:512:202	940.725	4:512:8:512:202	2319	31.679%	59.434%
Radeon VII	197000003	1:1K:16:256:101	5176.519	4:512:8:512:202	2599.901	4:512:8:512:202	6011	13.883%	56.748%
RTX 3080	197000003	1:1K:16:256:101	1602.872	4:512:8:512:202	937.902	4:512:8:512:202	2330	31.207%	59.747%
Radeon VII	210000017	1:1K:8:512:101	3960.082	1:1K:8:512:101	3960.082	1:1K:8:512:101	6977	43.241%	43.241%
RTX 3080	210000017	1:1K:16:256:101	1619.643	1:1K:8:512:202	1526.112	1:1K:8:512:202	2769	41.508%	44.886%

Across these eight production points:

historical production AUTO geometric-mean time is 0.6096x tuned prpll,
i.e. 39.04% lower iteration time or 1.640x throughput;
Best PrMers geometric-mean time is 0.4358x tuned prpll,
i.e. 56.42% lower iteration time or 2.295x throughput.

These geometric means use only production-loop measurements. Autotune
comparePlans medians are not mixed into them.

Correctness

Canonical residues used by the comparison:

p	RES64
150000001	37cbda4ed677020b
170000009	106376418c42599a
197000003	b614a0a6a21c1f4e
210000017	aae43412d514e705

At p210, 4:512:8:512:202 is correctness-invalid in the current Aevum path
and remains excluded regardless of its timing.

Sanitized AUTO selector validation

The later selector work addresses the historical AUTO misses without
hardcoding a Radeon or NVIDIA winner. Runtime exact comparePlans remains
authoritative.

GPU	p	sanitized / post-fix AUTO decision	selector-local result
Radeon VII	150000001	1:512:8:512:101	1.0884x vs native
RTX 3080	150000001	1:512:8:512:202	1.1015x vs native
Radeon VII	170000009	4:512:8:512:202	1.7744x vs native
RTX 3080	170000009	4:512:8:512:202	1.5722x vs native
Radeon VII	197000003	4:512:8:512:202	2.2758x vs native
RTX 3080	197000003	4:512:8:512:202	1.6506x vs native
Radeon VII	210000017	1:1K:8:512:101	1.4050x vs native
RTX 3080	210000017	1:512:16:512:202	1.0672x vs native

For PRP exponents 198000000 through 230000000, the merged fix guarantees
early evaluation of the measured-safe Type1 strategic seeds:

1:1K:8:512:202
1:1K:8:512:101
1:512:16:512:202

This prevents cold-start Gpu::make/JIT cost from exhausting the normal
wall-clock tuning budget before the strategically important candidates are
measured.

The p210 post-fix validation measured all three strategic seeds on both GPUs.
Radeon VII independently selected 1:1K:8:512:101; RTX 3080 independently
selected 1:512:16:512:202. All selected candidates passed exact comparison,
and invalid Type4 was not admitted.

The selector-local speedups above are not substituted into the production
throughput matrix. A new post-fix production geomean would require a complete
steady-state production timing set for the newly selected plans; this report
does not manufacture such a number from autotune timings.

Conclusion

Two conclusions are empirically supported:

On the completed representative steady-state production matrix, PrMers/Aevum
is materially faster than the audited tuned-prpll reference on both Radeon
VII and RTX 3080.
The later AUTO work fixes the demonstrated selector-coverage problem,
including the p210 cold-start budget defect, while preserving manual/tune
precedence, cache safety, correctness gates, p170 Type4 behavior, and the
p210 invalid-Type4 exclusion.

The next performance work should return to kernel-level optimization rather
than further selector rediscovery.
