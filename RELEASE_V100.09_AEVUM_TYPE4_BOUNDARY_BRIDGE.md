# PrMers v100.09 — Aevum Type4 PRP boundary bridge

This release fixes the next plan-selection problem exposed by issue #36 retesting.

Ordinary PRP still starts from Aevum's native Type1 automatic selector. PrMers
considers the upstream-style Type4 plan `4:1K:8:256:101` only when it is a
real transform-size bridge over native Type1.

The promotion requires all of the following:

- ordinary PRP plugin-auto request;
- native Aevum selected Type1;
- Type4 is at least 1.5x smaller than native Type1;
- Type4 is at or below 46.97 bpw.

At 147800003 both Type1 and Type4 are 4M, so issue #36 behavior is preserved:
Type1 remains selected and Aevum receives `plugin-auto`.

At 196999969 native Type1 is 8M while `4:1K:8:256:101` is 4M at ~46.97 bpw,
so the bridge selects and actually runs the 4M Type4 plan.

At 220000001 the same 4M plan is above the measured bpw boundary, so PrMers
returns to plugin-native Type1. PFA9 remains available explicitly for separate
experimentation above the Type4 bridge window.

Pre-release Radeon VII measurements at 196999969:

- forced Type4 4M: ~520 IPS after warm-up;
- PFA9 9M: ~263 IPS;
- native/default Type1 8M: ~243 IPS.

The bridge is PRP-only in v100.09. LL is intentionally unchanged until the
same high-bpw Type4 geometry is validated for the LL operation mix.

The embedded Aevum plugin code is unchanged in this release; this is a PrMers
selection/runtime-routing correction.
