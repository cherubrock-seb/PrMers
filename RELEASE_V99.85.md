# PrMers v99.85 stable release and Aevum workload plan audit

This release deliberately excludes the experimental PFA3-M19 branch.

## Validated correctness fixes

- Classic P-1 Stage 2 BSGS keeps the giant step canonical and uses the ordinary
  backend-neutral prepared multiplication path.
- V-trace and classic BSGS both find `55470673` for M569 with B1=9, B2=677 on
  Ubuntu and macOS.
- All P-1 Stage 2 implementations now emit the stable machine-readable line
  `P-1 factor stage 2 found: <factor>` in addition to their historical output.
- ECM stops after a newly discovered factor by default;
  `-ecm-continue-after-factor` continues deterministic later curves.
- P-1 skips a requested Stage 2 after a new Stage 1 factor by default;
  `-pm1-continue-stage2-after-factor` explicitly continues.
- Apple Aevum remains opt-in and is accepted only for validated PRP/LL stock
  Type1 FFT3161. Apple Type4/PFA and Aevum ECM/P-1 requests are rejected before
  GPU execution instead of risking an invariant or Gerbicz failure.

## Workload-specific Aevum plan audit

`scripts/audit_aevum_plans_ubuntu.sh` compares the stock Type1, power-of-two
Type4 and admissible PFA candidates separately for PRP, LL, P-1 Stage 1 and ECM.
A plan is eligible for recommendation only when:

1. its synthetic operation chain is word-exact against Type1;
2. its actual finite P-1/ECM state hash matches the Type1 state;
3. it has no Gerbicz, invariant, kernel or runtime error;
4. it is faster after kernel compilation is excluded from the measured loop.

P-1 Stage 2 V-trace and classic BSGS use Marin and therefore do not select an
Aevum FFT plan.

## Aevum engine

The embedded and standalone engines are identical and report
`v0.3.77-stable-apple-safety-bsgs-fix`.

Candidate lists may be overridden independently with `--large-plans` and
`--factoring-plans`; by default the audit discovers admissible candidates from
`throughput:auto` for each exponent.
