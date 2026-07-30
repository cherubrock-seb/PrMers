# PrMers v99.95 — complete Gaussian pair pipeline

PrMers version:

    4.20.84-alpha-v99.95-gaussian-pair-full-pipeline

Aevum remains:

    v0.3.78-workload-plan-policy-audit-fix

## Scope

Every native Gaussian work type now accepts a final family selector:

    GM
    GQ
    BOTH

Legacy lines without that selector keep their historical GM behavior.

Supported lines:

    GMTF=p,from_bits,to_bits[,family[,chunk_candidates[,sieve_prime]]]
    GMPROTH=p[,sieve_limit[,family]]
    GMPRP=p[,sieve_limit[,family]]
    GMPMINUS1=p,B1,B2[,base[,sieve_limit[,chunk_bits[,family]]]]
    GMECM=p,B1,B2,curves[,sigma[,sieve_limit[,chunk_bits[,family]]]]
    GMCHAIN=p,pm1_B1,pm1_B2[,ecm_B1[,ecm_B2[,curves[,sieve_limit[,chunk_bits[,finish[,family]]]]]]]

## Backend policy

Trial factoring remains a direct 64-bit OpenCL kernel. It does not use an FFT,
Aevum, or Marin. One candidate pass classifies GM and GQ factors together.

The other modes continue to use the common engine abstraction and the existing
selection policy:

- GM/GQ PRP and GM Proth: PRP workload, normally `throughput:prp`;
- GM/GQ P-1: P-1 workload, normally `throughput:pm1`;
- GM/GQ ECM: ECM workload, normally `throughput:ecm`;
- `-aevum` still forces Aevum;
- `-engine-marin` still forces Marin;
- automatic mode still lets the compatibility and transform policy choose.

A `GMCHAIN` now runs a complete pipeline independently for GM and then GQ. It
reconfigures the workload before each phase, avoids repeating later work for a
family already factorized, and preserves explicitly forced FFT plans.

## Primality semantics

GM keeps the deterministic Proth proof path.

GQ uses a Fermat probable-prime test in both `GMPRP` and the GQ side of
`GMPROTH`. Its JSON therefore reports:

    test_method: fermat-prp
    outcome: probable-prime or composite

It is not mislabeled as a deterministic Proth proof.

## Performance note

GPU TF is genuinely shared between GM and GQ. In v99.95, P-1, ECM, PRP and
Proth are integrated as one assignment and one command, but their GM and GQ
arithmetic is executed sequentially for correctness. This release does not
claim the specialized one-pass GM/GQ primality optimization used by some
Gaussian-aware LLR implementations.

## Results

Schema-v2 result files are written independently per target family, for example:

    gm_pm1_p19_result.json
    gq_pm1_p19_result.json
    gm_ecm_p19_result.json
    gq_ecm_p19_result.json
    gm_prp_p19_result.json
    gq_prp_p19_result.json
    gm_proth_p19_result.json
    gq_proth_p19_result.json

The TF result remains one combined JSON containing a classified `factors[]`
array.

## Ubuntu backend validation

The Ubuntu validation script separates two concerns:

- tiny exponents remain in place for fast arithmetic, JSON and chain checks;
- backend selection is exercised with the real known Gaussian-Mersenne exponent
  `p=3704053`, whose exact lift is `4p=14816212`.

That lifted exponent selects Aevum automatically for PRP, P-1 and ECM policy
checks. Physical P-1 and ECM smoke runs use deliberately tiny bounds, so the
backend is genuinely initialized without turning the release test into a long
primality run.
