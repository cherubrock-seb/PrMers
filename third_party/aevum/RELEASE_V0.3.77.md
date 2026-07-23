# Aevum v0.3.77 stable Apple safety and workload audit

- keeps the validated non-Apple Type 1, Type 4, PFA and throughput-auto paths;
- rejects unsupported Apple Type 4/PFA plans before kernel creation;
- documents that Apple Aevum ECM/P-1 is not numerically validated;
- retains the Apple prepared-cache and staged-kernel safety changes;
- adds `workload-plan-audit-build` and a word-exact workload plan comparator;
- documents workload-specific plan selection and the proposed GB202 tuning.

The M19 experimental branch is not part of this release.
