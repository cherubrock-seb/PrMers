# R6.1 final hardware gate

R6.1 preserves the R6 PRP shape + bounded/resumable `-use` autotuner and removes the rejected `PRP_CARRY_EPOCH` experiment completely from production sources. `src/cl/carryfused.cl` is byte-for-byte the Pass-4 baseline again.

Required final hardware gate:

- RTX 3080 p=21000029: automatic completed-positive; selected `-use` same-version gain >= 1.03; Pass4 non-regression; final cache hit.
- Radeon VII p=147800003: defaults-complete or validated positive; Pass4 non-regression >= 0.985 median gate.
- Radeon VII p=150000007: the old false +18% winner must NOT be promoted unless the engine-style gate independently gives >= 1.03; Pass4 non-regression.
- Radeon VII p=210000017: preserve the ~+6% real `-use` winner if it still passes the strict engine gate; Pass4 non-regression; cache hit.

No release until all four ranges PASS.
