# PrMers v99.96 — Gaussian ECM seed fix

This release fixes a GM-ECM-only seed handling bug.

The `-seed` CLI option is parsed into `curve_seed`, while the Gaussian-Mersenne ECM path was incorrectly reading another seed field. Different GM-ECM runs could therefore reuse the same default Suyama curve sequence even when different seeds were requested.

Regular Mersenne ECM is unaffected.

Validation:
- p = 21403643
- B1 = B2 = 1000
- seed = 82620262040
- independently predicted first successful curve = 13
- predicted sigma = 7162284406271848144
- PrMers found factor 148455667849 exactly on curve 13

A regression test is included for the GM-ECM seed path.
