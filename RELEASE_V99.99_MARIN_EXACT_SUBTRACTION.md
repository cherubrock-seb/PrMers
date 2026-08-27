# PrMers v99.99 — Marin exact parallel subtraction

Version: `4.20.88-alpha-v99.99-marin-exact-subtraction`

Fixes a correctness bug in the Marin backend exposed by Gaussian-Mersenne ECM
on large transforms. Aevum found the deterministic p=21403643 Stage-1 factor
`482978801775374901713`; Marin previously reached a different projective point.

Root cause: the historical fast weighted subtraction can lose a high
mixed-radix borrow. The serial strong subtraction is exact but far too slow.

v99.99 adds a group-parallel exact subtraction:
1. exact subtraction in each `4*CWM_WG_SZ` digit group,
2. compact cyclic borrow scan across groups,
3. application of the incoming unit borrow only where required.

The algorithm is exactly equivalent to subtraction modulo `2^q-1`.
Aevum is unchanged. Marin fused addsub temporarily uses fast-add plus the exact
subtraction path until an exact fused scan is introduced.
