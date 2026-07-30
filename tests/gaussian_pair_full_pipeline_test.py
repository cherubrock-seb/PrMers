#!/usr/bin/env python3
from pathlib import Path


def chi2(p: int) -> int:
    return 1 if p & 7 in (1, 7) else -1


def gm(p: int) -> int:
    return (1 << p) - chi2(p) * (1 << ((p + 1) // 2)) + 1


def gq(p: int) -> int:
    num = (1 << p) + chi2(p) * (1 << ((p + 1) // 2)) + 1
    assert num % 5 == 0
    return num // 5

for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 47):
    a, b = gm(p), gq(p)
    assert a * (5 * b) == (1 << (2 * p)) + 1
    assert ((1 << (4 * p)) - 1) % a == 0
    assert ((1 << (4 * p)) - 1) % b == 0

assert gm(19) % 525313 == 0
assert gq(19) % 457 == 0

root = Path(__file__).resolve().parents[1]
prp = (root / "src/modes/RunGaussianMersenne.cpp").read_text()
factor = (root / "src/modes/RunGaussianMersenneFactor.cpp").read_text()
app = (root / "src/core/App.cpp").read_text()
cli = (root / "src/io/CliParser.cpp").read_text()
work = (root / "src/io/WorktodoParser.cpp").read_text()

for source in (prp, factor):
    assert 'requested_family == "BOTH"' in source or 'requested == "BOTH"' in source
    assert r'\"target_family\"' in source
    assert r'\"schema_version\": 2' in source

assert 'test_method' in prp
assert 'output_mode_tag = options.gm_prp_only ? "prp" : "proth"' in prp
assert 'fermat-prp' in prp
assert 'proth' in prp
assert '-gm-family' in cli
assert 'GM|GQ|BOTH' in work

# Backend policy remains centralized. Pair PRP/Proth, P-1, and ECM still use
# the same engine workloads and engine::create_gpu dispatch as before.
for needle in ('throughput:prp', 'throughput:pm1', 'throughput:ecm'):
    assert needle in app
assert prp.count('engine::create_gpu(lift_exponent') == 1
assert factor.count('engine::create_gpu(t.lift') == 2

# TF deliberately remains its own 64-bit direct OpenCL kernel rather than a
# transform workload.
tf = (root / "src/modes/RunGaussianTrialFactor.cpp").read_text()
assert 'gm_trial_factor.cl' in tf
assert 'engine::create_gpu' not in tf

print("Gaussian pair full-pipeline mathematics and backend-policy audit passed")
