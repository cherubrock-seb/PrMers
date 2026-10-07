#!/usr/bin/env python3
"""Source-level regression checks for small ECM fixes."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
json_src = (root / 'src/io/JsonBuilder.cpp').read_text()
mont = (root / 'src/modes/RunEcm.cpp').read_text()
te = (root / 'src/modes/RunEcmTwistedEdwards.cpp').read_text()


def test_json_base_seed():
    m = re.search(r'"base-seed[^;]*;', json_src)
    assert m, 'base-seed field missing'
    assert 'opts.base_seed' in m.group(0), 'base-seed must report the base seed, not the curve seed'


def test_te_sigma_forces_one_curve():
    assert '(forceCurve && !forcedSeedSeries) || forceSigma) curves = 1ULL;' in te, \
        '-sigma must force a single curve in the twisted Edwards path as in the Montgomery path'


def test_gcd_n_drops_checkpoints():
    for name, src in (('RunEcm.cpp', mont), ('RunEcmTwistedEdwards.cpp', te)):
        assert 'singular or failure, retrying' not in src, name + ': gcd=N skips the curve, it does not retry it'
        assert 'gcd=N, retrying' not in src, name
        assert 'stage 1 gcd=N' in src, name
        for m in re.finditer(r'stage 1 gcd=N[^\n]*\n(.*?)continue;', src, re.S):
            assert 'fs::remove(ckpt_file' in m.group(1), name + ': stage-1 gcd=N leaves the checkpoint behind'


test_json_base_seed()
test_te_sigma_forces_one_curve()
test_gcd_n_drops_checkpoints()
print('PrMers ECM small-items source test passed')
