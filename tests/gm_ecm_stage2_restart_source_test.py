#!/usr/bin/env python3
"""Legacy GM ECM: after an unusable Stage 2 checkpoint the Stage 1 point is recomputed;
that point must get the same factor / singular-point checks as the original Stage 1
point (a factor found while restarting must not be dropped, and a non-normalized
point must not seed Stage 2).

Source-level check: the situation needs a CRC-valid Stage 2 checkpoint whose point is
degenerate, which the writer never produces."""
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / 'src/modes/RunGaussianMersenneFactor.cpp').read_text()
start = src.index('"GM ECM Stage 1 restart curve "')
end = src.index('Resuming ECM Stage 2 at prime index', start)
block = src[start:end]
assert 'point = project_point(eng.get(), r, t.n);' in block
after = block[block.index('point = project_point(eng.get(), r, t.n);'):]
assert 'is_proper_factor(point.factor, t.n)' in after, 'restarted Stage 1 factor is ignored'
assert 'write_json_result(' in after, 'restarted Stage 1 factor is not recorded'
assert '!point.normalized' in after and 'continue;' in after, 'singular restarted point reaches Stage 2'
print('PrMers GM ECM Stage 2 restart check test passed')
