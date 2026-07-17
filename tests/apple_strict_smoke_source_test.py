#!/usr/bin/env python3
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "scripts/smoke_aevum_ll_macos.py"
source = path.read_text(encoding="utf-8")
ast.parse(source)

required = [
    're.findall(r"Iter:\\s*(\\d+)", text)',
    'if max_iter >= 1 and markers_ready:',
    'Loaded fftWGF61WidthFinalApple',
    'Loaded tailMulGF61LoadScalarApple',
    'Loaded tailMulGF61ReverseStockApple',
    'Loaded tailMulGF61PairStockApple',
    '"gpu read failed" in lowered',
    '"gpu double-read mismatch" in lowered',
    'Aevum macOS smoke passed at Iter >= 1',
]
for needle in required:
    assert needle in source, needle

assert "time.monotonic() - progress_seen_at >= 15" not in source
assert 'if "Progress:" not in text' not in source

print("PrMers Apple strict real-iteration smoke source test passed")
