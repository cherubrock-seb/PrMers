#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
app = (root / "src/core/App.cpp").read_text()
cli = (root / "src/io/CliParser.cpp").read_text()

policy_marker = app.index("// Apple ships the legacy OpenCL 1.2 stack")
start = app.rindex("#if defined(__APPLE__)", 0, policy_marker)
split = app.index("#else", policy_marker)
end = app.index("#endif", split)
block = app[policy_marker:split]

assert "o.aevum ? engine::gpu_backend::aevum : engine::gpu_backend::marin" in block
assert "Marin selected by platform default; use -aevum to opt in" in block
assert "gpu_backend::auto_select" not in block
assert 'o.aevum_fft_spec = "pow2:auto"' in app
assert "PRP/LL uses staged stock Type1 FFT3161" in app

non_apple = app[split:end]
assert "gpu_backend::auto_select" in non_apple
assert "macOS still defaults to Marin unless -aevum is used" in cli
print("PrMers macOS Marin-default policy test passed")
