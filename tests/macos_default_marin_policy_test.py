#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
app = (root / "src/core/App.cpp").read_text()
cli = (root / "src/io/CliParser.cpp").read_text()

start = app.index("#if defined(__APPLE__)", app.index("gpu_workload workload"))
split = app.index("#else", start)
end = app.index("#endif", split)
block = app[start:split]

assert "o.aevum ? engine::gpu_backend::aevum : engine::gpu_backend::marin" in block
assert "Marin selected by platform default; use -aevum to opt in" in block
assert "gpu_backend::auto_select" not in block

non_apple = app[split:end]
assert "gpu_backend::auto_select" in non_apple
assert "macOS still defaults to Marin unless -aevum is used" in cli
print("PrMers macOS Marin-default policy test passed")
