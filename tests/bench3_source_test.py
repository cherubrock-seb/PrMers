#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
hpp = (ROOT / "include/io/CliParser.hpp").read_text()
cli = (ROOT / "src/io/CliParser.cpp").read_text()
app = (ROOT / "src/core/App.cpp").read_text()
bench = (ROOT / "src/core/Bench3.cpp").read_text()
bench_h = (ROOT / "include/core/Bench3.hpp").read_text()
docs = (ROOT / "docs/GPU_BENCHMARKS.md").read_text()

for token in ("bool bench3 = false;", "bench3_mode", "bench3_output", "bench3_resume"):
    assert token in hpp
for token in ('"-bench3"', '"-bench3-mode"', '"-bench3-out"', '"-bench3-no-resume"'):
    assert token in cli
assert '#include "core/Bench3.hpp"' in app
assert "core::bench3::run(options)" in app
assert "if (!o.gaussian_mersenne && !o.bench && !o.bench2 && !o.bench3)" in app
assert "!hasWorktodoEntry_ && o.exponent == 0 &&\n        !o.bench && !o.bench2 && !o.bench3" in app
assert "!options.bench && !options.bench2 && !options.bench3 && options.filemers.empty()" in app
assert "bench3.v1" in bench
assert "pow2:auto" in bench
assert "pfa:auto" in bench
assert "pfa:3" in bench
assert "pfa7:1:512:7:512:202" in bench
assert "pfa:9" in bench
assert "aevum_engine_resolve_fft" in bench
assert "engine::gpu_backend::auto_select" in bench
assert "engine::gpu_backend::aevum" in bench
assert "engine::gpu_workload::prp" in bench
assert "engine::create_gpu" in bench
assert "exact_residue" in bench
assert "production_timing" in bench
assert 'detail.find("small-factor capacity exceeded")' in bench
assert 'r.status = "SKIPPED_ILLEGAL"' in bench
assert 'r.error = "PRP factor-safe rejection: " + detail' in bench
assert "bench3.json" in bench and "bench3.csv" in bench and "bench3.txt" in bench
assert "BENCH3_RESUME_SKIP" in bench
assert "int run(const io::CliOptions& options);" in bench_h
assert "## `-bench3`" in docs
print("bench3 source regression: PASS")
