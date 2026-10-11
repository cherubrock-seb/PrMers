#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
hpp = (ROOT / "include/io/CliParser.hpp").read_text()
cli = (ROOT / "src/io/CliParser.cpp").read_text()
app = (ROOT / "src/core/App.cpp").read_text()
bench = (ROOT / "src/core/Bench3.cpp").read_text()
bench_h = (ROOT / "include/core/Bench3.hpp").read_text()
docs = (ROOT / "docs/GPU_BENCHMARKS.md").read_text()

for token in (
    "bool bench3 = false;",
    "bench3_mode",
    "bench3_output",
    "bench3_resume",
    "bench3_plan_timeout_sec",
    "bench3_worker",
    "bench3_worker_exponent",
    "bench3_worker_request",
):
    assert token in hpp
for token in (
    '"-bench3"',
    '"-bench3-mode"',
    '"-bench3-out"',
    '"-bench3-plan-timeout"',
    '"-bench3-no-resume"',
    '"-bench3-worker"',
    '"-bench3-worker-exp"',
    '"-bench3-worker-index"',
    '"-bench3-worker-request"',
    '"-bench3-worker-plan"',
    '"-bench3-worker-partial"',
):
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
assert 'detail.find("workgroup size too large for kernel")' in bench
assert 'r.error = "device workgroup-limit rejection: " + detail' in bench
assert "bench3.json" in bench and "bench3.csv" in bench and "bench3.txt" in bench
assert "BENCH3_RESUME_SKIP" in bench
assert "BENCH3_PROGRESS p=" in bench
assert '"setup_begin"' in bench
assert '"exact_begin"' in bench
assert '"warmup_begin"' in bench
assert '"production_probe_begin"' in bench
assert "bench3.plan.v1" in bench
assert "BENCH3_PLAN_CHECKPOINT" in bench
assert "BENCH3_PLAN_RESUME" in bench
assert "auto-residue.raw" in bench
assert "mpz_out_raw" in bench and "mpz_inp_raw" in bench
assert "partial_point_path" in bench
assert "run_plan_worker_process" in bench
assert "measure_plan_isolated" in bench
assert "BENCH3_PLAN_WORKER_START" in bench
assert "BENCH3_PLAN_WORKER_EXIT" in bench
assert "BENCH3_PLAN_TIMEOUT" in bench
assert 'row.status = "TIMEOUT"' in bench
assert "CreateProcessA" in bench
assert "posix_spawn" in bench
assert "TerminateProcess" in bench
assert "SIGKILL" in bench
assert 'loaded.status == "ERROR" || loaded.status == "TIMEOUT"' in bench
assert "BENCH3_POINT_PARTIAL" in bench
assert "point-status.json" in bench
assert "point-status.csv" in bench
assert "point-status.txt" in bench
assert 'row.status == "TIMEOUT" || row.status == "ERROR"' in bench
assert "int run(const io::CliOptions& options);" in bench_h
assert "## `-bench3`" in docs
print("bench3 source regression: PASS")
