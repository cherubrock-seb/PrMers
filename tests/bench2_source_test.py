#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cli_h = (ROOT / "include/io/CliParser.hpp").read_text()
cli = (ROOT / "src/io/CliParser.cpp").read_text()
app = (ROOT / "src/core/App.cpp").read_text()
bench = (ROOT / "src/core/Bench2.cpp").read_text()
engine_h = (ROOT / "include/marin/engine.h").read_text()
prp = (ROOT / "src/modes/RunPrpOrLlMarin.cpp").read_text()
aevum_h = (ROOT / "include/aevum/EngineAevum.hpp").read_text()

assert "bool bench = false;" in cli_h
assert "bool bench2 = false;" in cli_h
assert '"-bench"' in cli
assert '"-bench2"' in cli
assert '"-bench2-mode"' in cli
assert '"-bench2-out"' in cli
assert '"-bench2-no-resume"' in cli

# Historical benchmark stays a separate path.
assert "rc = runGpuBenchmarkMarin();" in app
assert "rc = core::bench2::run(options);" in app

# Production selector/backend path, not a duplicated selector.
assert "engine::configure_gpu_backend(" in bench
assert "engine::gpu_backend::auto_select" in bench
assert "engine::gpu_workload::prp" in bench
assert "engine::create_gpu(" in bench
assert "kPrpRegisterCount = 8" in bench

# Actual Aevum instantiated-plan truth, not resolver preview.
assert "aevum_engine_active_plan(eng.get())" in bench
assert "aevum_engine_active_plan" in aevum_h

# Required tranche-A output / resume contract.
for token in (
    'kSchemaVersion = "bench2.v1"',
    '"quick"',
    '"standard"',
    '"dense"',
    '"full"',
    '"bench2.json"',
    '"bench2.jsonl"',
    '"bench2.csv"',
    '"bench2.txt"',
    '"records"',
    "atomic_write_record",
    "std::filesystem::rename",
    '"SKIPPED"',
    '"NOT_VALIDATED_IN_BENCH2_TIMING"',
    '"NOT_COLLECTED_IN_TIMING_PHASE"',
):
    assert token in bench, token

# Additive production-PRP timing keeps the validated square-hot-path metric.
for token in (
    "ProductionPrpTiming",
    "measure_production_prp_timing",
    "gerbicz_block",
    "gerbicz_checkpasslevel",
    "gerbicz_full_check_interval",
    "gerbicz_boundary_us",
    "gerbicz_full_check_us",
    "gerbicz_amortized_us_per_iter",
    "production_prp_us_per_iter",
    "production_prp_iterations_per_second",
    "production_prp_estimated_seconds",
    "production_prp_probe_exact",
    "bench2 production PRP Gerbicz probe mismatch",
):
    assert token in bench, token

# Guard the production cadence source of truth against silent drift.
assert "options.gl_block >= 2 ? options.gl_block : 1000" in prp
assert "(1000 * desiredIntervalSeconds) / (double)B" in prp
assert "eng->copy(R3, R1);" in prp
assert "eng->set_multiplicand(R2, R0);" in prp
assert "eng->mul(R1, R2);" in prp

# Wide campaign grid and selector-boundary neighborhood.
for p in (
    "37156667u",
    "58000003u",
    "77232917u",
    "82589933u",
    "100000007u",
    "130000007u",
    "145000007u",
    "150000007u",
    "160000003u",
    "170000003u",
    "175000003u",
    "180000017u",
    "190000003u",
    "196999937u",
    "196999969u",
    "197000003u",
    "200000033u",
    "210000017u",
    "220000013u",
    "230000003u",
    "250000013u",
    "280000027u",
    "300000007u",
    "320000077u",
    "340000019u",
    "360000019u",
    "400000009u",
    "500000003u",
    "600000001u",
):
    assert p in bench, p

assert "static engine * create_gpu" in engine_h

print("bench2 source regression: PASS")
