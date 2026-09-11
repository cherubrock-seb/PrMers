#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
api = (root / "third_party/aevum/src/EngineApi.cpp").read_text()
gpu = (root / "third_party/aevum/src/Gpu.cpp").read_text()
host = (root / "src/aevum/EngineAevum.cpp").read_text()
app = (root / "src/core/App.cpp").read_text()
policy = (root / "src/aevum/AutoPolicy.cpp").read_text()
validate = (root / "scripts/aevum_pass4_validate.py").read_text()
bench_sh = (root / "scripts/bench_aevum_pass4.sh").read_text()

checks = {
    "workload-aware ABI": "aevum_engine_create_ex" in api and "create_ex" in host,
    "manual precedence": "manual_plan_env" in api and "compatible_tune_entry" in api,
    "persistent cache": "storeAtomic" in api and "cache hit workload=" in api,
    "bounded candidate set": "AEVUM_AUTOTUNE_MAX_CANDIDATES" in api and "AEVUM_AUTOTUNE_BUDGET_MS" in api,
    "exact differential gate": "WORD MISMATCH" in api and "reference != candidate" in api,
    "PFA excluded from runtime selection": "if (fft.isPfa()) return std::nullopt" in api,
    "measured boundary seed": 'add("4:512:8:512:202")' in api,
    "prepared multiply retained-width primitive": "regMulPreparedStep" in gpu and "carryFused(buf1)" in gpu,
    "PFA excluded from shared bridge": "!useLongCarry && !fft.isPfa()" in gpu,
    "FUSED_LL still present": "carryFusedLL(buf1)" in gpu and "fused_ll_enabled_" in api,
    "PM1/ECM default delegate to plugin": '? plan_override : ""' in app,
    "old boundary force only when tuner off": "!runtime_autotune_enabled()" in policy,
    "structured GPU identity": "AEVUM_DEVICE vendor_id=" in api and "gpu_identification" in validate,
    "compact pass5 artifact": "residue.unlink" in validate and "candidate_timings" in validate and "tuning-output.zip" in bench_sh,
    "bridge cache validation": "bridge_runtime_cache" in validate,
}
missing = [name for name, ok in checks.items() if not ok]
if missing:
    raise SystemExit("Pass4 source audit failed: " + ", ".join(missing))
print("aevum_pass4_source_test: OK")
