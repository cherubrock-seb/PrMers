#!/usr/bin/env python3
"""Plan autotune must benchmark the production register paths, in an OpenCL
context of its own, and stop at the first engine error."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
engine = (root / "src/EngineApi.cpp").read_text()


def body(signature):
    start = engine.index(signature)
    brace = engine.index("{", start)
    depth = 0
    for i in range(brace, len(engine)):
        if engine[i] == "{":
            depth += 1
        elif engine[i] == "}":
            depth -= 1
            if depth == 0:
                return engine[brace:i + 1]
    raise SystemExit(f"unterminated body: {signature}")


# One source of truth for the fused-LL / lead-cache switches.
sequence = body("void runPlanSequence(")
for capability in ("regSupportsFusedLL", "regSupportsLeadCache"):
    if capability in sequence:
        raise SystemExit(f"runPlanSequence decides {capability} itself instead of using the production switches")
if "paths.fused_ll" not in sequence or "paths.lead_cache" not in sequence:
    raise SystemExit("runPlanSequence does not follow the production RegPaths")

compare = body("PlanComparison comparePlans(")
if compare.count("productionRegPaths(") != 2:
    raise SystemExit("comparePlans must derive both engines' paths from productionRegPaths")

if not re.search(r"const RegPaths reg_paths = productionRegPaths\(\*gpu_, fft, device_name\);", engine):
    raise SystemExit("the production Runtime must take its switches from productionRegPaths")
if re.search(r"fused_ll_enabled_ = gpu_->regSupportsFusedLL\(\)", engine):
    raise SystemExit("the production Runtime decides fused LL outside productionRegPaths")

# Candidates run in a tuning context, never on the production one.
if "comparePlans(exponent_, workload_, shared_," in engine:
    raise SystemExit("plan autotune benchmarks candidates on the production OpenCL context")
if "comparePlans(exponent_, workload_, tune->shared," not in engine:
    raise SystemExit("plan autotune does not use its tuning context")
if "comparePreparedMulLead(*gpu_" in engine:
    raise SystemExit("the prepared-multiply bridge benchmark runs on the production engine")

# An engine error ends the search; it is not logged as a rejection and carried on from.
loop_start = engine.index("for (size_t candidate_index = 0;")
loop = engine[loop_start:engine.index("tune.reset();", loop_start)]
if "rejected (" in loop:
    raise SystemExit("an engine error during autotune is still treated as a candidate rejection")
if not re.search(r"if \(!failure\.empty\(\)\) \{.*?break;", loop, re.S):
    raise SystemExit("plan autotune does not stop at the first engine error")

print("engine_autotune_isolation_test: OK")
