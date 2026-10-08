#!/usr/bin/env python3
"""Plan autotune must benchmark the production register paths, in an OpenCL
context of its own, skip a candidate the engine refuses to build, and stop at
the first engine error."""
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

# A candidate the engine refuses to build is skipped, and the search goes on.  Only the
# candidate's own Gpu construction is classified; the native build and the benchmark are not.
if "struct CandidateRefused : std::runtime_error" not in engine:
    raise SystemExit("no distinct exception for a refused candidate plan")
if "Gpu::make(exponent, shared, native_fft, {}, false)" not in compare:
    raise SystemExit("comparePlans no longer builds the native engine directly")
if "makeCandidateGpu(exponent, shared, candidate_fft)" not in compare or "Gpu::make(exponent, shared, candidate_fft" in compare:
    raise SystemExit("comparePlans does not build the candidate through makeCandidateGpu")
if "CandidateRefused" in compare:
    raise SystemExit("comparePlans turns errors after candidate construction into refusals")
make = body("std::unique_ptr<Gpu> makeCandidateGpu(")
cuda = make[make.index("#if defined(CUDA_BACKEND)"):make.index("#else")]
if "try" in cuda or "CandidateRefused" in cuda:
    raise SystemExit("CUDA driver errors cannot be told apart from refusals; they must stop the search")
opencl = make[make.index("#else"):]
order = [opencl.index(c) for c in ("catch (const gpu_error&) {\n    throw;",
                                    "catch (const std::bad_alloc&) {\n    throw;",
                                    "catch (const std::exception& e) {\n    throw CandidateRefused(")]
if order != sorted(order):
    raise SystemExit("OpenCL errors and allocation failures must pass through before the refusal catch")
clwrap_h = (root / "src/clwrap.h").read_text()
if "class gpu_error : public std::runtime_error" not in clwrap_h:
    raise SystemExit("gpu_error must be visible to EngineApi")
refused = re.search(r"catch \(const CandidateRefused& e\) \{(.*?)\}\s*catch \(const std::exception& e\)", loop, re.S)
if not refused or "continue;" not in refused.group(1):
    raise SystemExit("a refused candidate does not continue the search before the engine-error catch")
if "tried.insert(candidate_spec);" not in loop[:loop.index("try {")]:
    raise SystemExit("a refused candidate is not recorded as tried")

print("engine_autotune_isolation_test: OK")
