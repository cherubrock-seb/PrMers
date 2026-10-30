#!/usr/bin/env python3
"""The plan search's time budget must not count the tuning context or the engine builds (kernel compiles)."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
engine = (root / "src/EngineApi.cpp").read_text()

loop_start = engine.index("for (size_t candidate_index = 0;")
loop = engine[loop_start:engine.index("tune.reset();", loop_start)]
head = engine[engine.rindex("const auto tune_start", 0, loop_start):loop_start]

if "TuneBudget<> budget(tune_start);" not in head:
    raise SystemExit("the plan search has no TuneBudget")
if re.search(r"now\(\)\s*-\s*tune_start", loop):
    raise SystemExit("the plan search budget still counts from tune_start, setup included")
if "const auto elapsed = budget.elapsedMs(std::chrono::steady_clock::now());" not in loop:
    raise SystemExit("the plan search does not check its budget through TuneBudget")
ctx = loop.index("tune.emplace(selected_device, shared_);")
if "budget.exclude(" not in loop[ctx:ctx + 200]:
    raise SystemExit("creating the tuning context is counted in the budget")
if "native_fft, *candidate_fft, &setup);" not in loop or "~ExcludeSetup() { budget.exclude(setup); }" not in loop:
    raise SystemExit("candidate engine builds are counted in the budget")

start = engine.index("PlanComparison comparePlans(")
compare = engine[start:engine.index("struct BridgeComparison", start)]
if "if (gpu) *setup += gpu->kernelLoadTime();" not in compare:
    raise SystemExit("the lazily compiled kernels are counted in the budget")
if "~SetupAccount()" not in compare or "} setup_account{setup, build_start, built, build, native_gpu, candidate_gpu};" not in compare:
    raise SystemExit("a refused or failed candidate's build is counted in the budget")
gpu_h = (root / "src/Gpu.h").read_text()
if "kernelLoadTime() const { return compiler.loadTime(); }" not in gpu_h:
    raise SystemExit("Gpu does not report its kernel load time")
compiler = (root / "src/KernelCompiler.cpp").read_text()
load = compiler[compiler.index("std::future<KernelHolder> KernelCompiler::load("):]
if "~AddTime() { total += (std::chrono::steady_clock::now() - start).count(); }" not in load:
    raise SystemExit("KernelCompiler::load does not add up its load time")

print("autotune_budget_source_test: OK")
