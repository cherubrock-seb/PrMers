#!/usr/bin/env python3
"""GPU-free source-flow regression for P-1 two-stage reporting.

This checks both extension and normal Stage-1 branches, together with
Stage-2 result-write failures and the user-visible progress option.
It is intentionally not a substitute for end-to-end Windows GPU tests.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pm1 = (root / "src/modes/RunPM1.cpp").read_text()
cli = (root / "src/io/CliParser.cpp").read_text()
spinner = (root / "src/core/Spinner.cpp").read_text()
opts = (root / "include/io/CliParser.hpp").read_text()

assert pm1.count("const bool runRequestedStage2 = core::pm1::stage2_required(") == 2
assert pm1.count("deferring result submission until B2=") == 2
assert "if (!runRequestedStage2 && !options.pm1_no_stage1_gcd)" in pm1
assert 'if (!runRequestedStage2) {' in pm1
for suffix in ("_stage2_vtrace", "_stage2_resume2reg", "_stage2_ext"):
    assert 'std::string(options.mode) + "' + suffix + '"' in pm1
assert "external_stage2_report_failure" in pm1
assert pm1.count("return Pm1Stage2Result::Error;") > 10
assert pm1.count("core::pm1::progress_due(") >= 10
assert pm1.count("core::pm1::progress_prefix(options.exponent, 2)") >= 6
assert "core::pm1::progress_prefix(expo, 1)" in spinner
assert "-pm1-progress-seconds" in cli
assert "pm1_progress_seconds = 300" in opts
assert "-pm1-no-progress-seconds" in cli
assert "pm1_no_progress_seconds = 0" in opts
assert "NoProgressWatchdog" in pm1
assert "[PM1-WATCHDOG]" in pm1
assert "std::_Exit(124)" in pm1
assert "P-1 and ECM Stage 2 handoff" in cli
print("PM1_REPORTING_SOURCE_GATE=PASS")
