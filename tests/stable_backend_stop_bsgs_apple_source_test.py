#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cli_h = (ROOT / "include/io/CliParser.hpp").read_text()
cli = (ROOT / "src/io/CliParser.cpp").read_text()
pm1 = (ROOT / "src/modes/RunPM1.cpp").read_text()
ecm = (ROOT / "src/modes/RunEcm.cpp").read_text()
te = (ROOT / "src/modes/RunEcmTwistedEdwards.cpp").read_text()
auto_policy = (ROOT / "src/aevum/AutoPolicy.cpp").read_text()
engine_aevum = (ROOT / "src/aevum/EngineAevum.cpp").read_text()
app = (ROOT / "src/core/App.cpp").read_text()
gpu = (ROOT / "src/marin/gpu.cpp").read_text()
fftconfig = (ROOT / "third_party/aevum/src/FFTConfig.cpp").read_text()
engine_api_test = (ROOT / "third_party/aevum/tests/engine_api_load_test.cpp").read_text()
audit_script = (ROOT / "scripts/audit_aevum_plans.py").read_text()
workload_audit = (ROOT / "third_party/aevum/tests/workload_plan_audit.cpp").read_text()

# Stop/continue CLI policies.
assert "pm1_continue_stage2_after_factor = false" in cli_h
assert "ecm_continue_after_factor = false" in cli_h
assert "-pm1-continue-stage2-after-factor" in cli
assert "-ecm-continue-after-factor" in cli
assert "New Stage 1 factor found; Stage 2 skipped by default" in pm1
assert "newStage1FactorFound" in pm1
assert "New factor found; stopping ECM by default" in ecm
assert "New factor found; stopping ECM by default" in te
assert "factor_rc == 2 && !options.ecm_continue_after_factor" in ecm
assert "factor_rc == 2 && !options.ecm_continue_after_factor" in te

# A fixed seed remains one curve by default, but continuation + K>1 creates a
# reproducible series so the override can really run later curves.
assert "forcedSeedSeries" in ecm
assert "forcedSeedSeries" in te
assert "forcedCurveSeedValue" in ecm
assert "forcedCurveSeedValue" in te

# Classic BSGS correctness: Apple avoids segmented dense slabs, and all
# platforms use a canonical giant multiplied by the prepared baby step.  The
# historical backend-specific partial-forward mul_new route must stay absent.
assert "Apple correctness guard: segmented classic BSGS is disabled" in pm1
assert 'classicMem.vendor.find("Apple")' in pm1
assert "[PM1-CLASSIC] canonical giant/baby multiplication enabled" in pm1
assert "eng->copy((engine::Reg)RTMP, (engine::Reg)RGIANT)" in pm1
assert "eng->mul((engine::Reg)RTMP, (engine::Reg)babyReg)" in pm1
assert "eng->mul_new((engine::Reg)RTMP" not in pm1

# Every Stage 2 implementation emits one stable machine-readable factor line
# in addition to its historical human-readable message.
assert pm1.count("P-1 factor stage 2 found:") >= 8

# Workload-specific plan audit is shipped and requires exact outputs before
# it can recommend a faster plan.
assert "word differential mismatch" in audit_script
assert "canonical mathematical state differs from Type1 baseline" in audit_script
assert "AEVUM WORKLOAD PLAN DIFFERENTIAL PASSED" in workload_audit
assert "workload must be prp, ll, pm1, or ecm" in workload_audit
assert "parse_resume_record" in audit_script
assert "canonical_state" in audit_script
assert "recommended-workload-plans.env" in audit_script
assert "--strict-policy" in audit_script

# Backend reporting and Apple safety policy.
assert '<< ", family=" << plan_family(result.fft_spec)' in auto_policy
assert "Type4 FFT323161" in auto_policy
assert "requested-plan=" in engine_aevum
assert "PRP/LL uses staged stock Type1 FFT3161" in app
assert 'o.aevum_fft_spec = "pow2:auto"' in app
assert '"throughput:prp"' in app
assert '"throughput:ll"' in app
assert '"throughput:pm1"' in app
assert '"throughput:ecm"' in app
assert "PRMERS_AEVUM_PM1_FFT" in app
assert 'spec == "throughput:pm1"' in fftconfig
assert 'spec == "throughput:ecm"' in fftconfig
assert "Apple OpenCL 1.2 supports only stock FFT3161" in fftconfig
assert "validated only for PRP/LL stock FFT3161" in gpu
assert "mixed/prepared multiplication failed invariant/Gerbicz validation" in gpu
assert "#if defined(__APPLE__)" in engine_api_test
assert "Aevum Apple type4 rejection" in engine_api_test

print("PrMers v99.86 workload plan policy/source test passed")
