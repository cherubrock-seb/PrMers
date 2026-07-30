#!/usr/bin/env python3
"""Guard the full Gaussian-pair backend policy and TF isolation."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
app = (root / "src/core/App.cpp").read_text(encoding="utf-8")
prime = (root / "src/modes/RunGaussianMersenne.cpp").read_text(encoding="utf-8")
factor = (root / "src/modes/RunGaussianMersenneFactor.cpp").read_text(encoding="utf-8")
tf = (root / "src/modes/RunGaussianTrialFactor.cpp").read_text(encoding="utf-8")

# The Gaussian modes must keep using the same centralized workload policy as
# their ordinary Mersenne counterparts.  This is what drives Aevum auto-plan
# selection and the explicit Aevum/Marin overrides.
assert 'o.mode == "prp" || o.mode == "gm-proth" || o.mode == "gm-prp"' in app
assert 'workload = engine::gpu_workload::prp' in app
assert 'o.mode == "gm-pm1" || o.mode == "gm-chain"' in app
assert 'workload = engine::gpu_workload::pm1' in app
assert 'o.mode == "gm-ecm"' in app
assert 'workload = engine::gpu_workload::ecm' in app
for selector in ('throughput:prp', 'throughput:pm1', 'throughput:ecm'):
    assert selector in app

# PRP/Proth, P-1 and ECM all instantiate the common engine abstraction.
assert 'engine::create_gpu' in prime
assert factor.count('engine::create_gpu') >= 2
assert 'selected Aevum backend' in factor or 'Aevum' in factor
assert 'selected Marin backend' in factor or 'Marin' in factor

# Trial factoring is intentionally a separate direct 64-bit OpenCL sieve.  It
# must not accidentally inherit an FFT engine or Aevum/Marin plan.
assert 'engine::create_gpu' not in tf
assert 'clCreateProgramWithSource' in tf
assert 'clCreateKernel(program, "gm_trial_factor"' in tf
assert 'OpenCL-GPU-TF' in tf


# GMCHAIN reconfigures each phase and completes one family before starting the other.
assert 'run_family_pipeline("GM")' in app
assert 'run_family_pipeline("GQ")' in app
assert 'configure_gaussian_phase_backend(options, engine::gpu_workload::pm1' in app
assert 'configure_gaussian_phase_backend(options, engine::gpu_workload::ecm' in app
assert 'configure_gaussian_phase_backend(options, engine::gpu_workload::prp' in app
assert 'aevum_fft_spec_explicit' in (root / "include/io/CliParser.hpp").read_text(encoding="utf-8")

print("Gaussian pair backend policy and direct-TF isolation audit passed")
