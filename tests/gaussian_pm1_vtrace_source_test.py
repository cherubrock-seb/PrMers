#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
new = (root / "src/modes/RunGaussianMersennePm1VTrace.cpp").read_text()
old = (root / "src/modes/RunGaussianMersenneFactor.cpp").read_text()
pm1 = (root / "src/modes/RunPM1.cpp").read_text()
auto = (root / "src/aevum/AutoPolicy.cpp").read_text()
mk = (root / "Makefile").read_text()
app = (root / "include/core/App.hpp").read_text()

needles = [
    "int App::runGaussianMersennePM1()",
    "runGaussianMersennePM1Legacy()",
    "V_{n+1}=V_1 V_n - V_{n-1}",
    "engine::create_gpu(t.lift, reg_count",
    "const bool aevum = eng->is_aevum_backend();",
    "PRMERS_GM_PM1_PRODUCT_STAGE2",
    "options.pm1_vtrace_off",
    "options.gm_safe_replay",
    "legacy_checkpoint_present",
    "VRegs::MUL_VD",
    "PRMERS_GM_VTRACE_GCD_TERMS",
]
for n in needles:
    assert n in new, n

# Legacy implementation remains physically untouched and separately callable.
assert "const mpz_class qprod = product_range" in old
assert "int runGaussianMersennePM1Legacy();" in app
assert "GmPm1LegacyRename.hpp" in mk

# Ordinary Mersenne V-trace already has an Aevum-aware prepared-cache contract.
assert "const bool aevum_vtrace_backend = eng->is_aevum_backend();" in pm1
assert "eng->set_multiplicand((engine::Reg)RMUL_VD, (engine::Reg)RVD);" in pm1
assert "eng->set_multiplicand((engine::Reg)RMUL_V1, (engine::Reg)RV1);" in pm1

print("v100.13 GM V-trace/Aevum source contract: PASS")

# RC policy: P-1 (including Stage 2) and ECM are admitted up to equal transform
# size like PRP/LL; hardware A/B gates decide whether this is promoted/tagged.
assert 'return {1.00, "P-1 Stage 1", "AEVUM_AUTO_PM1_STAGE1_MAX_RATIO"};' in auto
assert 'return {1.00, "P-1 low-memory (3-register)", "AEVUM_AUTO_PM1_LOWMEM_MAX_RATIO"};' in auto
assert 'return {1.00, "ECM mixed-operation", "AEVUM_AUTO_ECM_MAX_RATIO"};' in auto
