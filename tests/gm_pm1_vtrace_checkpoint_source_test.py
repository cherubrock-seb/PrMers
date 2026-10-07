#!/usr/bin/env python3
"""The default GM/GQ P-1 V-trace driver must checkpoint Stage 1 in the legacy
`_stage1.ckpt` format, so that its legacy-checkpoint detection resumes it."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
vt = (root / 'src/modes/RunGaussianMersennePm1VTrace.cpp').read_text()
legacy = (root / 'src/modes/RunGaussianMersenneFactor.cpp').read_text()
hdr = (root / 'include/core/GmFactorCheckpoint.hpp').read_text()

# One implementation of the checkpoint format, shared by both drivers.
assert 'struct FactorCheckpointHeader' in hdr
assert 'struct FactorCheckpointHeader' not in legacy
assert '#include "core/GmFactorCheckpoint.hpp"' in legacy
assert '#include "core/GmFactorCheckpoint.hpp"' in vt
assert 'PM1_WINDOW_REGS' in hdr

# The engine register count (part of the checkpoint size check) is the legacy one.
assert 'static constexpr std::size_t count = core::gm_factor_ckpt::PM1_WINDOW_REGS;' in legacy
assert 'std::size_t s1_regs = gfc::PM1_WINDOW_REGS;' in vt

# Stage 1 saves periodically and on interrupt, with the bits still remaining.
assert 'gfc::save_factor_checkpoint(legacy_s1, s1.get(), 1, 1, t, B1, B2, e_bits,' in vt
assert 'save_s1(i - 1);' in vt           # periodic
assert 'save_s1(i);' in vt               # interrupt
assert 'No compact V-trace checkpoint was written' not in vt

# The finished Stage 1 residue is kept as the legacy Stage 2 checkpoint, which
# the terminal paths of the V-trace driver remove again.
assert 'gfc::save_factor_checkpoint(legacy_s2, s1.get(), 1, 2, t, B1, B2, primes.size(),' in vt
assert vt.count('gfc::clear_checkpoint(legacy_s2);') >= 4
assert 'gfc::clear_checkpoint(legacy_s1);' in vt

# The legacy-checkpoint detection that resumes these files is still there.
assert 'legacy checkpoint detected' in vt
print('GM P-1 V-trace checkpoint source test passed')
