#!/usr/bin/env python3
"""Every GM ECM driver must resume after its completed curves, and the
legacy ECM / P-1 drivers must resume Stage 2 without redoing Stage 1."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
modes = root / 'src/modes'

for name in ('RunGaussianMersenneFactor.cpp',
             'RunGaussianMersenneEcmFast.cpp',
             'RunGaussianMersenneEcmOptimized.cpp'):
    s = (modes / name).read_text()
    assert '#include "core/GmEcmProgress.hpp"' in s, name
    assert 'core::gm_ecm_progress::Guard progress_guard(' in s, name
    assert 'for (std::uint64_t curve = first_curve; curve < curves; ++curve)' in s, name
    assert 'for (std::uint64_t curve = 0; curve < curves; ++curve)' not in s, name
    assert 'core::gm_ecm_progress::save(progress_file, progress_key, curve)' in s, name

legacy = (modes / 'RunGaussianMersenneFactor.cpp').read_text()
assert legacy.count('Stage 2 checkpoint found; skipping Stage 1') == 2   # ECM and P-1
assert 'const bool resumed_s2 = options.resume && !s2primes.empty() &&' in legacy

opt = (modes / 'RunGaussianMersenneEcmOptimized.cpp').read_text()
assert 'Stage 2 checkpoint found; skipping Stage 1' in opt
assert 'stage1_complete_from_checkpoint' in opt
assert 'resumed_s1 && remaining == 0' in opt
assert 'completed Stage1 checkpoint found; ' in opt
assert 'skipping fused Stage1.' in opt
assert 'if (!stage1_complete_from_checkpoint)' in opt
assert 'if (stage2_enabled) save_s1(0);' in opt
print('GM ECM resume source test passed')
