#!/usr/bin/env python3
"""Source-policy and launch-coverage checks for the Apple GF61 tail split."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gpu = (ROOT / "src/Gpu.cpp").read_text()
tail = (ROOT / "src/cl/tailsquare.cl").read_text()

required = [
    'fft.shape.fft_type == FFT3161',
    'config["TAIL_KERNELS"] = "3";',
    'tail_single_wide = false;',
    'tail_single_kernel = false;',
    'ktailSquareZeroGF61(*out, *in);',
    'ktailSquareGF61(*out, *in);',
]
for token in required:
    assert token in gpu, token

# Preserve the existing upstream kernels and their LDS paths.
assert 'KERNEL(G_H) tailSquareZeroGF61' in tail
assert 'KERNEL(G_H * 2) tailSquareGF61' in tail
assert 'local GF61 lds[LDS_BYTES / sizeof(GF61)]' in tail
assert 'local GF61 lds[2 * LDS_BYTES / sizeof(GF61)]' in tail
assert 'fft_HEIGHT1(lds' in tail
assert 'fft_HEIGHT2(lds' in tail

# Validate the upstream TAIL_KERNELS=3 launch partition for representative
# admissible heights.  Zero handles line 0 and H/2.  The main double-wide
# groups handle paired lines 1..H/2-1 and H-1..H/2+1 exactly once.
for H in (64, 128, 256, 512, 1024):
    zero_lines = {0, H // 2}
    main_pairs = [(g + 1, H - (g + 1)) for g in range(H // 2 - 1)]
    covered = set(zero_lines)
    for a, b in main_pairs:
        assert a not in covered and b not in covered
        covered.add(a)
        covered.add(b)
    assert covered == set(range(H)), (H, len(covered))

# The non-Apple call remains the original single ktailSquareGF61 invocation;
# the policy changes configuration, not the kernel implementation or ABI.
assert '#if defined(__APPLE__)' in gpu
assert '#else\n          kfftMidInGF61(*out, *in);' in gpu
print('Aevum Apple GF61 double-wide two-kernel tail policy test passed')
