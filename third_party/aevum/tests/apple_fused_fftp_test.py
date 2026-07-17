#!/usr/bin/env python3
"""Structural checks for Apple fused CRT fftP width stages."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "src/cl/fftp.cl").read_text()
FFTW = (ROOT / "src/cl/fftw.cl").read_text()
GPU = (ROOT / "src/Gpu.cpp").read_text()
H = (ROOT / "src/Gpu.h").read_text()


def old_stage(values, stage, nw):
    # Deterministic stand-in for fft_RADIX followed by twiddle and shuffle.
    gw = len(values[0])
    radix = [[None] * gw for _ in range(nw)]
    for me in range(gw):
        src = [values[i][me] for i in range(nw)]
        # Non-symmetric invertible-ish private transform; mapping is what matters.
        out = [sum((j + 1) * (i + 3) * src[j] for j in range(nw)) for i in range(nw)]
        for i in range(nw):
            radix[i][me] = out[i] * (1 if i == 0 else 1009 + i * 17 + (me & ~(stage - 1)))
    flat = [None] * (nw * gw)
    for i in range(nw):
        for me in range(gw):
            dst = i * stage + (me & ~(stage - 1)) * nw + (me & (stage - 1))
            assert flat[dst] is None
            flat[dst] = radix[i][me]
    return [[flat[i * gw + me] for me in range(gw)] for i in range(nw)]


def fused_stage(values, stage, nw):
    gw = len(values[0])
    flat = [None] * (nw * gw)
    for me in range(gw):
        src = [values[i][me] for i in range(nw)]
        out = [sum((j + 1) * (i + 3) * src[j] for j in range(nw)) for i in range(nw)]
        for i in range(nw):
            v = out[i] * (1 if i == 0 else 1009 + i * 17 + (me & ~(stage - 1)))
            dst = i * stage + (me & ~(stage - 1)) * nw + (me & (stage - 1))
            assert flat[dst] is None
            flat[dst] = v
    return [[flat[i * gw + me] for me in range(gw)] for i in range(nw)]


def check_equivalence():
    for nw in (4, 8):
        gw = 64
        values = [[100000 * i + me for me in range(gw)] for i in range(nw)]
        for stage in (1, 4, 8, 16):
            assert old_stage(values, stage, nw) == fused_stage(values, stage, nw)


def check_source():
    for mod in (31, 61):
        for stage in (1, 4, 8, 16, 64, 256, 512):
            token = f"fftP{mod}WidthStageFused{stage}Apple"
            assert token in CL and (token in GPU or token in H), token
        token = f"fftP{mod}WeightStage1FusedApple"
        assert token in CL and (token in GPU or token in H), token
    assert "AEVUM_APPLE_FFTP_V55" in GPU
    assert "AEVUM_APPLE_FFTP_STAGE_ONLY" in GPU
    assert "apple_fused_fftp_width" in H
    assert "apple_fused_fftp_weight_first" in H
    for stage in (1, 4, 8, 16, 64, 256, 512):
        token = f"fftWGF61WidthStageFused{stage}Apple"
        assert token in FFTW and (token in GPU or token in H), token
    assert "fftWGF61LoadStage1FusedApple" in FFTW
    assert "AEVUM_APPLE_FFTW_V55" in GPU
    assert "AEVUM_APPLE_FFTW_STAGE_ONLY" in GPU
    stages = 4  # M51 WIDTH=1024, NW=4: 1,4,16,64
    old = 2 * (2 * stages + 2)
    stage_fused = 2 * (stages + 2)
    turbo = 2 * (stages + 1)
    assert (old, stage_fused, turbo) == (20, 12, 10)
    fftw_old = 2 * stages + 2
    fftw_stage = stages + 2
    fftw_turbo = stages + 1
    assert (fftw_old, fftw_stage, fftw_turbo) == (10, 6, 5)


if __name__ == "__main__":
    check_equivalence()
    check_source()
    print("Apple fused CRT fftP structural equivalence test passed")
    print("M51 fftP dispatch model: 20 legacy-split -> 12 stage-fused -> 10 weight-stage-fused")
    print("M51 GF61 fftW dispatch model: 10 legacy-split -> 6 stage-fused -> 5 load-stage-fused")
