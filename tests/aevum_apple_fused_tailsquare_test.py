#!/usr/bin/env python3
"""Structural equivalence checks for the Apple fused GF61 tailSquare path."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAIL = (ROOT / "third_party/aevum/src/cl/tailsquare.cl").read_text()
GPU = (ROOT / "third_party/aevum/src/Gpu.cpp").read_text()
GPU_H = (ROOT / "third_party/aevum/src/Gpu.h").read_text()


def gather_source(out_i: int, out_me: int, f: int, radix: int, g_h: int) -> tuple[int, int]:
    logical = out_i * g_h + out_me
    remainder = logical & (f - 1)
    src_i = (logical // f) % radix
    src_me = (logical // (f * radix)) * f + remainder
    return src_i, src_me


def scatter_dest(src_i: int, src_me: int, f: int, radix: int, g_h: int) -> tuple[int, int]:
    remainder = src_me & (f - 1)
    logical = ((src_me // f) * radix + src_i) * f + remainder
    return logical // g_h, logical % g_h


def check_shuffle_bijection() -> None:
    for radix in (4, 8):
        g_h = 64
        for f in (1, 2, 4, 8, 16):
            if f > g_h:
                continue
            seen: set[tuple[int, int]] = set()
            for src_i in range(radix):
                for src_me in range(g_h):
                    out = scatter_dest(src_i, src_me, f, radix, g_h)
                    assert out not in seen, (radix, f, out)
                    seen.add(out)
                    assert gather_source(*out, f, radix, g_h) == (src_i, src_me)
            assert len(seen) == radix * g_h


def normal_lines(h: int) -> list[int]:
    return [line for line in range(1, h) if line != h // 2]


def reverse_coord(h: int, nh: int, g_h: int,
                  coord: tuple[int, int, int]) -> tuple[int, int, int]:
    line, slot, me = coord
    if slot < nh // 2:
        return coord
    j = slot - nh // 2
    return h - line, nh // 2 + (nh // 2 - 1 - j), g_h - 1 - me


def fake_pair(a: int, b: int, line: int, pair_slot: int) -> tuple[int, int]:
    # Non-symmetric deterministic stand-in for onePairSq.  Mapping equivalence
    # must hold independently of the actual GF61 arithmetic.
    salt = 17 * line + 5 * pair_slot
    return a + 2 * b + salt, 3 * a - b - salt


def check_reverse_pair_reverse() -> None:
    for h, nh, g_h in ((16, 4, 8), (32, 8, 16)):
        coords = [(line, slot, me) for line in normal_lines(h)
                  for slot in range(nh) for me in range(g_h)]
        a = {coord: 100000 * coord[0] + 1000 * coord[1] + coord[2] for coord in coords}

        # Legacy B=R(A), C=pair(B), D=R(C).
        b = {reverse_coord(h, nh, g_h, coord): value for coord, value in a.items()}
        c = dict(b)
        pair_slots = nh // 2
        quarter = nh // 4
        for line in normal_lines(h):
            for pair_slot in range(pair_slots):
                i = pair_slot % quarter
                typ = pair_slot // quarter
                a_slot = i + typ * quarter
                b_slot = a_slot + nh // 2
                for me in range(g_h):
                    ca = (line, a_slot, me)
                    cb = (line, b_slot, me)
                    c[ca], c[cb] = fake_pair(b[ca], b[cb], line, pair_slot)
        d = {reverse_coord(h, nh, g_h, coord): value for coord, value in c.items()}

        # Fused direct mapping used by tailSquareGF61PairCrossFusedApple.
        fused: dict[tuple[int, int, int], int] = {}
        for line in normal_lines(h):
            for pair_slot in range(pair_slots):
                i = pair_slot % quarter
                typ = pair_slot // quarter
                a_slot = i + typ * quarter
                cross = (h - line, nh - 1 - a_slot, None)
                for me in range(g_h):
                    ca = (line, a_slot, me)
                    cb = (cross[0], cross[1], g_h - 1 - me)
                    assert ca not in fused and cb not in fused
                    fused[ca], fused[cb] = fake_pair(a[ca], a[cb], line, pair_slot)
        assert fused == d


def check_source_wiring() -> None:
    for token in (
        "tailSquareGF61LoadStageFusedApple",
        "tailSquareGF61StageFusedApple",
        "tailSquareGF61PairCrossFusedApple",
    ):
        assert token in TAIL, token
        assert token in GPU or token in GPU_H, token
    assert "appleTailGF61ScatterFused" in TAIL
    assert "AEVUM_APPLE_TAILSQUARE_LEGACY" in GPU
    assert "ktailSquareGF61LoadStageFusedApple.startLoad(&compiler)" in GPU
    assert "using validated legacy staging" in GPU
    assert "apple_fused_tailsquare_gf61 && groupSize > 1" in GPU
    assert "for (u32 stage = nH; stage < groupSize; stage *= nH)" in GPU
    assert "for (u32 stage = 1; stage < groupSize; stage *= nH)" in GPU

    # M51 has nH=4 and SMALL_H=256: stages f={1,4,16}.
    stages = 3
    legacy_dispatches = 1 + 2 * (3 * stages + 1) + 3
    fused_dispatches = (stages + 1) + 1 + (stages + 1)
    assert legacy_dispatches == 24
    assert fused_dispatches == 9


if __name__ == "__main__":
    check_shuffle_bijection()
    check_reverse_pair_reverse()
    check_source_wiring()
    print("Apple fused GF61 tailSquare structural equivalence test passed")
    print("M51 normal-tail dispatch model: 24 legacy -> 9 fused")
