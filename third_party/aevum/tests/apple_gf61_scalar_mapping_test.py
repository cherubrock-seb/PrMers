#!/usr/bin/env python3
"""Verify scalar Apple GF61 indexing/weights match the grouped fftP loop."""

MASK64 = (1 << 64) - 1


def make_u64(hi: int, lo: int) -> int:
    return ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)


def check_case(
    exp: int,
    width: int,
    big_height: int,
    nw: int,
    frac_hi: int,
    exhaustive: bool = False,
) -> None:
    nwords = width * big_height * 2
    gw = width // nw
    combo_step_value = frac_hi + 1

    def combo_frac(i: int) -> int:
        # All production plans covered here use a power-of-two word count.
        return (i * combo_step_value - 1) & MASK64

    log2_root_two = ((1 << 60) // nwords) % 61
    big_shift = (nwords - exp % nwords) * log2_root_two % 61
    shift_minus_one = (big_shift + 60) % 61
    combo_step = make_u64(shift_minus_one, frac_hi)
    combo_bigstep = (
        combo_frac(gw * big_height * 2 - 1)
        + make_u64((gw * big_height * 2 - 1) * shift_minus_one, 0)
    ) % (61 << 32)

    def grouped(line: int, me: int, lane: int) -> tuple[int, int]:
        word_index = (me * big_height + line) * 2
        combo = (
            combo_frac(word_index)
            + make_u64(word_index * shift_minus_one, 0xFFFFFFFF)
        ) & MASK64
        high = (combo >> 32) % 61
        for i in range(lane + 1):
            shift0 = high
            combo = (combo + combo_step) & MASK64
            high = (combo >> 32) % 61
            shift1 = high
            if i == lane:
                return shift0, shift1
            combo = (combo + combo_bigstep) & MASK64
            high = (combo >> 32) % 61
        raise AssertionError("unreachable")

    def scalar(p: int) -> tuple[int, int]:
        line = p // width
        x = p - line * width
        word_index = (line + big_height * x) * 2
        combo = (
            combo_frac(word_index)
            + make_u64(word_index * shift_minus_one, 0xFFFFFFFF)
        ) & MASK64
        shift0 = (combo >> 32) % 61
        combo = (combo + combo_step) & MASK64
        shift1 = (combo >> 32) % 61
        return shift0, shift1

    if exhaustive:
        points = (
            (line, me, lane)
            for line in range(big_height)
            for me in range(gw)
            for lane in range(nw)
        )
    else:
        lines = sorted({0, 1, big_height // 2, big_height - 2, big_height - 1})
        mes = sorted({0, 1, gw // 2, gw - 2, gw - 1})
        lanes = range(nw)
        points = ((line, me, lane) for line in lines for me in mes for lane in lanes)

    for line, me, lane in points:
        p = line * width + lane * gw + me
        assert grouped(line, me, lane) == scalar(p), (
            exp,
            width,
            big_height,
            nw,
            p,
            grouped(line, me, lane),
            scalar(p),
        )


# Exact M2 smoke plan and two nearby power-of-two layouts.
check_case(1_362_763, 256, 512, 4, 852_672_511, exhaustive=True)
check_case(136_279_841, 1024, 4096, 8, 1_251_848_447)
check_case(2_147_483_647, 4096, 8192, 16, 4_294_967_295)
print("Aevum Apple scalar GF61 mapping test passed")
