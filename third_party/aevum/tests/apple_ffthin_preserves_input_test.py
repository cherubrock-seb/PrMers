#!/usr/bin/env python3
"""The Apple staged fftHinGF61 must leave its input intact, as the stock kernel does.

Gpu::exponentiate records fftMidIn(buf1); fftHin(buf2, buf1) and then squares
buf1 (tailSquare(buf1)).  The staged Apple fftHin ping-pongs between `out` and
an alternate bank.  This models the GF61-plane buffer flow of that replay
(Apple forces INPLACE=0) with the alternate bank the source actually uses.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gpu = (ROOT / "src/Gpu.cpp").read_text()

start = gpu.index("if (kern == KFFTHIN) {")
block = gpu[start:gpu.index("if (kern == KTAILSQUARE) {", start)]
apple = block[block.index("#if defined(__APPLE__)"):block.index("#else")]
m = re.search(r"Buffer<double>\* next = current == out \? (&buf3|in|out) : out;", apple)
assert m, "Apple staged fftHinGF61: ping-pong bank selection not found"

exp_start = gpu.index("void Gpu::exponentiate(")
exponentiate = gpu[exp_start:gpu.index("\n}\n", exp_start)]
order = [exponentiate.index(s) for s in (
    "fftMidIn(buf1);", "fftHin(buf2, buf1);", "tailSquare(buf1);", "fftMidOut(buf1);")]
assert order == sorted(order), "exponentiate no longer squares buf1 right after fftHin(buf2, buf1)"


def exponentiate_flow(alternate, stages):
    """Return the reads that see a clobbered value; [] when the flow is sound."""
    buf = {}
    bad = []

    def read(name, want, who):
        if buf.get(name) != want:
            bad.append(f"{who} reads {name}={buf.get(name)}, wants {want}")

    # fftP(buf1, bufInOut) runs at once on Apple: output buf3, buf1 as scratch.
    buf["buf3"], buf["buf1"] = "P", "scratch"
    # Replay, GF61 group, in recording order.
    read("buf3", "P", "fftMidIn")
    buf["buf1"], buf["buf3"] = "MI", "factor"          # buf3 becomes the factor buffer
    read("buf1", "MI", "fftHin load")
    buf["buf2"] = "H0"
    current = "buf2"
    for _ in range(stages):
        nxt = alternate if current == "buf2" else "buf2"
        buf[nxt] = "partial"
        current = nxt
    buf["buf2"] = "H"
    read("buf1", "MI", "tailSquare")                   # the square of the base
    buf["buf3"], buf["buf1"] = "T", "scratch"
    read("buf3", "T", "fftMidOut")
    return bad


bank = {"&buf3": "buf3", "in": "buf1", "out": "buf2"}[m.group(1)]
for small_h, nh in ((256, 4), (512, 8), (1024, 4)):
    stages = 0
    stage = 1
    while stage < small_h // nh:
        stages += 1
        stage *= nh
    # The model must catch the old aliasing, else it proves nothing.
    assert exponentiate_flow("buf1", stages), "model does not detect an input-aliasing fftHin"
    bad = exponentiate_flow(bank, stages)
    assert not bad, f"SMALL_H={small_h} NH={nh}: " + "; ".join(bad)

assert "if (out == &buf3 || in == &buf3)" in apple, "Apple staged fftHinGF61 must refuse buf3 as out/in"

print("Aevum Apple staged fftHinGF61 input-preservation test passed")
