#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
engine = (root / "src/EngineApi.cpp").read_text()
gpu_h = (root / "src/Gpu.h").read_text()
gpu = (root / "src/Gpu.cpp").read_text()
fft = (root / "src/FFTConfig.cpp").read_text()

required = [
    "pending_reg_ = index;",
    "gpu_->regSquareStep(reg(index), pending_lead_width_, true);",
    "gpu_->regSquareStep(reg(index), lead_in, false);",
    "flush_pending_square();",
    "AEVUM_REG_LEAD_CACHE",
]
for text in required:
    if text not in engine:
        raise SystemExit(f"missing lead-cache engine transition: {text}")
if "bool regSupportsLeadCache() const;" not in gpu_h:
    raise SystemExit("Gpu lead-cache capability API missing")
for text in ["return !useLongCarry", "fft.pfa_radix == 9", "lead_in ? LEAD_WIDTH : LEAD_NONE", "lead_out ? LEAD_WIDTH : LEAD_NONE"]:
    if text not in gpu:
        raise SystemExit(f"Gpu lead-cache routing missing: {text}")
if "FFT323161" not in fft or "throughput:auto" not in fft:
    raise SystemExit("throughput auto FFT323161 selector is not enabled")

# Pure state-machine check: one logical square is always held pending, all
# earlier squares are emitted as WIDTH->WIDTH, and flush emits exactly one
# final ->NONE square.  This mirrors the C++ scheduling invariant.
def schedule(n):
    pending = False
    lead = False
    emitted = []
    for _ in range(n):
        if not pending:
            pending = True
            lead = False
        else:
            emitted.append((lead, True))
            lead = True
    if pending:
        emitted.append((lead, False))
    return emitted

for n in range(1, 100):
    seq = schedule(n)
    if len(seq) != n or seq[-1][1] or any(not out for _, out in seq[:-1]):
        raise SystemExit(f"invalid lead-cache schedule for {n}: {seq}")
    if seq[0][0] or any(not lead for lead, _ in seq[1:]):
        raise SystemExit(f"invalid lead input sequence for {n}: {seq}")

print("Aevum register lead-cache scheduling/source test passed")
