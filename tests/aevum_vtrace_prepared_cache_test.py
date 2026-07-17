#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
s = (root / "src/modes/RunPM1.cpp").read_text()
assert "const bool aevum_vtrace_backend = eng->is_aevum_backend();" in s

start = s.index("    auto precompute_baby_window =")
end = s.index("\n    if (!resumed_s2)", start)
body = s[start:end]

prepare_v1 = "eng->set_multiplicand((engine::Reg)RMUL_V1, (engine::Reg)RV1);"
prepare_vd = "eng->set_multiplicand((engine::Reg)RMUL_VD, (engine::Reg)RVD);"
recurrence_mul = "eng->mul((engine::Reg)RBNEXT, (engine::Reg)RMUL_V1);"

assert body.count("if (aevum_vtrace_backend)") >= 2
assert prepare_v1 in body and prepare_vd in body and recurrence_mul in body
assert body.index(prepare_v1) < body.index(recurrence_mul) < body.rindex(prepare_vd)

# Strict isolation: both refreshes inside the window are guarded by the Aevum
# backend predicate. Marin retains the original outer preparation schedule.
for needle in (prepare_v1, prepare_vd):
    pos = body.index(needle)
    guard = body.rfind("if (aevum_vtrace_backend)", 0, pos)
    close = body.find("}", guard)
    assert guard >= 0 and pos < close

outer = s[end:s.index("    uint64_t idx = 0;", end)]
assert outer.count("if (!aevum_vtrace_backend)") >= 3

# Model the two-slot LRU sequence that failed in v99.58.
slots = {}
clock = 0
def touch(name):
    global clock
    clock += 1
    if name in slots:
        slots[name] = clock
        return
    if len(slots) == 2:
        victim = min(slots, key=slots.get)
        del slots[victim]
    slots[name] = clock

touch("VD")
touch("TMP")
touch("V1")
for _ in range(8):
    touch("V1")
touch("VD")
touch("TMP")
assert set(slots) == {"VD", "TMP"}
print("Aevum-only V-trace prepared-cache liveness test passed")
