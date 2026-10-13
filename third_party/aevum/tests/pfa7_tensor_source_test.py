#!/usr/bin/env python3
"""Source-level proof and wiring guard for exact Tensor Core PFA7/GF31."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "src/cl/pfa7tensor.cl").read_text()
BASE = (ROOT / "src/cl/base.cl").read_text()
USE = (ROOT / "src/UseOptions.h").read_text()
MIN = (ROOT / "src/cl/fftmiddlein.cl").read_text()
MOUT = (ROOT / "src/cl/fftmiddleout.cl").read_text()

M31 = (1 << 31) - 1
RADIX = 7
BASE11 = 1 << 11


def array(name):
    match = re.search(
        rf"constant\s+i32\s+{name}\[21\]\s*=\s*\{{(.*?)\}};",
        SRC,
        re.S,
    )
    assert match, f"missing {name}"
    values = [int(v) for v in re.findall(r"-?\d+", match.group(1))]
    assert len(values) == 21, (name, len(values))
    assert all(-1024 <= value <= 1023 for value in values), name
    return [values[i : i + 3] for i in range(0, 21, 3)]


def reconstruct(digits):
    return sum(digit * (BASE11 ** limb) for limb, digit in enumerate(digits)) % M31


forward = array("PFA7_TC_FWD_DIGITS")
inverse = array("PFA7_TC_INV_NORM_DIGITS")

root = pow(5, (M31 - 1) // RADIX, M31)
inv_root = pow(root, M31 - 2, M31)
inv7 = pow(RADIX, M31 - 2, M31)

assert pow(root, RADIX, M31) == 1 and root != 1
assert [reconstruct(v) for v in forward] == [pow(root, e, M31) for e in range(RADIX)]
assert [reconstruct(v) for v in inverse] == [
    (inv7 * pow(inv_root, e, M31)) % M31 for e in range(RADIX)
]

# FP16 stores every signed 11-bit limb exactly.  Every possible seven-product
# partial sum is also an exact FP32 integer, independent of MMA accumulation order.
assert RADIX * 1024 * 1024 < (1 << 23)

assert "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" in SRC
assert "bar.warp.sync 0xffffffff" in SRC
assert "#define AEVUM_TC_PFA7_GF31 0" in BASE
assert "HAS_PTX < 750" in BASE
assert 'key == "AEVUM_TC_PFA7_GF31"' in USE
assert "pfaDft7TensorGF31(u, 0u, lds);" in MIN
assert "pfaDft7TensorGF31(u, 1u, lds);" in MOUT

print("ok: exact Tensor Core PFA7/GF31 tables, bounds, and opt-in wiring")
