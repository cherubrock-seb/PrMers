#!/usr/bin/env python3
"""TAIL_KERNELS outside 0..3, or not an integer at all, must be rejected by the host; outside 0..3 also by the kernels.

The host (Gpu.cpp clDefines) only recognises 0..3 when choosing the tail launch geometry, while tailutil.cl derives
SINGLE_WIDE = (TAIL_KERNELS < 2) and SINGLE_KERNEL = ((TAIL_KERNELS & 1) == 0) from the same -DTAIL_KERNELS value.  For any other value
the two disagree, so host launch sizes and the kernels' expectations differ and the run computes garbage.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
gpu = (ROOT / "src" / "Gpu.cpp").read_text()
tail = (ROOT / "src" / "cl" / "tailutil.cl").read_text()

fail = []

m = re.search(r'if \(k == "TAIL_KERNELS"\) \{(.*?)\n    \}\n', gpu, re.S)
if not m:
    fail.append("Gpu.cpp: TAIL_KERNELS handling not found")
else:
    body = m.group(1)
    if not re.search(r'<\s*0\s*\|\|[^;{]*>\s*3', body):
        fail.append("Gpu.cpp: TAIL_KERNELS is not range-checked against 0..3")
    if "throw" not in body:
        fail.append("Gpu.cpp: an out-of-range TAIL_KERNELS does not throw")
    # The text must be parsed strictly: atoi() maps "garbage" to 0 and "3x" to 3, which would pass the range check while
    # the raw text is still forwarded to the kernels as -DTAIL_KERNELS=<text> (behaviour covered by strict_int_test.cpp).
    if "atoi(" in body:
        fail.append("Gpu.cpp: TAIL_KERNELS must not be parsed with atoi (accepts nonnumeric text)")
    if body.count("parseStrictInt(") != 1:
        fail.append("Gpu.cpp: TAIL_KERNELS must be parsed once with parseStrictInt and the result range-checked")
    elif not re.search(r'!\s*parseStrictInt\([^;{]*\)\s*\|\|', body):
        fail.append("Gpu.cpp: a TAIL_KERNELS value that does not parse as an integer is not rejected")

if not re.search(r'#if\s+TAIL_KERNELS\s*<\s*0\s*\|\|\s*TAIL_KERNELS\s*>\s*3\s*\n#error', tail):
    fail.append("tailutil.cl: no #error for TAIL_KERNELS outside 0..3")
if tail.find("#error TAIL_KERNELS") > tail.find("#define SINGLE_WIDE"):
    fail.append("tailutil.cl: the range check must come before SINGLE_WIDE is derived")

for f in fail:
    print("FAIL:", f)
if fail:
    sys.exit(1)
print("ok: TAIL_KERNELS is range-checked in Gpu.cpp and tailutil.cl")
