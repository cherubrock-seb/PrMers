#!/usr/bin/env python3
"""PRP_MIDDLE1 must not be enabled for a plan with an FP64 plane.

The fused MIDDLE=1 read/write in prp_middle1.cl exists only for FP32, GF31 and GF61 data.  tailSquare for an FP64 plane built with
AEVUM_PRP_MIDDLE1 calls prpReadMiddle1/prpWriteMiddle1 on T2 and does not compile, so Gpu::Gpu must leave prpMiddle1 off for any plan with
fft.FFT_FP64 rather than record the kernel that cannot be built (reachable through AEVUM_PRP_MIDDLE1=1 on an explicit MIDDLE=1 FP64 plan).
"""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parent.parent / "src"
gpu = (root / "Gpu.cpp").read_text()
cl = (root / "cl" / "prp_middle1.cl").read_text()

m = re.search(r"prpMiddle1 = args\.value\(\"PRP_MIDDLE1\", 0\)(.*?);", gpu, re.S)
fail = []
if not m:
    fail.append("Gpu.cpp: prpMiddle1 assignment not found")
else:
    has_fp64_version = re.search(r"#if\s+FFT_FP64", cl) is not None or re.search(r"\bT2\b", cl) is not None
    if not has_fp64_version and "!fft.FFT_FP64" not in m.group(1):
        fail.append("Gpu.cpp: prpMiddle1 can be enabled for an FP64 plan but prp_middle1.cl has no FP64 version")
for f in fail:
    print("FAIL:", f)
if fail:
    sys.exit(1)
print("ok: PRP_MIDDLE1 is not enabled for FP64 plans")
