#!/usr/bin/env python3
"""Source test: the fused add/sub carry kernels of kernels/marin.cl are gone.

carry_weight_addsub_p2, carry_weight_addsub_p2_copy and carry_weight_sub_p2 dropped a carry that survives a
whole digit group, and nothing launched them: engine_gpu_flat::addsub / addsub_copy / sub_reg go through the exact
group subtraction. The kernels and their host wrappers were removed so that nobody re-enables an inexact
subtraction by accident. This test keeps them out, and checks that include/marin/ocl/kernel.h (the embedded copy
of kernels/marin.cl) is still in sync with the kernel file.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CL = (ROOT / "kernels" / "marin.cl").read_text()
HDR = (ROOT / "include" / "marin" / "ocl" / "kernel.h").read_text()
ENG = (ROOT / "include" / "marin" / "engine_gpu.h").read_text()

DEAD = [
    "carry_weight_addsub_p1", "carry_weight_addsub_p2",
    "carry_weight_addsub_p1_copy", "carry_weight_addsub_p2_copy",
    "carry_weight_sub_p2",
]

fails = []

for name in DEAD:
    pat = re.compile(r"(?<![A-Za-z0-9_])_?%s(?![A-Za-z0-9_])" % re.escape(name))
    for label, text in (("kernels/marin.cl", CL), ("include/marin/ocl/kernel.h", HDR), ("include/marin/engine_gpu.h", ENG)):
        if pat.search(text):
            fails.append("%s still mentions %s" % (label, name))

for fn in ("carry_weight_addsub", "void addsub_copy(const size_t sum"):
    if fn in ENG:
        fails.append("engine_gpu.h still defines %s" % fn)

# the exact-subtraction path that replaced them must still be wired
for needed in ("subtract_reg_group_exact", "carry_weight_sub_p2_phase"):
    if needed not in ENG:
        fails.append("engine_gpu.h lost %s" % needed)


def embed(src: str) -> str:
    lines = src.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out = ["/* This file is generated from kernels/marin.cl. */", "", "#pragma once", "",
           "static const char * const src_ocl_kernel = \\"]
    for l in lines:
        out.append('"' + l.replace("\\", "\\\\").replace('"', '\\"') + '\\n" \\')
    out[-1] = out[-1][:-2] + ";"
    return "\n".join(out) + "\n"


if embed(CL) != HDR:
    fails.append("include/marin/ocl/kernel.h is not the embedded form of kernels/marin.cl")

if fails:
    print("\n".join("FAIL: " + f for f in fails))
    sys.exit(1)
print("Marin dead addsub kernels source test: PASS")
