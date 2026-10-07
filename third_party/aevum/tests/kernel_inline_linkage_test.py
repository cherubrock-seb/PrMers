#!/usr/bin/env python3
"""Helper functions in src/cl must be `static inline`, never plain `inline`.

OpenCL C follows C99 inline semantics: a plain `inline` function has no external definition, so when the compiler does not inline the call
(-cl-opt-disable, which Apple's driver is retried with, or any -O0 build) the kernel refers to an undefined symbol and the program fails to
build.  `static inline` always has a definition.

Part 1 scans src/cl for plain inline definitions.  Part 2 compiles a kernel that calls gf61limb.cl's aevumMul61 at -O0 with clang and checks the
helper is defined, not just declared (skipped when clang is missing).
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

CL_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "cl"

fail = []
for path in sorted(CL_DIR.glob("*.cl")):
    for n, line in enumerate(path.read_text().split("\n"), 1):
        if re.match(r"\s*inline\s", line):
            fail.append(f"{path.name}:{n}: plain inline helper; use static inline")

clang = shutil.which("clang")
if clang:
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "k.cl"
        src.write_text('#include "base.cl"\n#include "math.cl"\n#include "gf61limb.cl"\n'
                       "kernel void k(global ulong* o) { o[0] = aevumMul61(o[1], o[2]); }\n")
        defs = dict(FFT_TYPE=3, FFT_FP64=0, FFT_FP32=0, NTT_GF31=0, NTT_GF61=1, EXP="1257787u", WIDTH="256u", SMALL_HEIGHT="256u", MIDDLE="2u",
                    NW="4u", NH="4u", WordSize="8u", CARRY_LEN="8u", PFA_RADIX="0u", MAXBPW="3998u", FFT_VARIANT="101u", FRAC_BPW_HI="1u",
                    FRAC_BPW_LO="4294967295u", TAILTGF61="U2(1ul,1ul)", DISTGF61="0ul", NO_ASM=1)
        cmd = [clang, "-x", "cl", "-cl-std=CL2.0", "-O0", "-target", "spir64", "-emit-llvm", "-S", "-o", "-", str(src), "-I", str(CL_DIR)]
        cmd += [f"-D{k}={v}" for k, v in defs.items()]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            fail.append("clang -O0 failed:\n" + "\n".join(l for l in p.stderr.splitlines() if "error" in l)[:500])
        elif re.search(r"^declare .*@aevum", p.stdout, re.M):
            fail.append("at -O0 a gf61limb.cl helper is declared but not defined (plain inline)")
        elif not re.search(r"^define .*@aevumMul61", p.stdout, re.M):
            fail.append("at -O0 aevumMul61 is not defined in the output")

for f in fail:
    print("FAIL:", f)
if fail:
    sys.exit(1)
print("ok: all src/cl helpers are static inline" + ("" if clang else " (clang -O0 check skipped: clang not found)"))
