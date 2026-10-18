#!/usr/bin/env python3
"""carryFused's per-wavefront ready flag (the !OLD_FENCE hand-off) needs a warp-level sync() on both sides.

Writer: one lane per wavefront publishes the "carries are ready" flag on behalf of every lane, so every lane's CSSTORE must be complete first.
sync() must precede the flag store (a sync() after it, or none, lets lane 0 publish early on nVidia Volta and later, where the lanes of a warp
are scheduled independently).  Reader: lane 0 spins on the flag and the other lanes read the shuttle on its say-so, so the warp must reconverge
after the spin and before the loads.  gpuowl 3dd7c5d.

Checks every carryFused variant in src/cl/carryfused.cl, plus that sync() is a real warp barrier on nVidia sm_60+ and nothing elsewhere.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

CL_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "cl"
lines = (CL_DIR / "carryfused.cl").read_text().split("\n")
fail = []

stores = [i for i, l in enumerate(lines) if "AEVUM_READY_STORE(&ready[pos])" in l]
spins = [i for i, l in enumerate(lines) if "AEVUM_READY_LOAD(&ready[pos]) == 0" in l]
kernels = [i for i, l in enumerate(lines) if l.startswith("KERNEL(")]

if len(stores) != len(kernels):
    fail.append(f"{len(kernels)} carryFused kernels but {len(stores)} per-wavefront ready stores")
if len(spins) != len(kernels):
    fail.append(f"{len(kernels)} carryFused kernels but {len(spins)} per-wavefront ready spins")

for i in stores:
    window = [l.strip() for l in lines[max(0, i - 5):i]]
    if "sync();" not in " ".join(w.split("//")[0] for w in window).split():
        fail.append(f"carryfused.cl:{i + 1}: ready flag stored without a preceding sync()")
    # the fence stays ahead of the sync
    elif not any(w.startswith("write_mem_fence") for w in window):
        fail.append(f"carryfused.cl:{i + 1}: no write_mem_fence before the ready flag")

for i in spins:
    after = [l.strip() for l in lines[i + 1:i + 4]]
    code = [a for a in after if a and not a.startswith("//")]
    # do { spin(); } while(...);  }  sync();  mem_fence(...)
    if len(code) < 2 or code[0] != "}" or not code[1].startswith("sync();"):
        fail.append(f"carryfused.cl:{i + 1}: no sync() after the ready spin-wait, before the carries are read")
    elif len(code) < 3 or "mem_fence(CLK_GLOBAL_MEM_FENCE)" not in code[2]:
        fail.append(f"carryfused.cl:{i + 1}: the fence no longer follows the post-spin sync()")

# The OLD_FENCE path (whole-workgroup bar()) must stay free of sync().
text = "\n".join(lines)
for m in re.finditer(r"#if OLD_FENCE\n(.*?)#else", text, re.S):
    if re.search(r"\bsync\(\)", m.group(1)):
        fail.append("sync() inside an #if OLD_FENCE branch of carryfused.cl")

clang = shutil.which("clang")
if clang:
    def preprocess(defs):
        with tempfile.NamedTemporaryFile("w", suffix=".cl", delete=False) as f:
            f.write('#include "base.cl"\n')
            src = f.name
        cmd = [clang, "-x", "cl", "-cl-std=CL2.0", "-E", "-P", src, "-I", str(CL_DIR), "-DEXP=17000023u", "-DWIDTH=512u", "-DSMALL_HEIGHT=256u",
               "-DMIDDLE=2u", "-DNW=8u", "-DNH=4u", "-DCARRY_LEN=8u", "-DPFA_RADIX=0u", "-DMAXBPW=3998u", "-DFFT_VARIANT=202u", "-DFFT_TYPE=1",
               "-DFFT_FP64=0", "-DFFT_FP32=0", "-DNTT_GF31=1", "-DNTT_GF61=1", "-DWordSize=8"] + [f"-D{d}" for d in defs]
        p = subprocess.run(cmd, capture_output=True, text=True)
        pathlib.Path(src).unlink()
        return p.returncode, p.stdout, p.stderr

    for name, defs, want in (("nVidia sm_70", ["NVIDIAGPU=1", "CC=700"], True), ("nVidia sm_60", ["NVIDIAGPU=1", "CC=600"], True),
                             ("nVidia sm_50", ["NVIDIAGPU=1", "CC=500"], False), ("AMD", ["AMDGPU=1", "WAVEFRONT=64"], False), ("other", [], False)):
        rc, out, err = preprocess(defs)
        if rc != 0:
            fail.append(f"{name}: clang -E failed: {err[:300]}")
        elif ("bar.warp.sync" in out) != want:
            fail.append(f"{name}: bar.warp.sync {'missing from' if want else 'present in'} sync()")
else:
    print("SKIP (partial): clang not found, only the carryfused.cl text was checked")

if fail:
    print("\n".join(fail))
    sys.exit(1)
print(f"carryFused ready-flag sync(): {len(kernels)} variants ok")
