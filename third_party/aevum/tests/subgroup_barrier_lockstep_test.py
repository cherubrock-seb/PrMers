#!/usr/bin/env python3
"""bar(WG) and halfBar() may skip the barrier for a group no larger than a wavefront only where a wavefront runs in lock-step.

AMD (a lock-step wavefront) and NVIDIA before sm_60 skip it.  NVIDIA sm_60+ (Volta's Independent Thread Scheduling lets a
warp's lanes drift apart) must reconverge with sync() (bar.warp.sync) and order the LDS traffic with a local fence.  Every
other device (Intel, Apple, a CPU device under PoCL, NVIDIA without PTX) must take a real barrier: it used to do nothing
at all there.  Today only AEVUM_RADIX8_H256 on NVIDIA (G_H = 32) reaches the branch, but WAVEFRONT is a define, and a
shape with G_W or G_H <= WAVEFRONT would otherwise race silently.

Preprocesses src/cl/base.cl with clang for each device kind and inspects what bar(WG) and halfBar() do once WG (or half
the work-group) is <= WAVEFRONT.  Skipped when clang is not installed.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

CL_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "cl"
CLANG = shutil.which("clang")
COMMON = dict(EXP="17000023u", WIDTH="512u", SMALL_HEIGHT="256u", MIDDLE="2u", NW="8u", NH="4u", CARRY_LEN="8u",
              PFA_RADIX="0u", MAXBPW="3998u", FFT_VARIANT="202u", FFT_TYPE=1, FFT_FP64=0, FFT_FP32=0, NTT_GF31=1,
              NTT_GF61=1, WordSize=8)


def preprocess(extra, target):
    """base.cl preprocessed for these defines; it must also compile (-fsyntax-only) for the device's clang target."""
    defs = dict(COMMON)
    defs.update(extra)
    with tempfile.NamedTemporaryFile("w", suffix=".cl", delete=False) as f:
        f.write('#include "base.cl"\n')
        src = f.name
    base = [CLANG, "-x", "cl", "-cl-std=CL2.0", "-I", str(CL_DIR), src] + [f"-D{k}={v}" for k, v in defs.items()]
    p = subprocess.run(base + ["-E", "-P"], capture_output=True, text=True)
    c = subprocess.run(base + target + ["-fsyntax-only", "-Xclang", "-cl-ext=+cl_khr_fp64"], capture_output=True, text=True)
    pathlib.Path(src).unlink()
    if p.returncode:
        raise SystemExit(f"preprocessing failed for {extra}:\n{p.stderr[:800]}")
    if c.returncode:
        errors = "\n".join(l for l in c.stderr.splitlines() if "error" in l)
        raise SystemExit(f"base.cl does not compile for {extra}:\n{errors[:800]}")
    return p.stdout


def body(text, signature):
    i = text.find(signature)
    if i < 0:
        raise SystemExit(f"{signature} not found")
    i = text.index("{", i)
    depth = 0
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j]
    raise SystemExit(f"unbalanced {signature}")


def small_group_path(fn):
    """Statements executed when the group fits in a wavefront: the 'if (... <= WAVEFRONT)' block (bar) or what follows
    the 'if (halfSize > WAVEFRONT) { ...; return; }' guard (halfBar), up to a return."""
    fn = " ".join(fn.split())
    m = re.search(r"if \(WG <= \d+\) \{(.*?)\}(.*)$", fn)
    if m:   # bar(WG): the block, then (if it does not return) the rest of the function
        block, rest = m.group(1), m.group(2)
        return block if "return" in block else block + rest
    m = re.search(r"if \((?:halfSize|get_\w+\(0\) / 2) > \d+\) \{[^}]*\}(.*)$", fn)
    if m:   # halfBar(), or a bar(WG) that only acts for a group larger than a wavefront
        return m.group(1)
    m = re.search(r"if \(WG > \d+\) \{[^}]*\}(.*)$", fn)
    if m:
        return m.group(1)
    raise SystemExit("cannot find the small-group branch in:\n" + fn)


AMD = ["-target", "amdgcn-amd-amdhsa", "-mcpu=gfx906", "-nogpulib"]
NV = ["-target", "nvptx64-nvidia-nvcl"]
DEVICES = {
    # name: (defines, clang target, expectation for the small-group path)
    "AMD": (dict(AMDGPU=1, WAVEFRONT=64, NO_ASM=1), AMD, "skip"),
    "NVIDIA sm_86": (dict(NVIDIAGPU=1, CC=806), NV, "warp-sync"),
    "NVIDIA sm_61": (dict(NVIDIAGPU=1, CC=601), NV, "warp-sync"),
    "NVIDIA sm_52": (dict(NVIDIAGPU=1, CC=502), NV, "skip"),
    "NVIDIA without PTX": (dict(NVIDIAGPU=1, CC=806, NO_ASM=1), NV, "barrier"),
    "other (Intel, Apple, PoCL)": (dict(), [], "barrier"),
    "other, WAVEFRONT forced to 1024": (dict(WAVEFRONT=1024), [], "barrier"),
}


def main():
    if not CLANG:
        print("SKIP: clang not found")
        return 0
    failures = []
    for name, (defs, target, want) in DEVICES.items():
        text = preprocess(defs, target)
        for fn, sig in (("bar(WG)", "void __attribute__((overloadable)) bar(const u32 WG)"), ("halfBar()", "void halfBar()")):
            path = small_group_path(body(text, sig))
            has_barrier = re.search(r"\bbar\(\)|\bbarrier\(", path) is not None
            has_sync = "bar.warp.sync" in path or "sync()" in path
            has_fence = "mem_fence" in path
            got = "barrier" if has_barrier else ("warp-sync" if has_sync and has_fence else "skip")
            if got != want:
                failures.append(f"{name}: {fn} with a group <= WAVEFRONT does '{got}', expected '{want}': {path.strip()[:200]}")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("ok: sub-wavefront barriers are skipped only where a wavefront runs in lock-step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
