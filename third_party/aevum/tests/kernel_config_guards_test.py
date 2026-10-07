#!/usr/bin/env python3
"""Configurations the kernels cannot honour must fail to compile instead of silently computing a wrong result.

Compiles src/cl/base.cl alone with clang (-fsyntax-only) under a given FFT type and explicit -D overrides, the way a -use key or per-FFT
config line would deliver them.  Skipped when clang is not installed.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

CL_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "cl"
CLANG = shutil.which("clang")

FFT_TYPES = {
    "FFT64": dict(FFT_TYPE=0, FFT_FP64=1, FFT_FP32=0, NTT_GF31=0, NTT_GF61=0, WordSize=4),
    "FFT3161": dict(FFT_TYPE=1, FFT_FP64=0, FFT_FP32=0, NTT_GF31=1, NTT_GF61=1, WordSize=8),
    "FFT6431": dict(FFT_TYPE=51, FFT_FP64=1, FFT_FP32=0, NTT_GF31=1, NTT_GF61=0, WordSize=4),
    "FFT323161": dict(FFT_TYPE=4, FFT_FP64=0, FFT_FP32=1, NTT_GF31=1, NTT_GF61=1, WordSize=8),
}
COMMON = dict(EXP="17000023u", WIDTH="512u", SMALL_HEIGHT="256u", MIDDLE="2u", NW="8u", NH="4u", CARRY_LEN="8u", PFA_RADIX="0u",
              MAXBPW="3998u", FFT_VARIANT="202u")


def compiles(fft, extra=None):
    defs = dict(COMMON)
    defs.update(FFT_TYPES[fft])
    defs.update(extra or {})
    with tempfile.NamedTemporaryFile("w", suffix=".cl", delete=False) as f:
        f.write('#include "base.cl"\n')
        src = f.name
    cmd = [CLANG, "-x", "cl", "-cl-std=CL2.0", "-fsyntax-only", "-Xclang", "-cl-ext=+cl_khr_fp64", "-I", str(CL_DIR), "-DNO_ASM=1", src]
    cmd += [f"-D{k}={v}" for k, v in defs.items()]
    p = subprocess.run(cmd, capture_output=True, text=True)
    pathlib.Path(src).unlink()
    return p.returncode == 0, p.stderr


def error_lines(err):
    return "\n".join(l for l in err.splitlines() if "error" in l)[:600]


def expect(failures, name, fft, extra, ok_expected):
    ok, err = compiles(fft, extra)
    if ok != ok_expected:
        failures.append(f"{name}: {fft} {extra} {'failed to compile' if not ok else 'compiled but must be rejected'}\n{error_lines(err)}")


def main():
    if not CLANG:
        print("SKIP: clang not found")
        return 0
    failures = []
    # The baseline compiles at all (otherwise a rejection below proves nothing).
    for fft in FFT_TYPES:
        expect(failures, "baseline", fft, {}, True)

    # The fused weight butterfly only exists in the FFT64 carryFused.
    expect(failures, "FUSE_WEIGHT_BUTTERFLY=1 on FFT64", "FFT64", {"FUSE_WEIGHT_BUTTERFLY": 1}, True)
    for fft in ("FFT3161", "FFT6431", "FFT323161"):
        expect(failures, "FUSE_WEIGHT_BUTTERFLY=1", fft, {"FUSE_WEIGHT_BUTTERFLY": 1}, False)
        expect(failures, "FUSE_WEIGHT_BUTTERFLY=0", fft, {"FUSE_WEIGHT_BUTTERFLY": 0}, True)

    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("ok: unsupported kernel configurations are rejected at compile time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
